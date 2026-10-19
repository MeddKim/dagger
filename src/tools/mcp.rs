use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader},
    process::{Child, Command},
    sync::oneshot,
    task::JoinHandle,
};

use tokio::sync::Mutex as AsyncMutex;

use crate::{
    error::{DaggerError, Result},
    tools::{Tool, ToolContext, ToolOutput},
};

///单次RPC调用的超时时间
const RPC_TIMEOUT: Duration = Duration::from_secs(60);

///协议版本
const PROTOCOL_VERSION: &str = "2024-11-05";

/// 等待中的请求 id -> 响应通道发送端
type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>;

/// MCP 客户端。持有一个server子进程
pub struct McpClient {
    /// 请求ID计数器
    next_id: AtomicU64,
    /// 待响应表
    pending: PendingMap,
    /// 请求写入端
    writer: Arc<AsyncMutex<Box<dyn AsyncWrite + Unpin + Send>>>,
    /// 后台读任务句柄
    reader_task: JoinHandle<()>,
    /// server子进程
    _child: Option<Child>,
}

impl McpClient {
    /// 从任意异步读写对构造客户端
    ///
    /// 启动后台读任务：逐行读、按 id 路由响应、忽略通知。
    pub fn from_io<R, W>(reader: R, writer: W) -> Self
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let pending_for_task = pending.clone();

        let reader_task = tokio::spawn(async move {
            let mut lines = BufReader::new(reader).lines();
            loop {
                match lines.next_line().await {
                    // EOF：server 死了。清空 pending，让所有等待者立即拿到错误
                    Ok(None) => {
                        pending_for_task.lock().unwrap().clear();
                        break;
                    }
                    Ok(Some(line)) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let Ok(msg) = serde_json::from_str::<Value>(trimmed) else {
                            tracing::warn!(line = trimmed, "MCP 收到非 JSON 行，忽略");
                            continue;
                        };
                        // 只关心"带 id 的响应"；通知（无 id）仅记日志
                        if let Some(id) = msg.get("id").and_then(|v| v.as_u64()) {
                            let tx = pending_for_task.lock().unwrap().remove(&id);
                            if let Some(tx) = tx {
                                // 整个响应（含 result 或 error）交给调用方解析
                                let _ = tx.send(msg);
                            } else {
                                tracing::warn!(id, "MCP 收到无主的响应 id");
                            }
                        } else {
                            tracing::debug!(?msg, "MCP 通知（忽略）");
                        }
                    }
                    Err(e) => {
                        tracing::warn!("MCP 读取失败: {e}");
                        pending_for_task.lock().unwrap().clear();
                        break;
                    }
                }
            }
        });

        Self {
            next_id: AtomicU64::new(1),
            pending,
            writer: Arc::new(AsyncMutex::new(Box::new(writer))),
            reader_task,
            _child: None,
        }
    }
    /// 拉起 server 子进程并接管其 stdio
    pub async fn spawn(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<Self> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .envs(env)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            // server 的日志走 stderr，直接继承到我们的终端（traced 也能看到）
            .stderr(std::process::Stdio::inherit())
            // 客户端退出时杀掉 server，不留孤儿进程
            .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| DaggerError::Config(format!("MCP server 启动失败（{command}）: {e}")))?;

        let stdin = child.stdin.take().expect("已 piped");
        let stdout = child.stdout.take().expect("已 piped");

        let mut client = Self::from_io(stdout, stdin);
        client._child = Some(child);
        Ok(client)
    }

    /// 发送一条原始 JSON（请求或通知共用）。
    async fn write_line(&self, msg: &Value) -> Result<()> {
        let mut text = serde_json::to_string(msg)?;
        text.push('\n');
        let mut w = self.writer.lock().await;
        w.write_all(text.as_bytes()).await?;
        w.flush().await?; // 行缓冲协议：必须 flush，否则 server 收不到
        Ok(())
    }

    /// 发起 RPC 调用：注册 oneshot → 发请求 → 等响应（带超时）。
    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);

        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        // 写失败通常意味着 server 已死（BrokenPipe）→ 统一报连接中断
        if let Err(e) = self.write_line(&request).await {
            self.pending.lock().unwrap().remove(&id);
            return Err(DaggerError::Mcp(format!(
                "MCP 调用 {method} 失败：server 连接中断（{e}）"
            )));
        }

        // 等待响应：超时 or 通道被 clear（server 死亡）都要报错
        let msg = match tokio::time::timeout(RPC_TIMEOUT, rx).await {
            Ok(Ok(msg)) => msg,
            Ok(Err(_)) => {
                return Err(DaggerError::Mcp(format!(
                    "MCP 调用 {method} 失败：server 连接中断"
                )));
            }
            Err(_) => {
                self.pending.lock().unwrap().remove(&id);
                return Err(DaggerError::Mcp(format!(
                    "MCP 调用 {method} 超时（{} 秒）",
                    RPC_TIMEOUT.as_secs()
                )));
            }
        };

        // JSON-RPC 错误对象转换为自定义错误
        if let Some(err) = msg.get("error") {
            return Err(DaggerError::Mcp(format!(
                "MCP {method} 返回错误: {}",
                err.get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("未知")
            )));
        }
        Ok(msg.get("result").cloned().unwrap_or(Value::Null))
    }

    /// initialize 握手 + initialized 通知。任何其他调用前必须先做。
    pub async fn initialize(&self) -> Result<()> {
        let result = self
            .call(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {}, // 我们不需要 server 主动推送能力
                    "clientInfo": { "name": "dagger", "version": env!("CARGO_PKG_VERSION") },
                }),
            )
            .await?;
        tracing::info!(server = ?result.get("serverInfo"), "MCP 握手完成");

        // 握手完成通知（无 id，纯写入）
        self.write_line(&json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }))
        .await?;
        Ok(())
    }

    /// 发现 server 暴露的全部工具
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self.call("tools/list", json!({})).await?;
        let tools = result.get("tools").cloned().unwrap_or(Value::Array(vec![]));
        Ok(serde_json::from_value(tools)?)
    }

    /// 调用远端工具，把 content 数组拍平成文本
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let result = self
            .call(
                "tools/call",
                json!({ "name": name, "arguments": arguments }),
            )
            .await?;

        let is_error = result
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // content 是块数组：[{"type":"text","text":"..."}, ...]
        // 只提取 text 块（image 等多模态块暂不支持，标注跳过）
        let mut text = String::new();
        if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
            for block in content {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                            text.push_str(t);
                            text.push('\n');
                        }
                    }
                    other => {
                        text.push_str(&format!("[暂不支持的 MCP 内容类型: {other:?}]\n"));
                    }
                }
            }
        }
        if is_error {
            text = format!("MCP 工具返回错误：\n{text}");
        }
        Ok(text)
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.reader_task.abort(); // 停掉后台读任务（子进程由 kill_on_drop 处理）
    }
}

/// tools/list 返回的单个工具描述
#[derive(Debug, Clone, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "inputSchema", default)]
    pub input_schema: Value,
}

pub struct McpTool {
    client: Arc<McpClient>,
    /// 暴露出来的全名
    full_name: String,
    description: String,
    schema: Value,
    /// 远端原始名（调用时用）
    remote_name: String,
}

impl McpTool {
    pub fn new(client: Arc<McpClient>, server: &str, info: McpToolInfo) -> Self {
        Self {
            client,
            full_name: format!("mcp__{server}__{}", info.name),
            description: info
                .description
                .unwrap_or_else(|| format!("MCP 工具 {}（来自 {server}）", info.name)),
            schema: info.input_schema,
            remote_name: info.name,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> String {
        self.full_name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn schema(&self) -> Value {
        // server 给的 inputSchema 原样透传（模型层只认 JSON Schema，不 care 来源）
        self.schema.clone()
    }

    // 远端工具副作用未知 → 一律视为非只读，走权限审批
    fn is_read_only(&self) -> bool {
        false
    }

    async fn execute(&self, args: Value, _ctx: &ToolContext) -> Result<ToolOutput> {
        let text = self.client.call_tool(&self.remote_name, args).await?;
        Ok(ToolOutput::truncated_text(text, 32 * 1024))
    }
}

// ────────────────────────────────────────────────────────────
// 配置加载与一键挂载
// ────────────────────────────────────────────────────────────

/// .dagger/mcp.json 里一个 server 的配置（兼容 Claude Desktop 的配置形状）
#[derive(Debug, Deserialize)]
pub struct McpServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct McpConfigFile {
    #[serde(rename = "mcpServers", default)]
    servers: HashMap<String, McpServerConfig>,
}

/// 加载 MCP 配置：项目级 .dagger/mcp.json 优先，其次 ~/.dagger/mcp.json
pub fn load_configs(project_root: &std::path::Path) -> HashMap<String, McpServerConfig> {
    let candidates = [
        project_root.join(".dagger/mcp.json"),
        // 用户级配置：~/.dagger/mcp.json（与会话存储同约定）
        directories::BaseDirs::new()
            .map(|d| d.home_dir().join(".dagger/mcp.json"))
            .unwrap_or_default(),
    ];
    for path in &candidates {
        if let Ok(text) = std::fs::read_to_string(path) {
            match serde_json::from_str::<McpConfigFile>(&text) {
                Ok(cfg) => {
                    tracing::info!(path = %path.display(), "加载 MCP 配置");
                    return cfg.servers;
                }
                Err(e) => tracing::warn!(path = %path.display(), "MCP 配置解析失败: {e}"),
            }
        }
    }
    HashMap::new()
}

/// 已挂载的 MCP server：持有 client（保活子进程）+ 工具定义（供 registry 重建）
#[derive(Clone)]
pub struct MountedMcp {
    pub client: Arc<McpClient>,
    pub server_name: String,
    pub tools: Vec<McpToolInfo>,
}

impl MountedMcp {
    /// 把本 server 的所有工具注册进 registry（同步，供计划模式切回执行时重建）
    pub fn register_into(
        &self,
        mut registry: crate::tools::ToolRegistry,
    ) -> crate::tools::ToolRegistry {
        for info in &self.tools {
            registry = registry.register(McpTool::new(
                self.client.clone(),
                &self.server_name,
                info.clone(),
            ));
        }
        registry
    }
}

/// 连接全部配置的 server，把工具注册进 registry。
///
/// 返回 (注册表, 已挂载 server 列表)——调用方必须持有已挂载列表（或注册表），
/// 否则 drop 时 server 子进程会被杀。
/// 单个 server 失败只警告不致命：MCP 是增强，不该拖垮启动。
pub async fn mount_all(
    mut registry: crate::tools::ToolRegistry,
    project_root: &std::path::Path,
) -> (crate::tools::ToolRegistry, Vec<MountedMcp>) {
    let mut mounted = Vec::new();
    for (name, cfg) in load_configs(project_root) {
        let result = async {
            let client = McpClient::spawn(&cfg.command, &cfg.args, &cfg.env).await?;
            client.initialize().await?;
            let tools = client.list_tools().await?;
            Ok::<_, DaggerError>((client, tools))
        }
        .await;

        match result {
            Ok((client, tools)) => {
                let m = MountedMcp {
                    client: Arc::new(client),
                    server_name: name,
                    tools,
                };
                tracing::info!(server = %m.server_name, count = m.tools.len(), "MCP server 已挂载");
                registry = m.register_into(registry);
                mounted.push(m);
            }
            Err(e) => tracing::warn!(server = %name, "MCP server 挂载失败（跳过）: {e}"),
        }
    }
    (registry, mounted)
}
