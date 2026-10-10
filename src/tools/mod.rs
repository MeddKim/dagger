use std::{
    collections::HashMap,
    ffi::{OsStr, OsString},
    path::{Component, Path, PathBuf},
};

use crate::{error::Result, llm::unified::ToolDef};
use async_trait::async_trait;
use serde_json::Value;

#[derive(Default)]
pub struct ToolRegistry {
    // 工具合集
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.insert(tool.name(), Box::new(tool));
        self
    }

    ///加载所有定义好的工具
    pub fn load_tools(&self) -> Vec<ToolDef> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    pub fn get(&self, name: &String) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }

    /// 执行工具
    pub async fn execute(&self, name: &String, args: &Value, ctx: &ToolContext) -> Result<String> {
        let Some(tool) = self.get(name) else {
            return Ok(format!(
                "错误：未知工具\"{name}\"。可用工具：{}",
                self.tools.keys().cloned().collect::<Vec<_>>().join(", ")
            ));
        };

        if let Err(e) = validate_args(&tool.schema(), args) {
            return Ok(format!(
                "参数校验失败：{e} \n 工具：{} \n schema {}",
                tool.name(),
                tool.schema()
            ));
        }

        match tool.execute(args.clone(), ctx).await {
            Ok(out) => Ok(out.text),
            Err(e) => Err(e),
        }
    }
}

/// 工具定义
/// - 异步执行
/// - Send + Sync：保证可以跨await
#[async_trait]
pub trait Tool: Send + Sync {
    /// 工具名称
    fn name(&self) -> String;
    /// 工具描述
    fn description(&self) -> String;
    /// 参数
    fn schema(&self) -> Value;
    /// 执行工具
    ///     - 参数需要通过校验
    ///     - 工具自行处理业务异常（如文件不在），返回Ok(ToolOutput)以供模型处理
    ///     - Err 为工具异常，由Registry统一兜底处理
    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput>;

    fn definition(&self) -> ToolDef {
        ToolDef {
            name: self.name(),
            description: self.description(),
            input_schema: self.schema(),
        }
    }

    /// 当前工具是否只读工具，用于权限审批判定
    fn is_read_only(&self) -> bool {
        false
    }
}

/// 工具输出
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// 工具输出（需要喂回模型的文本）
    pub text: String,
    /// 处理输出被截断的情况
    pub truncated: bool,
}

impl ToolOutput {
    /// 快速构建普通输出
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            text: s.into(),
            truncated: false,
        }
    }
    ///超长输出统一截断处理
    pub fn truncated_text(s: impl Into<String>, max_byte: usize) -> Self {
        let s = s.into();
        if s.len() <= max_byte {
            return Self::text(s);
        }
        let mut end = max_byte;
        while !s.is_char_boundary(end) {
            end = end - 1;
        }
        Self {
            text: format!(
                "{} ...\n[输出过长已被截断：共 {} 字节，仅显示前 {} 字节，\
            请使用更精确的参数缩小范围]",
                &s[..end],
                s.len(),
                end
            ),
            truncated: true,
        }
    }
}

/// 工具运行上下文
/// 用于管理工具也行的共同信息
#[derive(Debug, Clone)]
pub struct ToolContext {
    // 工具的工作目录
    pub cwd: PathBuf,
}

impl ToolContext {
    /// 解析路径
    pub fn resolve(&self, path: &str) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() { p } else { self.cwd.join(p) }
    }
    /// 检查路径是否在工作目录内
    pub fn is_inside_workspace(&self, path: &Path) -> bool {
        // 统一路径的标准
        let abs = normalize_lenient(path);
        let root = normalize_lenient(&self.cwd);
        abs.starts_with(&root)
    }
}

fn normalize_lenient(path: &Path) -> PathBuf {
    let mut misssing: Vec<OsString> = Vec::new();
    let mut cur = path;
    let mut base = loop {
        if let Ok(c) = cur.canonicalize() {
            break c;
        }
        match (cur.file_name(), cur.parent()) {
            (Some(name), Some(parent)) => {
                misssing.push(name.to_os_string());
                cur = parent;
            }
            _ => return lexical_normalize(path),
        }
    };
    for name in misssing.iter().rev() {
        if name == OsStr::new(".") {
            continue;
        }
        if name == OsStr::new("..") {
            base.pop();
            continue;
        }
        base.push(name);
    }
    base
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn validate_args(schema: &Value, args: &Value) -> std::result::Result<(), String> {
    let Some(obj) = args.as_object() else {
        return Err("参数必须为 JSON 对象".into());
    };

    if let Some(required) = schema["required"].as_array() {
        let mut missing_keys = Vec::new();
        for key in required {
            let key = key.as_str().unwrap_or_default();
            if !obj.contains_key(key) {
                missing_keys.push(key);
            }
        }
        if !missing_keys.is_empty() {
            return Err(format!("缺少必填参数：\"{}\"", missing_keys.join(", ")));
        }
    }
    // 检查一级类型（properties 里声明了类型的字段）
    if let Some(props) = schema["properties"].as_object() {
        let mut error_type_params = Vec::new();
        for (key, prop) in props {
            if let Some(value) = obj.get(key) {
                let expected = prop["type"].as_str().unwrap_or_default();
                let ok = match expected {
                    "string" => value.is_string(),
                    "number" | "integer" => value.is_number(),
                    "boolean" => value.is_boolean(),
                    "array" => value.is_array(),
                    "object" => value.is_object(),
                    _ => true, // 未声明类型则放行
                };
                if !ok {
                    error_type_params.push(format!(
                        "参数 \"{key}\" 类型错误：期望 {expected}，实际 {value}"
                    ));
                }
            }
        }
        if !error_type_params.is_empty() {
            return Err(error_type_params.join("\n"));
        }
    }
    Ok(())
}

pub mod permission;

pub mod edit;
pub mod fs;
pub mod shell;

/// 初始化所有工具
pub fn standard_registry() -> ToolRegistry {
    ToolRegistry::new()
        .register(fs::ReadFile)
        .register(fs::WriteFile)
        .register(fs::Glob)
        .register(fs::Grep)
        .register(edit::EditFile)
        .register(shell::Bash)
}
