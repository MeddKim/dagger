use crate::{
    agent::Agent,
    error::{DaggerError, Result},
    provider::{ProviderConfig, build_provider},
    tools::{self, Tool, ToolContext, ToolOutput, ToolRegistry, permission::PermissionManager},
};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};

/// 子 agent 整体超时时间（10min），防止子任务空转
const SUBAGENT_TIMEOUT: Duration = Duration::from_secs(600);

/// 子 Agent 最大迭代步数：子Agent任务更聚焦，步数应该更小
const SUBAGENT_MAX_STEPS: usize = 16;

/// 子 Agent 报告的输出上限
const MAX_REPORT_BYTES: usize = 16 * 1024;

// 子agent类型
enum SubagentKind {
    // 浏览型：做只读浏览
    Explore,
    // 通用型：可读写
    General,
}

impl SubagentKind {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "explore" => Ok(Self::Explore),
            "general" => Ok(Self::General),
            other => Err(DaggerError::Config(format!(
                "未知 agent_type \"{other}\" （可选：explore / general）"
            ))),
        }
    }

    fn name(&self) -> String {
        match self {
            Self::Explore => "explore".into(),
            Self::General => "general".into(),
        }
    }

    /// 专门化系统提示词：角色鲜明才能行为稳定
    fn system_prompt(&self) -> String {
        match self {
            Self::Explore => {
                r#"你是一个代码库探索子 Agent。你的任务是按主 Agent 的指令调查代码库并汇报结论。

                    纪律：
                    - 你只有只读工具（read_file / glob / grep），不能也不应尝试修改任何文件；
                    - 探索要穷举：宁可多 grep 几次确认，不要猜；
                    - 报告必须给出证据：每个结论附带 文件路径:行号；
                    - 报告控制在 500 字以内：先说结论，再列证据，最后指出你不确定的部分。"#
                    .to_string()
            }

            Self::General => r#"你是一个通用子 Agent，负责完成主 Agent 委派的一个独立子任务。

                    纪律：
                    - 只做被委派的事，完成后立即给出报告，不要发散；
                    - 修改文件前先读懂相关代码；有测试就跑测试验证；
                    - 遇到无法解决的阻塞，报告"做不到 + 原因 + 建议"，不要硬撑；
                    - 报告聚焦"我做了什么、结果如何、主 Agent 需要知道什么"。"#
                .to_string(),
        }
    }

    /// 加载子agent对应的工具集
    fn registry(&self) -> ToolRegistry {
        match self {
            Self::Explore => tools::expore_registry(),
            Self::General => tools::standard_registry(),
        }
    }
}

/// Task工具
/// 用户将一个子任务委派给上下文隔离的子 Agent
/// provider_cfg 使用主 Agent 复制的 配置
/// permission 使用主 Agent 共享的权限管理器，保证用户的选择一直可用
pub struct Task {
    provider_cfg: ProviderConfig,
    permissions: Arc<PermissionManager>,
}

impl Task {
    pub fn new(provider_cfg: ProviderConfig, permissions: Arc<PermissionManager>) -> Self {
        Self {
            provider_cfg,
            permissions,
        }
    }
}

/// 将子agent注册为工具，提供给主agent使用
#[async_trait]
impl Tool for Task {
    fn name(&self) -> String {
        "task".into()
    }
    fn description(&self) -> String {
        "把一个子任务委派给上下文隔离的子 Agent。\
         适用：需要多次搜索/阅读的探索（agent_type=explore，只读），\
         或边界清晰的独立子任务（agent_type=general，可读写）。\
         子 Agent 的探索过程不占用主对话上下文，只有最终报告返回。\
         提供 description（3-5 词概括）和 prompt（给子 Agent 的完整指令，\
         它看不到主对话，指令必须自包含）。"
            .into()
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "3-5 词的子任务概括，用于 UI 展示与日志"
                },
                "prompt": {
                    "type": "string",
                    "description": "给子 Agent 的完整任务指令。子 Agent 看不到主对话历史，指令必须自包含：背景、目标、相关路径、期望的报告格式"
                },
                "agent_type": {
                    "type": "string",
                    "enum": ["explore", "general"],
                    "description": "explore=只读探索汇报；general=可执行完整子任务"
                }
            },
            "required": ["description", "prompt", "agent_type"]
        })
    }

    fn is_read_only(&self) -> bool {
        false
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let description = args["description"].as_str().unwrap_or("子任务").to_string();
        let prompt = args["prompt"].as_str().unwrap_or_default().to_string();
        let kind = SubagentKind::parse(args["agent_type"].as_str().unwrap_or("explore"))?;

        tracing::info!(agent_type = kind.name(), %description, "子agent启动");

        let provider = build_provider(&self.provider_cfg);
        let mut child = Agent::new(
            provider,
            self.provider_cfg.model.clone(),
            kind.system_prompt(),
            kind.registry(),
            ctx.clone(),
        )
        .with_max_steps(SUBAGENT_MAX_STEPS)
        .with_permissions(self.permissions.clone());

        let result = tokio::time::timeout(SUBAGENT_TIMEOUT, child.run(&prompt, None)).await;

        let report = match result {
            Ok(Ok(answer)) => answer,
            Ok(Err(e)) => format!("子 agent 执行失败: {e}"),
            Err(_) => format!(
                "子 Agent 超时（{}秒）被强制终止，未产出结论。",
                SUBAGENT_TIMEOUT.as_secs(),
            ),
        };

        let tool_calls = child.messages().iter().flat_map(|m| m.tool_uses()).count();

        tracing::info!(agent_type = kind.name(), tool_calls, "子ageng执行完成");

        Ok(ToolOutput::truncated_text(
            format!(
                "[子 Agent 报告 · {} · {} · {} · 次工具调用\n{report}]",
                kind.name(),
                description,
                tool_calls,
            ),
            MAX_REPORT_BYTES,
        ))
    }
}
