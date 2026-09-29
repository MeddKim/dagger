use crate::error::{DaggerError, Result};
use crate::llm::unified::{
    Block, Message, Role, StopReason, ToolDef, UnifiedRequest, UnifiedResponse, Usage,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const ANTHROPIC_VERSION: &str = "2023-06-01";

///anthropic协议请求结构体
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    /// 顶层 system 字段（对比 OpenAI：messages 首条 / instructions）
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicToolDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// 扩展思考配置：{"type":"enabled","budget_tokens":N}
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct AnthropicThinking {
    #[serde(rename = "type")]
    kind: String, // "enabled"
    budget_tokens: u32,
}

/// Anthropic 消息：role 只有 user/assistant，content 是 block 数组
#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicBlock>,
}

/// 协议 block：与统一 Block 同构，但字段名按 Anthropic 规范
/// （如工具参数叫 input、结果关联叫 tool_use_id）。
/// 独立的结构体保证协议演进不影响统一层。
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        /// 直接是 JSON 对象（对比 OpenAI 的字符串化 arguments）
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

#[derive(Debug, Serialize)]
struct AnthropicToolDef {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Debug, Serialize)]
struct AnthropicToolChoice {
    #[serde(rename = "type")]
    kind: String, // "auto"
}

/// anthropic协议 响应结构体
#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicBlock>,
    /// end_turn / tool_use / max_tokens / stop_sequence
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
}

// Anthropic 错误体：{"type":"error","error":{"type":"...","message":"..."}}
#[derive(Debug, Deserialize)]
struct AnthropicErrBody {
    error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorDetail {
    message: String,
}

// =============================================
// 类型装换：统一模型 转换为 Anthropic协议模型
// =============================================
/// 统一模型Block 转换为 Anthropic模型 Block
fn to_anthropic_block(block: &Block) -> Option<AnthropicBlock> {
    match block {
        Block::Text { text } => Some(AnthropicBlock::Text { text: text.clone() }),
        Block::Thinking {
            thinking,
            signature,
        } => Some(AnthropicBlock::Thinking {
            thinking: thinking.clone(),
            // signature 必须原样回传，否则多轮请求被 400 拒绝
            signature: signature.clone(),
        }),
        Block::ToolUse { id, name, input } => Some(AnthropicBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(), // 对象直接放，无需字符串化
        }),
        // ToolResult 不在这里处理——它要搬进 user 消息，见下
        Block::ToolResult { .. } => None,
    }
}

/// 统一消息数组 → Anthropic messages。
///
/// 关键转换：
/// 1. Role::Tool 消息（统一层）→ 拆出 ToolResult 块，包成 user 消息；
/// 2. System 角色 → 已在顶层 system 字段，跳过；
/// 3. 合并相邻同角色消息（协议要求严格交替）。
fn to_anthropic_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
    let mut out: Vec<AnthropicMessage> = Vec::new();

    for msg in messages {
        // system 不进 messages（顶层字段处理）
        if msg.role == Role::System {
            continue;
        }

        // 统一层的 role → 协议 role：Tool 结果归入 user
        let role = match msg.role {
            Role::User | Role::Tool => "user",
            Role::Assistant => "assistant",
            Role::System => unreachable!(),
        };

        // 块转换（ToolResult 在此被正确映射为 tool_result block）
        let blocks: Vec<AnthropicBlock> = msg
            .content
            .iter()
            .map(|b| match b {
                Block::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => Some(AnthropicBlock::ToolResult {
                    tool_use_id: tool_use_id.clone(),
                    content: content.clone(),
                    is_error: *is_error,
                }),
                other => to_anthropic_block(other),
            })
            .flatten()
            .collect();

        if blocks.is_empty() {
            continue;
        }

        // 合并相邻同角色消息：协议要求 user/assistant 严格交替。
        // Agent 循环天然产生"user(工具结果) 紧跟 assistant"的交替，
        // 但压缩/恢复会话后可能出现连续同角色，这里兜底合并。
        if let Some(last) = out.last_mut() {
            if last.role == role {
                last.content.extend(blocks);
                continue;
            }
        }
        out.push(AnthropicMessage {
            role: role.into(),
            content: blocks,
        });
    }
    out
}

// 统一模型工具 -> Anthropic协议工具
fn to_anthropic_tools(tools: &[ToolDef]) -> Vec<AnthropicToolDef> {
    tools
        .iter()
        .map(|t| AnthropicToolDef {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect()
}

// ════════════════════════════════════════════════════════════
// Anthropic 协议 → 统一模型
// ════════════════════════════════════════════════════════════
fn from_anthropic_block(block: AnthropicBlock) -> Block {
    match block {
        AnthropicBlock::Text { text } => Block::Text { text },
        AnthropicBlock::Thinking {
            thinking,
            signature,
        } => Block::Thinking {
            thinking,
            signature,
        },
        AnthropicBlock::ToolUse { id, name, input } => Block::ToolUse { id, name, input },
        // 响应里不会出现 tool_result，防御性映射
        AnthropicBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => Block::ToolResult {
            tool_use_id,
            content,
            is_error,
        },
    }
}

fn from_anthropic_response(resp: AnthropicResponse) -> Result<UnifiedResponse> {
    let stop_reason = match resp.stop_reason.as_deref() {
        Some("end_turn") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        _ => StopReason::Other,
    };
    let usage = resp
        .usage
        .map(|u| Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_read_tokens: u.cache_read_input_tokens,
            cache_write_tokens: u.cache_creation_input_tokens,
        })
        .unwrap_or_default();

    Ok(UnifiedResponse {
        message: Message {
            role: Role::Assistant,
            content: resp.content.into_iter().map(from_anthropic_block).collect(),
        },
        stop_reason,
        usage,
        response_id: None,
    })
}

// ----------------------------------
// Http 请求客户端
// ---------------------------------
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    http: Client,
    api_key: String,
    base_url: String,
}

impl AnthropicClient {
    pub fn new(api_key: impl Into<String>, base_url: Option<String>) -> Self {
        AnthropicClient {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url
                .unwrap_or_else(|| "https://api.anthropic.com/v1".to_string())
                .trim_end_matches("/")
                .to_string(),
        }
    }

    pub async fn complete(&self, req: &UnifiedRequest) -> Result<UnifiedResponse> {
        let body = AnthropicRequest {
            model: req.model.clone(),
            max_tokens: req.max_tokens, // Anthropic 必填
            system: req.system.clone(),
            messages: to_anthropic_messages(&req.messages),
            tools: to_anthropic_tools(&req.tools),
            tool_choice: if req.tools.is_empty() {
                None
            } else {
                Some(AnthropicToolChoice {
                    kind: "auto".into(),
                })
            },
            temperature: req.temperature,
            // 扩展思考：预算给 max_tokens 的一半，上限 16000（经验值）
            thinking: req.thinking.then_some(AnthropicThinking {
                kind: "enabled".into(),
                budget_tokens: (req.max_tokens / 2).min(16000).max(1024),
            }),
            stream: false,
        };
        let resp = self
            .http
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let message = serde_json::from_str::<AnthropicErrBody>(&text)
                .map(|e| e.error.message)
                .unwrap_or(text);
            return Err(DaggerError::Api {
                status: status.as_u16(),
                message,
            });
        }

        let parsed: AnthropicResponse = resp.json().await?;
        from_anthropic_response(parsed)
    }
}
