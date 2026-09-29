use crate::error::{DaggerError, Result};
use crate::llm::unified::{
    Block, Message, Role, StopReason, ToolDef, UnifiedRequest, UnifiedResponse, Usage,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// OpenAI Chat 协议 请求结构体
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    // 模型名称
    model: String,
    // 消息列表
    messages: Vec<ChatMessage>,
    // 工具列表
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ChatToolDef>,
    // 模型调用行为，我们基本使用默认值auto
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    // 单次请求模型最大输出
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    // 采样温度，coding场景下每种模型推荐值不一样
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    // 是否流式输出，一般agent工具会使用true
    // 采用流式输出一般是为了更好的交互效果
    stream: bool,
}

/// 正确响应
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
}

/// 错误响应
#[derive(Debug, Deserialize)]
struct ChatErrorBody {
    error: ChatErrorDetail,
}
#[derive(Debug, Deserialize)]
struct ChatErrorDetail {
    message: String,
}

/// Chat Completion协议下 消息模型
/// 单一协议对接下，直接简化取个大并集
#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    // system / user / assistant
    role: String,
    // 工具调用时，改值为null
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    // 仅 role = assitant 时携带，模型发起的工具调用
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ChatToolCall>,
    // 仅 role = tool 时携带，工具调用的结果
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    // 思考痕迹，非官方协议字段，deepseek等部分厂商支持
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

/// Chat Completion协议下 工具模型
#[derive(Debug, Serialize)]
struct ChatToolDef {
    // 固定值 function
    #[serde(rename = "type")]
    kind: String,
    function: ChatToolFunction,
}

/// function内容会真正模型转换的数据
#[derive(Debug, Serialize)]
struct ChatToolFunction {
    name: String,
    description: String,
    // 该值会最终转换为通用模型的 input_schema
    parameters: Value,
}

/// Chat Completion协议下 响应/请求 中 工具调用
#[derive(Debug, Serialize, Deserialize)]
struct ChatToolCall {
    id: String,
    #[serde(rename = "type", default = "default_tool_type")]
    kind: String,
    function: ChatToolCallFunc,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatToolCallFunc {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
    // 流式场景下，中间过程帧为null
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

// 工具定义 和 工具调用  type的默认值 function
fn default_tool_type() -> String {
    "function".to_string()
}

//-----------------------------------------------
// 类型转换：统一模型 -> OpenAI Chat Completions模型
//-----------------------------------------------

/// message转换
/// 该方法用户将 统一消息模型 转换为 OpenAI Chat 协议消息模型
/// 一般用在业务处理后，需要使用OpenAI Chat向模型发起请求
fn to_chat_messages(system: &Option<String>, messages: &[Message]) -> Vec<ChatMessage> {
    let mut out = Vec::new();

    // completions 协议系统提示词位于 message数组第一条
    if let Some(sys) = system {
        out.push(ChatMessage {
            role: "system".into(),
            content: Some(sys.into()),
            tool_calls: vec![],
            tool_call_id: None,
            reasoning_content: None,
        });
    }

    for msg in messages {
        match msg.role {
            Role::System | Role::Assistant | Role::User => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<ChatToolCall> = Vec::new();

                for block in &msg.content {
                    match block {
                        Block::Text { text } => text_parts.push(text.into()),
                        Block::Thinking { .. } => {
                            // 标准协议
                            // OpenAI Chat不处理思考内容
                        }
                        Block::ToolUse { id, name, input } => {
                            tool_calls.push(ChatToolCall {
                                id: id.into(),
                                kind: "function".into(),
                                function: ChatToolCallFunc {
                                    name: name.into(),
                                    arguments: serde_json::to_string(input)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                },
                            });
                        }
                        Block::ToolResult { .. } => {
                            // to_result 不会出现在 System / Assistant / User 类型消息中，直接忽略
                        }
                    }
                }
                let role = match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    _ => "system",
                };
                out.push(ChatMessage {
                    role: role.into(),
                    content: if text_parts.is_empty() && !tool_calls.is_empty() {
                        None
                    } else {
                        Some(text_parts.join(""))
                    },
                    tool_calls,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            Role::Tool => {
                for block in &msg.content {
                    if let Block::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } = block
                    {
                        let body = if *is_error {
                            format!("<tool_use_error>{content}</tool_user_error>")
                        } else {
                            content.into()
                        };
                        out.push(ChatMessage {
                            role: "tool".into(),
                            content: Some(body),
                            tool_calls: vec![],
                            tool_call_id: Some(tool_use_id.into()),
                            reasoning_content: None,
                        });
                    }
                }
            }
        }
    }
    out
}

/// 通用模型-工具 ——> OpenAI Chat 工具
fn to_chat_tool(tools: &[ToolDef]) -> Vec<ChatToolDef> {
    tools
        .iter()
        .map(|t| ChatToolDef {
            kind: "function".into(),
            function: ChatToolFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        })
        .collect()
}

//----------------------------------------------------
// 类型转换：OpenAI Chat Completions 协议模型 -> 统一模型
//----------------------------------------------------
fn from_chat_response(resp: ChatResponse) -> Result<UnifiedResponse> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| DaggerError::ParseResponse("响应中 choice 为空".into()))?;

    let mut blocks: Vec<Block> = Vec::new();

    if let Some(thinking) = choice.message.reasoning_content {
        if !thinking.is_empty() {
            blocks.push(Block::Thinking {
                thinking,
                signature: None,
            });
        }
    }

    if let Some(text) = choice.message.content {
        if !text.is_empty() {
            blocks.push(Block::Text { text });
        }
    }

    for call in choice.message.tool_calls {
        let input = serde_json::from_str(&call.function.arguments)
            .unwrap_or(Value::Object(serde_json::Map::new()));
        blocks.push(Block::ToolUse {
            id: call.id,
            name: call.function.name,
            input,
        });
    }

    let stop_reason = match choice.finish_reason.as_deref() {
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        _ => StopReason::Other,
    };

    let usage = resp
        .usage
        .map(|u| Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            ..Default::default()
        })
        .unwrap_or_default();

    Ok(UnifiedResponse {
        message: Message {
            role: Role::Assistant,
            content: blocks,
        },
        stop_reason,
        usage,
        response_id: None,
    })
}

// ----------------------------------
// Http 请求客户端
// ---------------------------------
pub struct OpenAIChatClient {
    http: Client,
    api_key: String,
    base_url: String,
}

impl OpenAIChatClient {
    pub fn new(api_key: impl Into<String>, base_url: Option<String>) -> Self {
        OpenAIChatClient {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
                .trim_end_matches("/")
                .to_string(),
        }
    }

    pub async fn complete(&self, req: &UnifiedRequest) -> Result<UnifiedResponse> {
        let body = ChatRequest {
            model: req.model.clone(),
            messages: to_chat_messages(&req.system, &req.messages),
            tools: to_chat_tool(&req.tools),
            tool_choice: if req.tools.is_empty() {
                None
            } else {
                Some("auto".into())
            },
            max_tokens: Some(req.max_tokens),
            temperature: req.temperature,
            stream: false,
        };
        let resp = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let message = serde_json::from_str::<ChatErrorBody>(&text)
                .map(|e| e.error.message)
                .unwrap_or(text);
            return Err(DaggerError::Api {
                status: status.as_u16(),
                message,
            });
        }

        let chat_resp: ChatResponse = resp.json().await?;
        from_chat_response(chat_resp)
    }
}
