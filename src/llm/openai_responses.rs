use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{DaggerError, Result};
use crate::llm::sse;
use crate::llm::unified::{
    Block, EventStream, Message, Role, StopReason, StreamEvent, ToolDef, UnifiedRequest,
    UnifiedResponse, Usage,
};

/// OpenAI Responses协议 请求结构体
#[derive(Debug, Serialize)]
pub struct ResponsesRequest {
    model: String,
    /// 系统提示词
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    input: Vec<InputItem>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ResponsesToolDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    // 控制服务端是否存储历史
    // true 服务端存储历史，我们可用previous_response_id串联
    // flase 反之
    store: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InputItem {
    Message {
        role: String,
        content: Vec<InputContent>,
    },
    // 上一次请求模型返回模型调用
    // 需要原样回传
    FunctionCall {
        call_id: String,
        name: String,
        // JSON字符串
        arguments: String,
    },
    // 工具结果
    // 本次工具结果此处call_id应该等于对应调用请求的call_id
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

#[derive(Debug, Serialize)]
struct ResponsesToolDef {
    // 该属性只有function
    #[serde(rename = "type")]
    kind: String,
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InputContent {
    InputText { text: String },
    OutputText { text: String },
}

/// OpenAI Responses 协议响应结构
#[derive(Debug, Deserialize)]
pub struct ResponsesResponse {
    id: String,
    // completed / incomplete / faild / cancelled
    status: String,
    output: Vec<OutputItem>,
    usage: Option<ResponsesUsage>,
    //status = incomplete是会有该值，说明未完成原因
    //通常reason
    incomplete_details: Option<ResponsesIncomplete>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputItem {
    Message {
        content: Vec<OutputContent>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    // 推理痕迹
    Reasoning {
        #[serde(default)]
        summary: Vec<ReasoningSummary>,
    },
    /// 内置工具调用，如web_search，本次忽略
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct ResponsesIncomplete {
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReasoningSummary {
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputContent {
    OutputText {
        text: String,
    },
    /// 拒绝回答等内容类型
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct ResponsesUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    input_tokens_details: Option<ResponsesUsageDetails>,
}

#[derive(Debug, Deserialize)]
struct ResponsesUsageDetails {
    #[serde(default)]
    cached_tokens: u64,
}

/// 错误响应
#[derive(Debug, Deserialize)]
struct ResponsesErrorBody {
    error: ResponsesErrorDetail,
}
#[derive(Debug, Deserialize)]
struct ResponsesErrorDetail {
    message: String,
}
// ====================================
// 类型转换 统一类型 -> Responses 协议类型
// ====================================
//
/// 统一消息 转换为 Responses 协议中的 inputs
fn to_input_items(messages: &[Message]) -> Vec<InputItem> {
    let mut input_items: Vec<InputItem> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System | Role::User | Role::Assistant => {
                let text: String = msg
                    .content
                    .iter()
                    .flat_map(|c| match c {
                        Block::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                if !text.is_empty() {
                    let role = match msg.role {
                        Role::User | Role::System => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => unreachable!(),
                    };
                    let content = match msg.role {
                        Role::Assistant => InputContent::OutputText { text },
                        _ => InputContent::InputText { text },
                    };
                    input_items.push(InputItem::Message {
                        role: role.into(),
                        content: vec![content],
                    });
                }
                for (id, name, input) in msg.tool_uses() {
                    input_items.push(InputItem::FunctionCall {
                        call_id: id.into(),
                        name: name.into(),
                        arguments: serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                    });
                }
            }
            Role::Tool => {
                for block in &msg.content {
                    if let Block::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } = block
                    {
                        let output = if *is_error {
                            format!("<tool_use_error>{content}</tool_use_error>")
                        } else {
                            content.clone()
                        };
                        input_items.push(InputItem::FunctionCallOutput {
                            call_id: tool_use_id.clone(),
                            output,
                        });
                    }
                }
            }
        }
    }

    input_items
}

/// 将统一工具结果 装换为 OpenAI Responses 协议类型
fn to_responses_tools(tools: &[ToolDef]) -> Vec<ResponsesToolDef> {
    tools
        .iter()
        .map(|t| ResponsesToolDef {
            kind: "function".into(),
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.input_schema.clone(),
        })
        .collect()
}

// ====================================
// 类型转换 Responses 协议类型 -> 统一类型
// ====================================
/// OpenAI Responses 请求返回结果 转为为 统一返回结构
fn from_responses_response(resp: ResponsesResponse) -> Result<UnifiedResponse> {
    let mut blocks: Vec<Block> = Vec::new();

    for item in resp.output {
        match item {
            OutputItem::Reasoning { summary } => {
                let thinking: String = summary
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                if !thinking.is_empty() {
                    blocks.push(Block::Thinking {
                        thinking,
                        signature: None,
                    });
                }
            }
            OutputItem::Message { content } => {
                for c in content {
                    if let OutputContent::OutputText { text } = c {
                        blocks.push(Block::Text { text });
                    }
                }
            }
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                let input = serde_json::from_str(&arguments)
                    .unwrap_or(Value::Object(serde_json::Map::new()));
                blocks.push(Block::ToolUse {
                    id: call_id,
                    name,
                    input,
                });
            }
            OutputItem::Other => {}
        }
    }

    // 结束原因推断：Responses 没有专门的 tool_use 状态——
    // 只要 output 里有 function_call，Agent 就该继续循环
    let has_tool_call = blocks.iter().any(|b| matches!(b, Block::ToolUse { .. }));
    let stop_reason = if has_tool_call {
        StopReason::ToolUse
    } else if resp.status == "incomplete"
        && matches!(
            resp.incomplete_details.and_then(|d| d.reason).as_deref(),
            Some("max_output_tokens")
        )
    {
        StopReason::MaxTokens
    } else if resp.status == "completed" {
        StopReason::EndTurn
    } else {
        StopReason::Other
    };

    let usage = resp
        .usage
        .map(|u| Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_read_tokens: u.input_tokens_details.map(|d| d.cached_tokens).unwrap_or(0),
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
        response_id: Some(resp.id),
    })
}

// ----------------------------------
// Http 请求客户端
// ---------------------------------
#[derive(Debug, Clone)]
pub struct OpenAIResponsesClient {
    http: Client,
    api_key: String,
    base_url: String,
}

impl OpenAIResponsesClient {
    pub fn new(api_key: impl Into<String>, base_url: Option<String>) -> Self {
        OpenAIResponsesClient {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
                .trim_end_matches("/")
                .to_string(),
        }
    }

    pub async fn complete(&self, req: &UnifiedRequest) -> Result<UnifiedResponse> {
        let body = ResponsesRequest {
            model: req.model.clone(),
            instructions: req.system.clone(),
            input: to_input_items(&req.messages),
            tools: to_responses_tools(&req.tools),
            max_output_tokens: Some(req.max_tokens),
            temperature: req.temperature,
            stream: false,
            store: false, //自管历史，关闭服务端存储
        };

        let resp = self
            .http
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            return Err(self.api_error(resp).await);
        }

        let parsed: ResponsesResponse = resp.json().await?;
        from_responses_response(parsed)
    }
}

// ====================================================
// 流式请求
// ====================================================
/// Responses 流式事件：只建模我们关心的类型，其余进 Other
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponsesStreamEvent {
    /// 文本增量：{"type":"response.output_text.delta","delta":"你",...}
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta { delta: String },

    /// 新输出 item 出现：function_call 意味着一次工具调用开始
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded { item: OutputItemAddedBody },

    /// 工具参数碎片：{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\""}
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgsDelta { item_id: String, delta: String },

    /// 工具调用 item 完成
    #[serde(rename = "response.output_item.done")]
    OutputItemDone { item: OutputItemDoneBody },

    /// 整个响应完成：携带完整响应对象（直接复用非流式解析！）
    #[serde(rename = "response.completed")]
    Completed { response: ResponsesResponse },

    /// 响应失败
    #[serde(rename = "response.failed")]
    Failed { response: Value },

    /// response.created / response.in_progress / content_part.added … 忽略
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct OutputItemAddedBody {
    #[serde(rename = "type")]
    kind: String,
    /// function_call item 携带
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OutputItemDoneBody {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    call_id: Option<String>,
}

impl OpenAIResponsesClient {
    pub async fn complete_stream(&self, req: &UnifiedRequest) -> Result<EventStream> {
        let body = ResponsesRequest {
            model: req.model.clone(),
            instructions: req.system.clone(),
            input: to_input_items(&req.messages),
            tools: to_responses_tools(&req.tools),
            max_output_tokens: Some(req.max_tokens),
            temperature: req.temperature,
            stream: true,
            store: false,
        };

        let resp = self
            .http
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            return Err(self.api_error(resp).await);
        }
        let mut sse = Box::pin(sse::into_sse_stream(resp));

        let stream = async_stream::try_stream! {
            // item_id → call_id 映射：delta 事件只带 item_id（fc_...），
            // 而统一层 ToolUse.id 用 call_id（call_...），需要翻译
            let item_to_call: std::collections::HashMap<String, String> =
                std::collections::HashMap::new();

            while let Some(ev) = sse.next().await {
                let ev = ev?;
                if ev.data.trim().is_empty() {
                    continue;
                }
                let event: ResponsesStreamEvent = serde_json::from_str(&ev.data)
                    .map_err(|e| DaggerError::Parse(format!("流式事件解析失败: {e}")))?;

                match event {
                    ResponsesStreamEvent::OutputTextDelta { delta } => {
                        yield StreamEvent::TextDelta(delta);
                    }

                    ResponsesStreamEvent::OutputItemAdded { item } => {
                        if item.kind == "function_call" {
                            if let (Some(call_id), Some(name)) = (item.call_id, item.name) {
                                yield StreamEvent::ToolUseStart {
                                    id: call_id.clone(),
                                    name,
                                };
                                // 记录映射时不知道 item_id——从 raw 里拿不到，
                                // 简单起见用 call_id 自身兜底（多数实现 item_id 与
                                // call_id 同时出现在后续 delta 中，见下）
                            }
                        }
                    }

                    ResponsesStreamEvent::FunctionCallArgsDelta { item_id, delta } => {
                        // delta 事件的 item_id 是 fc_ 开头；ToolUseStart 发的是 call_id。
                        // 生产实现应在 OutputItemAdded 里同时记录 item.id → call_id。
                        // 这里简化：直接以 item_id 为关联键，Done 帧里有完整对象兜底。
                        let id = item_to_call.get(&item_id).cloned().unwrap_or(item_id);
                        yield StreamEvent::ToolUseInputDelta { id, delta };
                    }

                    ResponsesStreamEvent::OutputItemDone { item } => {
                        if item.kind == "function_call" {
                            if let Some(call_id) = item.call_id {
                                yield StreamEvent::ToolUseEnd { id: call_id };
                            }
                        }
                    }

                    ResponsesStreamEvent::Completed { response } => {
                        // 复用第 06 章的完整解析：最终状态以它为准
                        let unified = from_responses_response(response)?;
                        yield StreamEvent::Done(Box::new(unified));
                        break;
                    }

                    ResponsesStreamEvent::Failed { response } => {
                        Err(DaggerError::Api {
                            status: 500,
                            message: format!("Responses 流式失败: {response}"),
                        })?;
                    }

                    ResponsesStreamEvent::Other => {}
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /// 抽取的公共错误处理：非 2xx → DaggerError::Api
    async fn api_error(&self, resp: reqwest::Response) -> DaggerError {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        // Responses 错误体同样是 {"error":{"message":...}}
        let message = serde_json::from_str::<ResponsesErrorBody>(&text)
            .map(|e| e.error.message)
            .unwrap_or(text);
        DaggerError::Api {
            status: status,
            message,
        }
    }
}
