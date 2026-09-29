use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 统一请求模型
#[derive(Debug, Clone)]
pub struct UnifiedRequest {
    // 模型名
    pub model: String,
    // 系统提示词
    pub system: Option<String>,
    // 交互消息集
    pub messages: Vec<Message>,
    // 工具集
    pub tools: Vec<ToolDef>,
    // 最大输出token
    pub max_tokens: u32,
    // 采样温度
    pub temperature: Option<f32>,
    // 是否开启思考
    pub thinking: bool,
}
/// 统一响应模型
pub struct UnifiedResponse {
    /// LLM返回的消息集
    pub message: Message,
    /// 结束原因
    pub stop_reason: StopReason,
    /// token消耗信息
    pub usage: Usage,
    /// Responses API 的响应id，用于多轮会话的 previous_response_id 串联
    /// 其他协议为 None
    pub response_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    // 消息角色
    pub role: Role,
    // 消息内容
    pub content: Vec<Block>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    /// 普通文本类型
    Text { text: String },
    /// 模型发起的工具调用
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// 工具的执行结果
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
    /// 模型推理思考痕迹
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

/// 工具描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    /// 工具名称
    pub name: String,
    /// 工具描述
    pub description: String,
    /// 参数说明 OpenAI协议中的tool.function.parameters / Anthropic协议中的tool.input_schema
    pub input_schema: Value,
}

/// 结束原因
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// 正常说完（ stop / end_turn / completed ）
    EndTurn,
    /// 工具调用（ tool_calls / tool_use ）
    ToolUse,
    /// 达到 max_tokens 被截断 ( length / max_tokens )
    MaxTokens,
    /// 其他
    Other,
}

/// token 用量，统一为四个维度
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// 命中缓存的输入token
    pub cache_read_tokens: u64,
    /// 写入缓存的输入token
    pub cache_write_tokens: u64,
}

//-----------------------------------
// 模型方法
// ----------------------------------
impl Message {
    // 便捷构建：用户消息
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![Block::Text { text: text.into() }],
        }
    }
    // 便捷构建：模型消息
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Block::Text { text: text.into() }],
        }
    }
    // 提取消息里面的全部工具调用
    // 该方法一般用于处理模型响应
    // 该方法返回元组(id, 方法名称，方法参数)
    pub fn tool_uses(&self) -> Vec<(&str, &str, &Value)> {
        self.content
            .iter()
            .filter_map(|m| match m {
                Block::ToolUse { id, name, input } => Some((id.as_str(), name.as_str(), input)),
                _ => None,
            })
            .collect()
    }

    // 提取消息中所有文本类型的内容
    // 拼接到一起
    // 用于展示或最终答案的提取
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|m| match m {
                Block::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}
