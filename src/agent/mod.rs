use crate::error::Result;
use crate::llm::unified::{Block, Role, StopReason, UnifiedRequest};
use crate::tools::ToolRegistry;
use crate::{llm::unified::Message, provider::Provider};

/// 默认最大循环步数（安全阀）
const DEFAULT_MAX_STEPS: usize = 32;

pub struct Agent {
    provider: Box<dyn Provider>,
    model: String,
    system_prompt: String,
    registry: ToolRegistry,
    max_steps: usize,
    messages: Vec<Message>,
}

impl Agent {
    pub fn new(
        provider: Box<dyn Provider>,
        model: String,
        system_prompt: String,
        registry: ToolRegistry,
    ) -> Self {
        Self {
            provider,
            model,
            system_prompt,
            registry,
            max_steps: DEFAULT_MAX_STEPS,
            messages: vec![],
        }
    }

    pub async fn run(&mut self, user_input: &str) -> Result<String> {
        self.messages.push(Message::user(user_input));

        // 一轮交互过程 ReAct 直至有最终答案或超过最大步数
        for _ in 1..=self.max_steps {
            let request = UnifiedRequest {
                model: self.model.clone(),
                system: Some(self.system_prompt.clone()),
                tools: self.registry.load_tools(),
                temperature: Some(0.7),
                max_tokens: 8192,
                messages: self.messages.clone(),
                thinking: false,
            };

            //Thought: 请求模型： 发起 Thought 过程
            let resp = self.provider.complete(&request).await?;

            //将 assistant 消息放入messages中（可能是文本，也可能是工具调用）
            self.messages.push(resp.message.clone());

            //解析看看是否有工具调用
            let tool_calls: Vec<(String, String, serde_json::Value)> = resp
                .message
                .tool_uses()
                .into_iter()
                .map(|(id, name, input)| (id.to_string(), name.to_string(), input.clone()))
                .collect();

            //Action 有工具调用的话执行工具调用
            if !tool_calls.is_empty() {
                // 调用工具并加入messages
                let mut content: Vec<Block> = Vec::new();
                for (id, name, input) in tool_calls {
                    let tool_result = self
                        .registry
                        .execute(&name, &input)
                        .await
                        .unwrap_or("未找到工具".into());
                    content.push(Block::ToolResult {
                        tool_use_id: id.clone(),
                        content: tool_result,
                        is_error: false,
                    });
                }
                //Observation 将工具调用结果回传模型，对应：Observation
                self.messages.push(Message {
                    role: Role::Tool,
                    content,
                });
                continue;
            }

            // Answer：没有工具调用，即已完成工作，对应
            match resp.stop_reason {
                StopReason::MaxTokens => {
                    self.messages
                        .push(Message::user("（输出内容被截断，请继续）"));
                    continue;
                }
                _ => {
                    let answer = resp.message.text();
                    return Ok(answer);
                }
            }
        }

        Err(crate::error::DaggerError::MaxStepsExceeded(self.max_steps))
    }
}
