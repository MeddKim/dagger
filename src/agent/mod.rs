use futures_util::StreamExt;
use tokio::sync::mpsc;

use crate::error::{DaggerError, Result};
use crate::llm::unified::{
    Block, EventStream, Role, StopReason, StreamEvent, UnifiedRequest, UnifiedResponse,
};
use crate::tools::{ToolContext, ToolRegistry};
use crate::{llm::unified::Message, provider::Provider};

pub mod prompt;

/// 默认最大循环步数（安全阀）
const DEFAULT_MAX_STEPS: usize = 32;
const MAX_CONTINUE: usize = 2;

/// Agent的事件，用于和其他组件交互
/// Agent广播事件，需要的组件进行订阅并处理
#[derive(Debug, Clone)]
pub enum AgentEvent {
    Stream(StreamEvent),
    /// 工具开始执行
    ToolStarted {
        name: String,
        input: serde_json::Value,
    },
    /// 工具执行结束
    ToolFinished {
        name: String,
        output: String,
        is_error: bool,
    },
    /// 整个任务完成，此时获取最终文本答案
    Finished {
        answer: String,
    },
    /// 运行错误，此时终止任务处理循环
    Error(String),
}

// 发送的时间别名
pub type AgentEventTx = mpsc::Sender<AgentEvent>;
pub type AgentEventRx = mpsc::Receiver<AgentEvent>;

pub struct Agent {
    provider: Box<dyn Provider>,
    model: String,
    system_prompt: String,
    registry: ToolRegistry,
    max_steps: usize,
    messages: Vec<Message>,
    ctx: ToolContext,
}

impl Agent {
    pub fn new(
        provider: Box<dyn Provider>,
        model: String,
        system_prompt: String,
        registry: ToolRegistry,
        ctx: ToolContext,
    ) -> Self {
        Self {
            provider,
            model,
            system_prompt,
            registry,
            max_steps: DEFAULT_MAX_STEPS,
            messages: vec![],
            ctx,
        }
    }

    pub async fn run(
        &mut self,
        user_input: &str,
        event_tx: Option<AgentEventTx>,
    ) -> Result<String> {
        self.messages.push(Message::user(user_input));

        // 允许模型因为输出被长度截断而续写几次
        let mut continue_count = 0usize;

        // 一轮交互过程 ReAct 直至有最终答案或超过最大步数
        for _ in 1..=self.max_steps {
            let request = UnifiedRequest {
                model: self.model.clone(),
                system: Some(self.system_prompt.clone()),
                tools: self.registry.load_tools(),
                temperature: Some(0.2),
                max_tokens: 8192,
                messages: self.messages.clone(),
                thinking: false,
            };

            //Thought: 请求模型： 发起 Thought 过程
            let resp = match self.stream_once(&request, event_tx.as_ref()).await {
                Ok(response) => response,
                Err(e) => {
                    self.messages.push(Message {
                        role: Role::Tool,
                        content: vec![Block::ToolResult {
                            tool_use_id: "llm_error".into(),
                            content: format!(""),
                            is_error: true,
                        }],
                    });
                    if self.messages.len() <= 2 {
                        return Err(e);
                    }
                    continue;
                }
            };

            //将 assistant 消息放入messages中（可能是文本，也可能是工具调用）
            self.messages.push(resp.message.clone());

            //解析出需要执行的工具
            let tool_calls: Vec<(String, String, serde_json::Value)> = resp
                .message
                .tool_uses()
                .into_iter()
                .map(|(id, name, input)| (id.to_string(), name.to_string(), input.clone()))
                .collect();

            //Action 有工具调用的话执行工具调用
            if !tool_calls.is_empty() {
                // 调用工具并加入messages
                continue_count = 0; // 本次响应是工具调用，那之前的截断继续的计数重置
                self.execute_tools(tool_calls, event_tx.as_ref()).await;
                continue;
            }

            // Answer：没有工具调用，即已完成工作
            match resp.stop_reason {
                // 处理输出被截断：直接添加提示词，继续输出内容
                StopReason::MaxTokens if continue_count < MAX_CONTINUE => {
                    continue_count += 1;
                    self.messages
                        .push(Message::user("（输出内容被截断，请继续）"));
                    continue;
                }
                _ => {
                    let answer = resp.message.text();
                    Self::emit(
                        event_tx.as_ref(),
                        AgentEvent::Finished {
                            answer: answer.clone(),
                        },
                    )
                    .await;
                    return Ok(answer);
                }
            }
        }

        Err(crate::error::DaggerError::MaxStepsExceeded(self.max_steps))
    }

    //发起流式请求，处理agent完成事件
    async fn stream_once(
        &self,
        req: &UnifiedRequest,
        event_tx: Option<&AgentEventTx>,
    ) -> Result<UnifiedResponse> {
        let mut stream: EventStream = self.provider.complete_stream(req).await?;

        let mut final_resp: Option<UnifiedResponse> = None;

        while let Some(item) = stream.next().await {
            let ev = item?;
            if let StreamEvent::Done(resp) = &ev {
                final_resp = Some((**resp).clone());
            }
            Self::emit(event_tx, AgentEvent::Stream(ev)).await;
        }
        final_resp.ok_or_else(|| DaggerError::Parse("响应流结束但未接收到 Done 事件".into()))
    }

    async fn execute_tools(
        &mut self,
        calls: Vec<(String, String, serde_json::Value)>,
        event_tx: Option<&AgentEventTx>,
    ) {
        let mut results = Vec::new();

        for (id, name, input) in calls {
            // 发送工具执行时间
            Self::emit(
                event_tx,
                AgentEvent::ToolStarted {
                    name: name.clone(),
                    input: input.clone(),
                },
            )
            .await;

            let (output, is_error) = match self.registry.execute(&name, &input, &self.ctx).await {
                Ok(out) => (out, false),
                Err(e) => (format!("工具内部错误:{e}"), true),
            };

            // 发送工具执行完成事件
            Self::emit(
                event_tx,
                AgentEvent::ToolFinished {
                    name: name.clone(),
                    output: output.clone(),
                    is_error,
                },
            )
            .await;
            results.push(Block::ToolResult {
                tool_use_id: id,
                content: output,
                is_error,
            });
        }

        self.messages.push(Message {
            role: Role::Tool,
            content: results,
        });
    }

    async fn emit(tx: Option<&AgentEventTx>, ev: AgentEvent) {
        if let Some(tx) = tx {
            let _ = tx.send(ev).await;
        }
    }
}
