use crate::llm::unified::{Block, Message, Role};

pub mod compact;
pub mod tokens;

///上下文预算配置
#[derive(Debug, Clone)]
pub struct ContextBudget {
    /// 模型上下文窗口总大小
    pub window: usize,
    /// 输出（max_tokens）
    pub output_reserve: usize,
    /// 压缩触发阈值
    pub compress_threshold: f64,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            window: 128000,
            output_reserve: 8192,
            compress_threshold: 0.8,
        }
    }
}

impl ContextBudget {
    // 可用预算
    pub fn available_buget(&self) -> usize {
        ((self.window as f64 * self.compress_threshold) as usize)
            .saturating_sub(self.output_reserve)
    }

    // 当前输出上下文是否超限
    pub fn exceeds(&self, messages: &[Message]) -> bool {
        tokens::count_messages(messages) > self.available_buget()
    }
}

/// 强制截断上下文
///     1. 对于超过500字符的ToolResult 截断（最近recent条不动）
///     2. 整轮丢弃最老对话 （注意要保证协议完整 antropic的 assistant和其tool result需要同时丢弃）
///     3. 永远保留第 0 条和 最近 keep_recent条
pub fn truncate(messages: &[Message], budget: usize, keep_recent: usize) -> Vec<Message> {
    let mut msgs = messages.to_vec();
    // ToolResult瘦身（最近recent条不动）
    let self_zero_start = msgs.len().saturating_sub(keep_recent);
    for msg in msgs[..self_zero_start].iter_mut() {
        if msg.role == Role::Tool {
            for block in msg.content.iter_mut() {
                if let Block::ToolResult { content, .. } = block {
                    if content.len() > 500 {
                        let mut end = 500;
                        while !content.is_char_boundary(end) {
                            end -= 1;
                        }
                        content.truncate(end);
                        content.push_str("..[历史工具输出已被瘦身]");
                    }
                }
            }
        }
    }
    if tokens::count_messages(&msgs) <= budget {
        return msgs;
    }
    // 整轮丢弃老对话（保留0条和最近recent条）
    let mut keep: Vec<Message> = Vec::new();
    if !msgs.is_empty() {
        keep.push(msgs[0].clone())
    }
    let mut recent: Vec<Message> = msgs[msgs.len().saturating_sub(keep_recent)..].to_vec();
    //保证边界完整，若recent以tool开头，则其assistant被丢弃，需要补充一条
    if let Some(first) = recent.first_mut() {
        if first.role == Role::Tool {
            *first = Message::user("(早期对话因上下文限制被截断)")
        }
    }
    keep.extend(recent);
    keep
}
