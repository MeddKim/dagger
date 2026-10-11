use tiktoken_rs::CoreBPE;

use crate::llm::unified::{Block, Message};

/// 每条消息的固定开销
const PER_MESSAGE_OVERHEAD: usize = 4;

/// 全局缓存的 BPE 实例
static BPE: std::sync::OnceLock<Option<CoreBPE>> = std::sync::OnceLock::new();

fn bpe() -> Option<&'static CoreBPE> {
    BPE.get_or_init(|| tiktoken_rs::cl100k_base().ok()).as_ref()
}

/// 估算文本 token数，tiktoken不可用时，按 字符数 / 3.5 兜底
/// 考虑中英文场景，宁可多估
pub fn count_text(text: &str) -> usize {
    match bpe() {
        Some(bpe) => bpe.encode_with_special_tokens(text).len(),
        None => (text.chars().count() as f64 / 3.5).ceil() as usize,
    }
}

/// 估算一条消息的tokens数目
pub fn count_message(msg: &Message) -> usize {
    let mut total = PER_MESSAGE_OVERHEAD;
    for block in &msg.content {
        total += match block {
            Block::Text { text } => count_text(text),
            Block::Thinking { thinking, .. } => count_text(thinking),
            Block::ToolUse { name, input, .. } => {
                count_text(name) + count_text(&input.to_string()) + PER_MESSAGE_OVERHEAD
            }
            Block::ToolResult { content, .. } => count_text(content) + PER_MESSAGE_OVERHEAD,
        }
    }
    total
}

///估算消息组token
pub fn count_messages(messages: &[Message]) -> usize {
    messages.iter().map(count_message).sum()
}
