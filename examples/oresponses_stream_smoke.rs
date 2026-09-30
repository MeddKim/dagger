//! OpenAI Responses 流式补全冒烟测试
//!
use anyhow::Ok;
use dagger::llm::openai_responses::OpenAIResponsesClient;
use dagger::llm::unified::{Message, StreamEvent, UnifiedRequest};
use futures_util::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = OpenAIResponsesClient::new(
        std::env::var("OPENAI_API_KEY")?,
        std::env::var("OPENAI_BASE_URL").ok(),
    );

    let request = UnifiedRequest {
        model: "deepseek-flash".into(),
        system: Some("你是一个AI助手".into()),
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: 12800u32,
        messages: vec![Message::user("用三句话介绍一下 Rust 的流式处理")],
        thinking: false,
    };

    // 拿到 EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>
    // Pin<Box<_>> 本身是 Unpin 的，可以直接调用 .next()
    let mut stream = client.complete_stream(&request).await?;

    println!("── 流式输出开始 ──────────────────────────\n");

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    while let Some(ev) = stream.next().await {
        match ev? {
            // 文本增量：实时打字机输出
            StreamEvent::TextDelta(t) => {
                out.write_all(t.as_bytes())?;
                out.flush()?;
            }
            // 思考增量：单独标记展示
            StreamEvent::ThinkingDelta(t) => {
                print!("[思考] {t}");
                std::io::stdout().flush()?;
            }
            // 工具调用生命周期（本例未传 tools，仅作完备匹配）
            StreamEvent::ToolUseStart { id, name } => {
                println!("\n[工具调用开始] id={id} name={name}");
            }
            StreamEvent::ToolUseInputDelta { id, delta } => {
                println!("[工具参数增量] id={id} delta={delta}");
            }
            StreamEvent::ToolUseEnd { id } => {
                println!("[工具调用结束] id={id}");
            }
            // 流结束：打印聚合结果
            StreamEvent::Done(resp) => {
                println!("\n\n── 流式输出结束 ──────────────────────────");
                println!("stop_reason: {:?}", resp.stop_reason);
                println!(
                    "usage: input={} output={} cache_read={} cache_write={}",
                    resp.usage.input_tokens,
                    resp.usage.output_tokens,
                    resp.usage.cache_read_tokens,
                    resp.usage.cache_write_tokens
                );
                println!("聚合文本: {}", resp.message.text());
                println!("聚合内容块数: {}", resp.message.content.len());
            }
        }
    }

    Ok(())
}
