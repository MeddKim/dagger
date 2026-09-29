use anyhow::Ok;
use dagger::llm::{
    anthropic::AnthropicClient,
    unified::{Message, UnifiedRequest},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let openclient = AnthropicClient::new(
        std::env::var("ANTHROPIC_API_KEY")?,
        std::env::var("ANTHROPIC_BASE_URL").ok(),
    );

    let request = UnifiedRequest {
        model: "deepseek-flash".into(),
        system: Some("你是一个AI助手".into()),
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: 12800u32,
        messages: vec![Message::user("简单介绍一下你自己")],
        thinking: false,
    };

    let res = openclient.complete(&request).await?;

    let msg = res.message.text();

    println!("{}", msg);

    Ok(())
}
