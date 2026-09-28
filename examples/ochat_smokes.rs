use anyhow::Ok;
use dagger::llm::{
    openai_chat::OpenAIChatClient,
    unified::{Message, UnifiedRequest},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let openclient = OpenAIChatClient::new(
        std::env::var("OPENAI_API_KEY")?,
        std::env::var("OPENAI_BASE_URL").ok(),
    );

    let request = UnifiedRequest {
        model: "deepseek-v4.1-flash".into(),
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
