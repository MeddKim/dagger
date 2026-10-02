//! Agent（ReAct 循环）冒烟测试
//!
//! 完整链路：环境变量 → ProviderConfig → build_provider → Agent → run
//!
//! 运行方式（与 provider_from_env 的约定一致）：
//!   DAGGER_PROVIDER=openai-chat \          # 可选：openai-chat / openai-responses / anthropic
//!   OPENAI_API_KEY=sk-xxx \
//!   OPENAI_BASE_URL=https://... \          # 可选，指向兼容网关
//!   DAGGER_MODEL=deepseek-v4-flash \       # 可选
//!     cargo run --example angent_smoke
//!
//! 验证点：
//! 1. ToolRegistry 的 get_weather 工具定义能正确下发到模型
//! 2. 模型发起工具调用时，Agent 能执行工具并把结果回传（Observation）
//! 3. ReAct 循环能走完 Thought → Action → Observation → Answer 全程
//! 4. 返回最终文本答案

use dagger::agent::Agent;
use dagger::provider::{build_provider, provider_from_env};
use dagger::tools::ToolRegistry;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info,dagger=debug"))
        .init();

    // ① 组装 Provider：环境变量 → 配置 → Box<dyn Provider>
    //    provider_from_env 返回 Result<_, DaggerError>，
    //    DaggerError 实现了 std::error::Error，? 可自动转成 anyhow::Error
    let cfg = provider_from_env()?;
    let provider = build_provider(&cfg);
    println!("provider: {}", provider.name());
    println!("model: {}", cfg.model);

    // ② 构造 Agent（ToolRegistry 内置 get_weather 工具）
    let mut agent = Agent::new(
        provider,
        cfg.model.clone(),
        "你是一个AI助手".into(),
        ToolRegistry::new(),
    );

    // ③ 提一个需要工具才能回答的问题，触发 ReAct 循环：
    //    模型应先发起 get_weather 工具调用，拿到结果后再组织最终答案
    println!("\n── 用户输入 ──");
    println!("北京今天天气怎么样？适合跑步吗？\n");

    let answer = agent.run("北京今天天气怎么样？适合跑步吗？").await?;

    // ④ 冒烟断言：拿到非空最终答案
    println!("── Agent 最终答案 ──");
    println!("{answer}");

    if answer.trim().is_empty() {
        anyhow::bail!("冒烟失败：Agent 返回了空答案");
    }
    println!("\n✅ 冒烟通过");

    Ok(())
}
