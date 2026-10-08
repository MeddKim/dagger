use clap::{Args, Parser, Subcommand};

use crate::{agent, provider, tools, ui};

#[derive(Parser, Debug)]
#[command(name = "dagger", version = "0.0.1", about, long_about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    #[command(flatten)]
    pub global: GlobalOpts,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    Run {
        prompt: String,
        #[arg(long)]
        resume: Option<String>,
    },
    Sesssions,
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// 打印当前生效的配置、来源与脱敏密钥
    Show,
    /// 把 API Key 存入 OS 钥匙串（按 provider 分条目）
    SetKey {
        /// 目标 provider：openai-chat / openai-responses / anthropic
        #[arg(long, default_value = "openai-chat")]
        provider: String,
        /// API Key；省略则安全地交互输入（不回显交给 rpassword，练习）
        key: Option<String>,
    },
    /// 从钥匙串删除指定 provider 的 API Key
    DeleteKey {
        #[arg(long, default_value = "openai-chat")]
        provider: String,
    },
}

/// 全局覆盖参数：优先级高于配置文件和环境变量
#[derive(Args, Debug, Default)]
pub struct GlobalOpts {
    /// 模型提供方：openai-chat / openai-response / anthropic
    #[arg(long, global = true)]
    pub provider: Option<String>,

    /// 模型名， 如 gpt-4o / claude-sonnet-4.5
    #[arg(long, global = true)]
    pub model: Option<String>,

    /// API Key （不推荐命令行传入，会留在 shell 历史；优先使用环境变量）
    #[arg(long, global = true, hide = true)]
    pub api_key: Option<String>,

    /// 自定义API端点
    #[arg(long, global = true)]
    pub base_url: Option<String>,

    /// 详细日志：-v 查看 info， -vv 查看debug
    #[arg(short, long, global=true, action=clap::ArgAction::Count)]
    pub verbose: u8,
}

// 主方法
pub async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        None => {
            let cfg = provider::provider_from_env()?;
            let provider = provider::build_provider(&cfg);
            let registry = tools::ToolRegistry::new();
            let mut agent = agent::Agent::new(
                provider,
                cfg.model.clone(),
                "你是一个AI助手".into(),
                registry,
            );
            ui::run_repl(&mut agent).await?;
        }
        Some(Command::Run { prompt, resume: _ }) => {
            let cfg = provider::provider_from_env()?;
            let provider = provider::build_provider(&cfg);
            let registry = tools::ToolRegistry::new();
            let mut agent = agent::Agent::new(
                provider,
                cfg.model.clone(),
                "你是一个AI助手".into(),
                registry,
            );
            let answer = agent.run(&prompt, None).await?;
            println!("{answer}");
        }
        Some(Command::Sesssions) => {}
        Some(Command::Config { action: _ }) => {}
    }
    Ok(())
}
