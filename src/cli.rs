use clap::{Args, Parser, Subcommand};

use crate::{
    agent::{self, prompt, subagent},
    tools::{self, ToolContext, permission::PermissionManager},
    ui,
};
use std::io::BufRead;

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
        /// API Key；省略则安全地交互输入
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
            let cwd = std::env::current_dir()?;
            // 读取配置
            let project_root = crate::project::find_root(&cwd);
            let cfg =
                crate::config::Config::load(&project_root, &cli.global)?.to_provider_config()?;
            // 权限管理工具
            let permissions =
                std::sync::Arc::new(PermissionManager::new(atty_is_terminal(), false));
            // 工具合集 + 子任务工具
            let registry = tools::standard_registry()
                .register(subagent::Task::new(cfg.clone(), permissions.clone()));
            // 工具上下文
            let tool_context = ToolContext { cwd: cwd.clone() };

            // 项目信息
            let project = crate::project::detect(&cwd);
            let project_section = crate::project::render_prompt_section(&project);
            //系统提示词
            let system = format!(
                "{}\n\n---\n\n{}",
                prompt::build_system_prompt(&cwd),
                project_section
            );

            let mut agent = agent::Agent::new(&cfg, system, registry, tool_context)
                .with_permissions(permissions.clone());
            ui::run_repl(&mut agent).await?;
        }
        Some(Command::Run { prompt, resume }) => {
            let cwd = std::env::current_dir()?;
            // 读取配置
            let project_root = crate::project::find_root(&cwd);
            let cfg =
                crate::config::Config::load(&project_root, &cli.global)?.to_provider_config()?;
            // 权限管理工具
            let permissions =
                std::sync::Arc::new(PermissionManager::new(atty_is_terminal(), false));
            // 工具合集 + 子任务工具
            let registry = tools::standard_registry()
                .register(subagent::Task::new(cfg.clone(), permissions.clone()));
            // 工具上下文
            let tool_context = ToolContext { cwd: cwd.clone() };

            // 项目信息
            let project = crate::project::detect(&cwd);
            let project_section = crate::project::render_prompt_section(&project);
            //系统提示词
            let system = format!(
                "{}\n\n---\n\n{}",
                prompt::build_system_prompt(&cwd),
                project_section
            );

            let mut agent = agent::Agent::new(&cfg, system, registry, tool_context)
                .with_permissions(permissions.clone());

            //恢复历史消息
            let store = crate::session::SessionStore::default_location()?;
            let session_id = if let Some(id) = resume {
                let history = store.load(&id)?;
                agent.set_messages(history);
                id.clone()
            } else {
                store.create(cfg.kind_name(), &cfg.model, &cwd)?
            };
            let answer = agent.run(&prompt, None).await?;
            // 采用全量添加的方式重新保存message
            store.rewrite_messages(&session_id, &agent.messages())?;

            println!("{answer}");
        }
        Some(Command::Sesssions) => {
            let store = crate::session::SessionStore::default_location()?;
            for s in store.list()? {
                println!(
                    "{} {:<20} {:<10} {}",
                    s.created_at.format("%m-%d %H%:%M"),
                    s.id,
                    s.model,
                    s.cwd
                )
            }
        }
        Some(Command::Config { action }) => match action {
            ConfigAction::Show => {
                let root = crate::project::find_root(&std::env::current_dir()?);
                let cfg = crate::config::Config::load(&root, &cli.global)?;
                println!("{}", cfg.render());
            }
            ConfigAction::SetKey { provider, key } => {
                let key = match key {
                    Some(k) => k,
                    None => {
                        // 简化：直接读一行；生产用 rpassword 关闭回显（练习）
                        eprint!("请输入 {provider} 的 API Key: ");
                        let mut buf = String::new();
                        std::io::stdin().lock().read_line(&mut buf)?;
                        buf.trim().to_string()
                    }
                };
                crate::config::keychain_set(&provider, &key)?;
                println!("已存入钥匙串（条目：{provider}）");
            }
            ConfigAction::DeleteKey { provider } => {
                crate::config::keychain_delete(&provider)?;
                println!("已从钥匙串删除（条目：{provider}）");
            }
        },
    }
    Ok(())
}

fn atty_is_terminal() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal()
}
