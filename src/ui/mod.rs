use crate::error::{DaggerError, Result};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, AgentEventTx};

pub mod render;

pub async fn run_repl(agent: &mut Agent) -> Result<()> {
    let mut editor = DefaultEditor::new()?;

    loop {
        match editor.readline("\n > ") {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                //处理命令
                if input.starts_with('/') {
                    if handle_command(input, agent) == CommandResult::Quit {
                        break;
                    }
                    continue;
                }
                if let Err(e) = run_once(agent, input).await {
                    eprintln!("\n agent 运行出错： {e}");
                }
            }
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
                println!("\n 再见")
            }
            Err(e) => return Err(e.into()),
        }
    }

    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum CommandResult {
    Continue,
    Quit,
}

fn handle_command(input: &str, _: &mut Agent) -> CommandResult {
    let mut parts = input.splitn(2, ' ');
    let cmd = parts.next().unwrap_or("");
    // let arg = parts.next().unwrap_or("").trim();

    match cmd {
        "/help" => {
            println!(
                "可用命令：
                      /help      显示本帮助
                      /plan      切换计划模式（只读探索 → 提交计划 → 审批后执行）
                      /cost      查看本次会话的 token 用量与估算成本
                      /clear     清空对话历史（开始新话题）
                      /history   查看当前对话的消息数
                      /quit      退出 dagger

                    直接输入文字即可与 Agent 对话。"
            )
        }
        "/clear" => {}
        "/history" => {}
        "/quit" | "/exit" => {
            println!("👋🏻 再见");
            return CommandResult::Quit;
        }
        other => println!("未知命令 {other} , 可输入 /help 查看可用命令"),
    }
    CommandResult::Continue
}

async fn run_once(agent: &mut Agent, input: &str) -> Result<()> {
    //接收数据
    let (tx, rx): (AgentEventTx, _) = mpsc::channel(64);

    //新启线程处理 agent事件
    let renderer = tokio::spawn(render::render_loop(rx));

    //本轮取消令牌
    let token = CancellationToken::new();
    agent.set_cancel_token(token.clone());

    //监听取消动作
    let cancel_watcher = {
        let token = token.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            token.cancel();
        })
    };

    let result = agent.run(input, Some(tx)).await;

    // 本轮对话结束，停止监听
    cancel_watcher.abort();
    let _ = renderer.await;

    match result {
        Ok(_) => Ok(()),
        Err(DaggerError::Cancelled) => {
            println!("\n 已取消 （对话历史已保留，可继续提问或修正方向）");
            Ok(())
        }
        Err(e) => Err(e),
    }
}
