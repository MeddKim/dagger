use crate::{
    agent::plan::{self, PlanDecision},
    error::{DaggerError, Result},
};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, AgentEventTx};

pub mod render;

pub async fn run_repl(agent: &mut Agent) -> Result<()> {
    let mut editor = DefaultEditor::new()?;

    // 计划模式状态
    let mut plan_mode = false;
    let default_prompt = agent.system_prompt().to_string();

    loop {
        match editor.readline("\n > ") {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                //处理命令
                if input.starts_with('/') {
                    match handle_command(input, agent) {
                        CommandResult::Quit => break,
                        CommandResult::TogglePlan => {
                            //已是计划模式要退出，反正进入计划模式
                            plan_mode = !plan_mode;
                            if plan_mode {
                                println!(
                                    "已进入计划模式：我只做只读探索并给出计划。再次 /plan 可退出。"
                                );
                            } else {
                                //退出计划模式，切换会原来工具集 / 系统提示词
                                agent.reset_tool_registry();
                                agent.set_system_prompt(default_prompt.clone());
                                plan_mode = false;
                                println!("已放弃该计划，退出计划模式。");
                            }
                        }
                        CommandResult::Continue => {}
                    }
                    continue;
                }
                if plan_mode {
                    match run_plan_turn(agent, input, &default_prompt).await {
                        PlanTurnResult::Approved(plan_text) => {
                            agent.set_system_prompt(default_prompt.clone());
                            agent.reset_tool_registry();
                            plan_mode = false;
                            let kickoff =
                                format!("计划已获用户批准，请严格按计划执行：\n\n{plan_text}");
                            if let Err(e) = run_once(agent, &kickoff).await {
                                eprintln!("\n 计划执行错误：{e}");
                            }
                        }
                        PlanTurnResult::Rejected => {
                            agent.reset_tool_registry();
                            agent.set_system_prompt(default_prompt.clone());
                            plan_mode = false;
                            println!("已放弃该计划，退出计划模式");
                        }
                        PlanTurnResult::Continue => {}
                    }
                } else if let Err(e) = run_once(agent, input).await {
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
    /// 切换计划模式（进入/退出由调用方决定）
    TogglePlan,
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
        "/plan" => {
            return CommandResult::TogglePlan;
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

/// 计划模式一轮的结果
enum PlanTurnResult {
    /// 计划获批（携带计划文本）
    Approved(String),
    /// 用户放弃
    Rejected,
    /// 未提交计划，或用户要求修订（留在计划模式）
    Continue,
}

/// 计划模式一轮：只读探索 → 提交计划 → 审批循环（修订则重新规划，直到裁决）
async fn run_plan_turn(agent: &mut Agent, input: &str, normal_prompt: &str) -> PlanTurnResult {
    // 新建计划槽位，交给 exit_plan_mode 写入
    let slot = plan::new_slot();
    agent.set_tool_registry(plan::plan_registry(slot.clone()));
    agent.set_system_prompt(format!("{normal_prompt}\n{}", plan::PLAN_MODE_ADDENDUM));

    let mut next_input = input.to_string();
    loop {
        // 规划阶段：只读探索 +（可能）提交计划
        let _ = run_once(agent, &next_input).await;

        match plan::take(&slot) {
            None => {
                println!("模型未提交计划。可补充信息后再试，或 /plan 退出计划模式。");
                return PlanTurnResult::Continue;
            }
            Some(plan_text) => match plan::prompt_approval(&plan_text) {
                PlanDecision::Approve => return PlanTurnResult::Approved(plan_text),
                PlanDecision::Reject => return PlanTurnResult::Rejected,
                PlanDecision::Revise(feedback) => {
                    next_input = format!(
                        "用户对计划的修改意见：{feedback}\n请据此修订计划，然后重新调用 exit_plan_mode 提交。"
                    );
                    // 继续循环：重新规划 → 再次审批
                }
            },
        }
    }
}
