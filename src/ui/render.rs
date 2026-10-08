use std::io::Write;

use tokio::sync::mpsc;

use crate::{agent::AgentEvent, llm::unified::StreamEvent};

pub async fn render_loop(mut rx: mpsc::Receiver<AgentEvent>) {
    let mut in_text = false;

    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::Stream(se) => match se {
                StreamEvent::TextDelta(t) => {
                    print!("{t}");
                    flush();
                    in_text = false;
                }
                StreamEvent::ThinkingDelta(_) => {}
                StreamEvent::ToolUseStart { name, .. } => {
                    if in_text {
                        println!();
                        in_text = false;
                    }
                    println!("⚙  调用工具 {name} ...");
                }
                StreamEvent::ToolUseInputDelta { .. } => {}
                StreamEvent::ToolUseEnd { .. } => {}
                StreamEvent::Done(_) => {
                    if in_text {
                        println!();
                        in_text = false;
                    }
                }
            },
            AgentEvent::ToolStarted { name, input } => {
                println!("  L 参数： {}", summarize(&input.to_string(), 100));
                let _ = name;
            }
            AgentEvent::ToolFinished {
                name,
                output,
                is_error,
            } => {
                let icon = if is_error { "❌" } else { "✓" };
                let summary = output.lines().take(2).collect::<Vec<_>>().join(" ");
                println!("   {icon} {name}: {}", summarize(&summary, 120));
            }
            AgentEvent::Finished { .. } => {
                // 最终文本在 TextDelta中已处理，不必重复
            }
            AgentEvent::Error(msg) => {
                eprintln!("\n ❌ {msg}");
            }
        }
    }
    if in_text {
        println!();
    }
}

fn summarize(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(max).collect::<String>())
    }
}

fn flush() {
    let _ = std::io::stdout().flush();
}
