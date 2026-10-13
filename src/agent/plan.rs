//! Plan Mode：只读规划 → 用户审批 → 按计划执行。
//!
//! 协作结构：
//!   REPL 持有一个 PlanSlot，构建规划期注册表时把克隆交给 ExitPlanMode；
//!   模型调用 exit_plan_mode(plan) → 工具写槽位；
//!   Agent.run 返回后 → REPL 读槽位 → 进入审批流。

use crate::error::Result;
use crate::tools::{self, Tool, ToolContext, ToolOutput, ToolRegistry};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

//计划槽位
pub type PlanSlot = Arc<Mutex<Option<String>>>;

pub fn new_slot() -> PlanSlot {
    Arc::new(Mutex::new(None))
}

/// 取走计划
pub fn take(slot: &PlanSlot) -> Option<String> {
    slot.lock().unwrap().take()
}

pub const PLAN_MODE_ADDENDUM: &str = r#"
---
⚠️ 当前处于计划模式（Plan Mode）：
- 你只能使用只读工具探索代码库，无法修改任何文件、无法执行命令；
- 充分探索后，调用 exit_plan_mode 提交一份实施计划；
- 计划要求：分步骤、每步注明涉及的文件、标注风险点与验证方式；
- 信息不足时在计划中列出你的假设，不要硬编；
- 提交计划后立即结束本轮，等待用户审批。
"#;

pub struct ExitPlanMode {
    slot: PlanSlot,
}

impl ExitPlanMode {
    pub fn new(slot: PlanSlot) -> Self {
        Self { slot }
    }
}

#[async_trait]
impl Tool for ExitPlanMode {
    fn name(&self) -> String {
        "exit_plan_mode".into()
    }

    fn description(&self) -> String {
        "计划模式下提交实施计划并退出规划。plan 参数为完整计划：\
         分步骤、注明涉及文件、标注风险与验证方式。\
         提交后等待用户审批，不要再做其他操作。"
            .into()
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "完整的实施计划（Markdown 格式）"
                }
            },
            "required": ["plan"]
        })
    }

    // 无副作用（只写内存槽位）→ 视为只读，权限模型自动放行
    fn is_read_only(&self) -> bool {
        true
    }

    async fn execute(&self, args: Value, _ctx: &ToolContext) -> Result<ToolOutput> {
        let plan = args["plan"].as_str().unwrap_or_default().to_string();
        if plan.trim().is_empty() {
            return Ok(ToolOutput::text("错误：计划内容为空，请重新提交。"));
        }
        *self.slot.lock().unwrap() = Some(plan);
        // 提示模型收尾
        Ok(ToolOutput::text(
            "计划已提交，等待用户审批。请立即结束本轮，不要输出其他内容。",
        ))
    }
}

pub fn plan_registry(slot: PlanSlot) -> ToolRegistry {
    tools::expore_registry().register(ExitPlanMode::new(slot))
}

pub enum PlanDecision {
    /// 批准：进入执行阶段
    Approve,
    /// 打回：附带修改意见，不退出计划模式，继续规划
    Revise(String),
    /// 拒绝：不执行计划，放弃并退出计划模式
    Reject,
}

pub fn prompt_approval(plan: &str) -> PlanDecision {
    println!("\n═══ 实施计划 ═══\n{plan}\n════════════════");
    print!("批准执行？[y=批准 / e=提出修改意见 / n=放弃] ");
    let _ = std::io::stdout().flush();

    let mut line = String::new();
    if std::io::stdin().lock().read_line(&mut line).is_err() {
        return PlanDecision::Reject;
    }
    match line.trim().to_ascii_lowercase().as_str() {
        "y" | "" => PlanDecision::Approve,
        "e" => {
            print!("修改意见");
            let _ = std::io::stdout().flush();
            let mut feedback = String::new();
            let _ = std::io::stdin().lock().read_line(&mut feedback);
            PlanDecision::Revise(feedback.trim().to_string())
        }
        _ => PlanDecision::Reject,
    }
}
