//! Shell 工具：在受限条件下执行终端命令。
//!
//! 防护三层：
//! 1. 权限层（第 15 章）：变更工具，默认需用户确认；
//! 2. 参数层（本文件）：危险命令模式直接拒绝；
//! 3. 运行时层：固定 cwd、超时强杀、输出头尾截断。

use super::{Tool, ToolContext, ToolOutput};
use crate::error::Result;
use async_trait::async_trait;
use serde_json::{Value, json};
use std::time::Duration;

/// 默认超时：覆盖 cargo build/test 等常见任务
const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// 输出总上限：头部 12KB + 尾部 12KB
const MAX_OUTPUT_BYTES: usize = 24 * 1024;

// ────────────────────────────────────────────────────────────
// 危险命令检测
// ────────────────────────────────────────────────────────────

/// 风险等级
#[derive(Debug, PartialEq)]
enum Risk {
    Safe,
    /// 直接拒绝：灾难性命令
    Dangerous(&'static str),
}

/// Deny 级模式：命中即拒。模式描述写在元组第二项（给用户/模型看的原因）。
const DANGEROUS_PATTERNS: &[(&str, &str)] = &[
    (
        r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)?/(\s|$)",
        "禁止删除根目录",
    ),
    (
        r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+~",
        "禁止递归删除 HOME 目录",
    ),
    (r"\bmkfs\b", "禁止格式化磁盘"),
    (r"\bdd\b.*\bof=/dev/", "禁止直接写块设备"),
    (r":\(\)\s*\{\s*:\|:&\s*\}\s*;:", "检测到 fork 炸弹"),
    (r"\b(shutdown|reboot|poweroff|halt)\b", "禁止关机/重启"),
    (
        r"\bgit\s+push\b.*(--force|-f)\b.*\b(main|master)\b",
        "禁止强推主分支",
    ),
    (r"\bchmod\s+-R\s+777\s+/", "禁止对根目录递归 777"),
];

/// 检查命令文本的风险等级
fn assess(command: &str) -> Risk {
    for (pattern, reason) in DANGEROUS_PATTERNS {
        if regex::Regex::new(pattern)
            .map(|re| re.is_match(command))
            .unwrap_or(false)
        {
            return Risk::Dangerous(reason);
        }
    }
    Risk::Safe
}

// ────────────────────────────────────────────────────────────
// 工具实现
// ────────────────────────────────────────────────────────────

pub struct Bash;

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> String {
        "bash".into()
    }

    fn description(&self) -> String {
        "在工作目录下执行 shell 命令，返回 stdout+stderr 合并输出。\n\
         参数：command 命令文本；可选 timeout_secs（默认 120 秒，超时强杀）。\n\
         限制：危险命令（rm -rf /、sudo 类、关机等）会被拒绝；\
         输出超过 24KB 时保留头尾、省略中间。\n\
         适用：构建、测试、git、包管理、查看系统信息。\n\
         不适用：编辑文件（用 edit_file）、读文件（用 read_file）——专用工具更可靠。"
            .into()
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"},
                "timeout_secs": {"type": "integer", "description": "超时秒数，默认 120"}
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let command = args["command"].as_str().unwrap_or_default().to_string();
        let timeout = Duration::from_secs(
            args["timeout_secs"]
                .as_u64()
                .unwrap_or(DEFAULT_TIMEOUT_SECS),
        );

        // 参数层防护：危险命令直接拒绝（不执行、不进子进程）
        if let Risk::Dangerous(reason) = assess(&command) {
            return Ok(ToolOutput::text(format!(
                "命令被拒绝执行：{reason}。\n命令: {command}\n\
                 如确有必要，请向用户说明并请求手动执行。"
            )));
        }

        // 跨平台 shell 选择
        #[cfg(windows)]
        let (shell, flag) = ("cmd", "/C");
        #[cfg(not(windows))]
        let (shell, flag) = ("sh", "-c");

        let child = match tokio::process::Command::new(shell)
            .arg(flag)
            .arg(&command)
            .current_dir(&ctx.cwd) // 固定工作目录：模型的"位置感"锚点
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // kill_on_drop：child 被 drop 时自动杀进程（超时分支的保障）
            .kill_on_drop(true)
            .spawn()
        {
            Ok(c) => c,
            Err(e) => return Ok(ToolOutput::text(format!("启动命令失败: {e}"))),
        };

        // 等待 + 超时竞速
        let result = tokio::time::timeout(timeout, child.wait_with_output()).await;

        match result {
            Err(_) => {
                // 超时：kill_on_drop 在 child 离开作用域时杀进程
                Ok(ToolOutput::text(format!(
                    "命令执行超时（{}s），已强制终止。\
                     若是长任务，请用更大 timeout_secs 重试或拆分为阶段执行。",
                    timeout.as_secs()
                )))
            }
            Ok(Err(e)) => Ok(ToolOutput::text(format!("等待命令结束失败: {e}"))),
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let merged = merge_output(&stdout, &stderr);
                let body = head_tail_truncate(&merged, MAX_OUTPUT_BYTES);

                let status_line = match output.status.code() {
                    Some(0) => "[退出码 0]".to_string(),
                    Some(code) => format!("[退出码 {code}]"),
                    None => "[被信号终止]".to_string(),
                };
                Ok(ToolOutput::text(format!("{status_line}\n{body}")))
            }
        }
    }
}

/// 合并 stdout/stderr：stderr 非空时附在后面并标注
fn merge_output(stdout: &str, stderr: &str) -> String {
    if stderr.trim().is_empty() {
        stdout.to_string()
    } else if stdout.trim().is_empty() {
        format!("[stderr]\n{stderr}")
    } else {
        format!("{stdout}\n[stderr]\n{stderr}")
    }
}

/// 头尾截断：保留前 2/3 预算 + 后 1/3 预算，中间标注省略。
/// 头部有命令上下文，尾部有最终错误——中间最不重要。
fn head_tail_truncate(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let head_budget = max_bytes * 2 / 3;
    let tail_budget = max_bytes - head_budget;

    let mut head_end = head_budget;
    while !s.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = s.len() - tail_budget;
    while !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }

    format!(
        "{}\n\n[... 中间省略 {} 字节 ...]\n\n{}",
        &s[..head_end],
        tail_start - head_end,
        &s[tail_start..]
    )
}
