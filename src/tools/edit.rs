//! 编辑工具：edit_file —— 精确字符串替换 + 模糊容错 + diff 输出。
//!
//! 核心语义：找到 old_string 的唯一出现位置，替换为 new_string。
//! - 0 处精确匹配 → 尝试空白归一化的模糊匹配；
//! - 模糊也 0 处 → 报错并给相近线索（模型据此修正）；
//! - 多处匹配 → 拒绝并要求更多上下文（宁可拒绝不可乱改）。

use super::{Tool, ToolContext, ToolOutput};
use crate::error::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

pub struct EditFile;

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> String {
        "edit_file".into()
    }

    fn description(&self) -> String {
        "对文件做局部精确替换：把 old_string 替换为 new_string。\n\
         要求：\n\
         1. 先用 read_file 读目标区域，old_string 必须与文件内容完全一致（含缩进）；\n\
         2. old_string 要在文件中唯一出现——太短会匹配多处而被拒绝，\
            请带上足够的上下文（前后几行）；\n\
         3. 插入内容：old_string 取插入位置的锚点行，new_string 包含锚点行+新内容；\n\
         4. 删除内容：new_string 传空字符串。\n\
         返回：成功时返回统一 diff；失败时返回原因与修正建议。"
            .into()
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目标文件路径"},
                "old_string": {"type": "string", "description": "要被替换的原文（须唯一出现）"},
                "new_string": {"type": "string", "description": "替换后的内容（删除则传空串）"}
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let path = ctx.resolve(args["path"].as_str().unwrap_or_default());
        let old = args["old_string"].as_str().unwrap_or_default();
        let new = args["new_string"].as_str().unwrap_or_default();

        if !ctx.is_inside_workspace(&path) {
            return Ok(ToolOutput::text("拒绝编辑：路径不在工作目录内".to_string()));
        }
        if old == new {
            return Ok(ToolOutput::text(
                "old_string 与 new_string 相同，无需修改（请检查参数）".to_string(),
            ));
        }

        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return Ok(ToolOutput::text(format!("读取文件失败: {e}"))),
        };

        // 三级匹配策略
        let outcome = match find_exact(&content, old) {
            MatchResult::Unique(range) => EditOutcome::Exact(range),
            MatchResult::Ambiguous(count) => {
                return Ok(ToolOutput::text(format!(
                    "替换被拒绝：old_string 在文件中出现 {count} 处，无法确定改哪一处。\n\
                     请扩大 old_string 的范围（带上前后文行）使其唯一后重试。"
                )));
            }
            MatchResult::NotFound => match find_fuzzy(&content, old) {
                Some((range, actual)) => EditOutcome::Fuzzy { range, actual },
                None => {
                    return Ok(ToolOutput::text(format!(
                        "未找到 old_string。可能原因：文件已被改动、内容记忆有误。\n\
                         建议：先 read_file 读取 {} 的最新内容，再基于真实内容编辑。\n\
                         你提供的 old_string 开头: {:?}",
                        path.display(),
                        &old[..old.len().min(100)]
                    )));
                }
            },
        };

        // 执行替换
        let (range, note) = match &outcome {
            EditOutcome::Exact(r) => (r.clone(), String::new()),
            EditOutcome::Fuzzy { range, actual } => (
                range.clone(),
                format!(
                    "注意：old_string 与文件内容有空白差异，已按实际内容匹配。\n实际匹配到的文本:\n{actual}\n"
                ),
            ),
        };

        let mut new_content = String::with_capacity(content.len() + new.len());
        new_content.push_str(&content[..range.start]);
        new_content.push_str(new);
        new_content.push_str(&content[range.end..]);

        if let Err(e) = tokio::fs::write(&path, &new_content).await {
            return Ok(ToolOutput::text(format!("写入失败: {e}")));
        }

        // 生成 unified diff 给用户/模型确认
        let diff = unified_diff(&content, &new_content, &path.display().to_string());
        Ok(ToolOutput::truncated_text(
            format!("{note}编辑成功。diff:\n{diff}"),
            16 * 1024,
        ))
    }
}

// ════════════════════════════════════════════════════════════
// 匹配引擎
// ════════════════════════════════════════════════════════════

enum MatchResult {
    Unique(std::ops::Range<usize>),
    Ambiguous(usize),
    NotFound,
}

enum EditOutcome {
    Exact(std::ops::Range<usize>),
    Fuzzy {
        range: std::ops::Range<usize>,
        actual: String,
    },
}

/// 精确匹配：统计 old 在 content 中的字节区间出现次数
fn find_exact(content: &str, old: &str) -> MatchResult {
    if old.is_empty() {
        return MatchResult::NotFound;
    }
    let mut found: Vec<usize> = Vec::new();
    let mut start = 0;
    while let Some(pos) = content[start..].find(old) {
        found.push(start + pos);
        start += pos + 1; // 允许重叠计数（对歧义判断更保守）
        if found.len() > 1 {
            return MatchResult::Ambiguous(found.len() + content[start..].matches(old).count());
        }
    }
    match found.len() {
        0 => MatchResult::NotFound,
        1 => MatchResult::Unique(found[0]..found[0] + old.len()),
        _ => MatchResult::Ambiguous(found.len()),
    }
}

/// 模糊匹配：逐行 trim_end 归一化后比较。
///
/// 容忍：行尾空白差异、空行增减。不容忍：内容文字差异。
/// 返回 (字节区间, 实际匹配到的原文) —— 替换用原文坐标，保证不破坏文件。
fn find_fuzzy(content: &str, old: &str) -> Option<(std::ops::Range<usize>, String)> {
    // 把 old 按行拆分并归一化（trim_end + 跳过纯空行）
    let norm_old: Vec<&str> = old
        .lines()
        .map(|l| l.trim_end())
        .filter(|l| !l.is_empty())
        .collect();
    if norm_old.is_empty() {
        return None;
    }

    // 文件行 + 每行的字节偏移（替换要用原始字节坐标）
    let mut offset = 0usize;
    let file_lines: Vec<(&str, usize)> = content
        .lines()
        .map(|l| {
            let entry = (l, offset);
            offset += l.len() + 1; // +1 for \n
            entry
        })
        .collect();

    // 滑动窗口：找连续 N 个非空行与 norm_old 逐行相等（trim_end 后）
    let mut window: Vec<(&str, usize)> = Vec::new();
    for (line, byte_off) in &file_lines {
        let norm = line.trim_end();
        if norm.is_empty() {
            continue; // 空行不参与匹配，天然容忍空行差异
        }
        window.push((norm, *byte_off));
        if window.len() > norm_old.len() {
            window.remove(0);
        }
        if window.len() == norm_old.len()
            && window.iter().map(|&(l, _)| l).eq(norm_old.iter().copied())
        {
            // 命中：区间 = 窗口首行起点 .. 窗口末行终点
            let start = window[0].1;
            let last_line_start = window[window.len() - 1].1;
            let last_line = &content[last_line_start..];
            let end = last_line_start + last_line.lines().next().unwrap_or("").len();
            return Some((start..end, content[start..end].to_string()));
        }
    }
    None
}

/// 生成 unified diff（similar crate）
fn unified_diff(old: &str, new: &str, path: &str) -> String {
    similar::TextDiff::from_lines(old, new)
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{path}"), &format!("b/{path}"))
        .to_string()
}
