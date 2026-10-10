use super::{Tool, ToolContext, ToolOutput};
use crate::error::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

const MAX_READ_BYTES: usize = 32 * 1024;
const MAX_GREP_RESULTS: usize = 100;
const MAX_GLOB_RESULTS: usize = 200;

// ===================================
// read_file
// ===================================
pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> String {
        "read_file".into()
    }

    fn description(&self) -> String {
        "读取文本文件内容，返回带行号的文本。\n\
         参数：path 文件路径；可选 offset（起始行，从 1 计）和 limit（行数）用于分段读大文件。\n\
         输出：每行带 \"行号→\" 前缀。超长会截断并提示。\n\
         典型用法：先 read_file 看全貌（小文件）或 grep 定位后 read_file(offset,limit) 精读。"
            .into()
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径（相对工作目录或绝对）"},
                "offset": {"type": "integer", "description": "起始行号（从 1 开始，默认 1）"},
                "limit": {"type": "integer", "description": "最多读取行数（默认全部，受 32KB 截断）"}
            },
            "required": ["path"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let path = ctx.resolve(args["path"].as_str().unwrap_or_default());
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolOutput::text(format!(
                    "读取 {} 失败: {e}（确认路径存在且是文本文件；可用 glob 查找正确路径）",
                    path.display()
                )));
            }
        };

        // offset/limit 分段：模型处理大文件的分页机制
        let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
        let limit = args["limit"].as_u64().map(|n| n as usize);

        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();
        let slice: Vec<&str> = lines
            .iter()
            .skip(offset - 1)
            .take(limit.unwrap_or(usize::MAX))
            .copied()
            .collect();

        let numbered: String = slice
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{}→{line}", offset + i)) // 行号用真实行号
            .collect::<Vec<_>>()
            .join("\n");

        let header = format!(
            "[文件共 {total} 行，当前显示第 {offset}~{} 行]\n",
            offset + slice.len().saturating_sub(1)
        );
        Ok(ToolOutput::truncated_text(
            format!("{header}{numbered}"),
            MAX_READ_BYTES,
        ))
    }
}

// ════════════════════════════════════════════════════════════
// write_file：整文件写入
// ════════════════════════════════════════════════════════════

pub struct WriteFile;

#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> String {
        "write_file".into()
    }

    fn description(&self) -> String {
        "将完整内容写入文件（不存在则创建，存在则覆盖）。\n\
         适用：创建新文件、整体重写小文件。\n\
         不适用：修改大文件的局部（请用 edit_file，避免重写出错）。\n\
         会自动创建不存在的父目录；只能写工作目录内的路径。"
            .into()
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目标文件路径（须在工作目录内）"},
                "content": {"type": "string", "description": "要写入的完整文件内容"}
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let path = ctx.resolve(args["path"].as_str().unwrap_or_default());
        let content = args["content"].as_str().unwrap_or_default();

        // 越界检查：防止 ../../ 逃逸工作区（第 15 章权限模型的路径级实现）
        if !ctx.is_inside_workspace(&path) {
            return Ok(ToolOutput::text(format!(
                "拒绝写入：{} 不在工作目录 {} 内",
                path.display(),
                ctx.cwd.display()
            )));
        }

        // 自动创建父目录
        if let Some(parent) = path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return Ok(ToolOutput::text(format!("创建目录失败: {e}")));
            }
        }

        let existed = path.exists();
        match tokio::fs::write(&path, content).await {
            Ok(()) => Ok(ToolOutput::text(format!(
                "已{} {}（{} 字节）",
                if existed { "覆盖写入" } else { "创建" },
                path.display(),
                content.len()
            ))),
            Err(e) => Ok(ToolOutput::text(format!(
                "写入 {} 失败: {e}",
                path.display()
            ))),
        }
    }
}

// ════════════════════════════════════════════════════════════
// glob：按模式查找文件
// ════════════════════════════════════════════════════════════

pub struct Glob;

#[async_trait]
impl Tool for Glob {
    fn name(&self) -> String {
        "glob".into()
    }

    fn description(&self) -> String {
        "按 glob 模式查找文件路径（不匹配内容，只匹配路径）。\n\
         示例：\"**/*.rs\" 全部 Rust 文件；\"src/**/mod.rs\"；\"*.{toml,md}\"。\n\
         自动跳过 .gitignore 忽略的内容（如 target/、node_modules/）。\n\
         输出：按修改时间倒序的路径列表（最最近修改的在前），最多 200 条。"
            .into()
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "glob 模式，如 **/*.rs"},
                "path": {"type": "string", "description": "搜索根目录（默认工作目录）"}
            },
            "required": ["pattern"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let pattern = args["pattern"].as_str().unwrap_or_default();
        let root = args["path"]
            .as_str()
            .map(|p| ctx.resolve(p))
            .unwrap_or_else(|| ctx.cwd.clone());

        let glob = match globset::Glob::new(pattern) {
            Ok(g) => g.compile_matcher(),
            Err(e) => return Ok(ToolOutput::text(format!("glob 模式无效: {e}"))),
        };

        // ignore::WalkBuilder：gitignore 感知的目录遍历（ripgrep 同款）
        let mut matches: Vec<(std::path::PathBuf, std::time::SystemTime)> = Vec::new();
        let walker = ignore::WalkBuilder::new(&root)
            .hidden(true) // 跳过 . 开头文件
            .git_ignore(true)
            .build();

        for entry in walker.flatten() {
            if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                let rel = entry.path().strip_prefix(&root).unwrap_or(entry.path());
                if glob.is_match(rel) {
                    let mtime = entry
                        .metadata()
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    matches.push((entry.path().to_path_buf(), mtime));
                    if matches.len() >= MAX_GLOB_RESULTS {
                        break;
                    }
                }
            }
        }

        // 按修改时间倒序：最近的在前——模型关心"活跃"代码
        matches.sort_by(|a, b| b.1.cmp(&a.1));

        if matches.is_empty() {
            return Ok(ToolOutput::text(format!(
                "没有匹配 \"{pattern}\" 的文件（搜索根：{}）",
                root.display()
            )));
        }

        let body = matches
            .iter()
            .map(|(p, _)| p.strip_prefix(&ctx.cwd).unwrap_or(p).display().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ToolOutput::text(format!(
            "匹配 {} 个文件（按修改时间倒序）:\n{body}",
            matches.len()
        )))
    }
}

// ════════════════════════════════════════════════════════════
// grep：内容搜索
// ════════════════════════════════════════════════════════════

pub struct Grep;

#[async_trait]
impl Tool for Grep {
    fn name(&self) -> String {
        "grep".into()
    }

    fn description(&self) -> String {
        "在文件内容中搜索正则表达式，输出 路径:行号:匹配行。\n\
         参数：pattern 正则；可选 path（默认工作目录）、include（glob 过滤，如 *.rs）。\n\
         自动跳过 .gitignore 内容与二进制文件。结果最多 100 条。\n\
         典型用法：grep \"fn main\" 定位函数 → read_file 精读。"
            .into()
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "正则表达式"},
                "path": {"type": "string", "description": "搜索根（默认工作目录）"},
                "include": {"type": "string", "description": "文件名 glob 过滤，如 *.rs"}
            },
            "required": ["pattern"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let pattern = args["pattern"].as_str().unwrap_or_default();
        let re = match regex::Regex::new(pattern) {
            Ok(r) => r,
            Err(e) => return Ok(ToolOutput::text(format!("正则表达式无效: {e}"))),
        };
        let root = args["path"]
            .as_str()
            .map(|p| ctx.resolve(p))
            .unwrap_or_else(|| ctx.cwd.clone());
        let include = args["include"]
            .as_str()
            .and_then(|g| globset::Glob::new(g).ok())
            .map(|g| g.compile_matcher());

        let mut results: Vec<String> = Vec::new();
        let walker = ignore::WalkBuilder::new(&root)
            .hidden(true)
            .git_ignore(true)
            .build();

        'files: for entry in walker.flatten() {
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            let rel = path.strip_prefix(&root).unwrap_or(path);
            if let Some(inc) = &include {
                if !inc.is_match(rel) {
                    continue;
                }
            }
            // read_to_string 失败即跳过（二进制文件自然被过滤）
            let Ok(content) = std::fs::read_to_string(path) else {
                continue;
            };

            for (i, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    results.push(format!(
                        "{}:{}:{}",
                        path.strip_prefix(&ctx.cwd).unwrap_or(path).display(),
                        i + 1,
                        line.trim()
                    ));
                    if results.len() >= MAX_GREP_RESULTS {
                        break 'files;
                    }
                }
            }
        }

        if results.is_empty() {
            Ok(ToolOutput::text(format!("未找到匹配 /{pattern}/ 的内容")))
        } else {
            Ok(ToolOutput::text(format!(
                "{} 条匹配:\n{}",
                results.len(),
                results.join("\n")
            )))
        }
    }
}
