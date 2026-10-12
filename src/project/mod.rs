//! 项目信息加载，包括根目录探测，AGENTS.md加载，精简目录树

use std::path::{Path, PathBuf};

///项目跟标记文件
const ROOT_MARKERS: &[&str] = &[
    ".git",
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "go.mod",
];

/// 项目指令文件的候选名称
const INSTRUCTION_FILES: &[&str] = &["AGENTS.md", "AGENT.md", "CLAUDE.md"];

/// 项目指令文件注入限制
const MAX_INSTRUCTION_BYTES: usize = 8 * 1024;

/// 项目数参数
const TREE_MAX_DEPTH: usize = 2; //目录遍历最多层数
const TREE_MAX_ENTRIES: usize = 24; //全局上线

/// 项目上下文
#[derive(Debug)]
pub struct ProjectContext {
    pub root: PathBuf,
    pub instructions: Option<String>,
    pub tree: String,
}

/// 从start开始向上逐级探测项目跟，找不到返回 start 本省
pub fn find_root(start: &Path) -> PathBuf {
    let mut dir = start;
    loop {
        for marker in ROOT_MARKERS {
            if dir.join(marker).exists() {
                return dir.to_path_buf();
            }
        }
        match dir.parent() {
            Some(p) => dir = p,
            None => return start.to_path_buf(),
        }
    }
}

pub fn load_instructions(root: &Path) -> Option<String> {
    for name in INSTRUCTION_FILES {
        let path = root.join(name);
        if let Ok(content) = std::fs::read_to_string(&path) {
            let truncated: String = content.chars().take(MAX_INSTRUCTION_BYTES).collect();
            return Some(format!(
                "（来自{name}）\n{truncated}{}",
                if content.len() > MAX_INSTRUCTION_BYTES {
                    "\n...[截断]"
                } else {
                    ""
                }
            ));
        }
    }
    None
}

/// 生成瘦身目录树。
/// 只展示TREE_MAX_DEPTH级
///
/// 输出形如：
/// ```text
/// src/
///   main.rs
///   lib.rs
///   llm/ (5 个文件)
/// ```
fn build_tree(root: &Path) -> String {
    let mut out = String::new();
    let mut builder = ignore::WalkBuilder::new(root);
    builder
        .max_depth(Some(TREE_MAX_DEPTH + 1))
        .hidden(true)
        .git_ignore(true);

    let mut entries: Vec<String> = builder
        .build()
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            if path == root {
                return None;
            }
            let rel = path.strip_prefix(root).ok()?;
            let depth = rel.components().count();
            if depth > TREE_MAX_DEPTH {
                return None;
            }
            let indent = "  ".repeat(depth - 1);
            let name = path.file_name()?.to_string_lossy();
            Some(if path.is_dir() {
                format!("{indent}{name}/")
            } else {
                format!("{indent}{name}")
            })
        })
        .collect();

    entries.sort();
    entries.sort();
    if entries.len() > TREE_MAX_ENTRIES {
        entries.truncate(TREE_MAX_ENTRIES);
        out.push_str(&entries.join("\n"));
        out.push_str("\n...[已省略，必要时可用 glob 工具查看完整结构]");
        return out;
    }
    out.push_str(&entries.join("\n"));
    out
}

/// 一站式探测：根 + 指令 + 目录树
pub fn detect(start: &Path) -> ProjectContext {
    let root = find_root(start);
    let instructions = load_instructions(&root);
    let tree = build_tree(&root);
    ProjectContext {
        root,
        instructions,
        tree,
    }
}

/// 把项目上下文渲染成系统提示词片段
pub fn render_prompt_section(ctx: &ProjectContext) -> String {
    let mut s = format!(
        "项目信息：\n- 项目根目录: {}\n\n项目结构（{TREE_MAX_DEPTH} 层）:\n```\n{}\n```",
        ctx.root.display(),
        ctx.tree
    );
    if let Some(inst) = &ctx.instructions {
        s.push_str(&format!("\n\n项目规范（必须遵守）：\n{inst}"));
    }
    s
}
