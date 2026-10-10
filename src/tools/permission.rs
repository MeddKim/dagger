//! 工具执行权限管理
//!

use std::{
    collections::HashSet,
    io::{BufRead, Write},
    sync::Mutex,
};

/// 权限决策结果
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    // 允许执行
    Allow,
    // 拒绝执行 + 拒绝原因
    Deny(String),
}

/// 权限管理：一个会话一个实例
pub struct PermissionManager {
    // 本次会话内 允许
    always_allowed: Mutex<HashSet<String>>,
    // 是否有交互终端
    interactive: bool,
    // --yes 全局放行
    yes_to_all: bool,
}

impl PermissionManager {
    pub fn new(interactive: bool, yes_to_all: bool) -> Self {
        Self {
            always_allowed: Mutex::new(HashSet::new()),
            interactive,
            yes_to_all,
        }
    }

    // 判定工具是否可执行
    //  - tool_name 工具名
    //  - read_only 工具的只读标志
    //  - preview 给用户的操作预览
    pub fn check(&self, tool_name: &String, read_only: bool, preview: &str) -> Decision {
        //只读工具 或 全局放行 直接放行
        if read_only || self.yes_to_all {
            return Decision::Allow;
        }

        //会话内记住 总是允许
        if self.always_allowed.lock().unwrap().contains(tool_name) {
            return Decision::Allow;
        }

        if !self.interactive {
            return Decision::Deny(format!(
                "当前工具无交互终端，当前工具 {tool_name} 默认拒绝 \
                如需放行，请使用 --yes 参数重新允许"
            ));
        }

        self.ask_user(tool_name, preview)
    }

    fn ask_user(&self, tool_name: &String, preview: &str) -> Decision {
        let stderr = std::io::stderr();
        let mut out = stderr.lock();
        let _ = writeln!(out, "\n Agent 请求执行变更操作：");
        let _ = writeln!(out, "      {preview}");
        let _ = writeln!(
            out,
            "允许？[y=仅此次 / n=拒绝 / a=本次会话总是拒绝 {tool_name}]"
        );
        let _ = out.flush();
        let mut line = String::new();
        let stdin = std::io::stdin();
        if stdin.lock().read_line(&mut line).is_err() {
            return Decision::Deny("读取用户输入失败".into());
        }
        match line.trim().to_ascii_lowercase().as_str() {
            // 将y 和 直接回车都算 y
            "y" | "" => Decision::Allow,
            "a" => {
                self.always_allowed
                    .lock()
                    .unwrap()
                    .insert(tool_name.clone());
                Decision::Allow
            }
            _ => Decision::Deny("用户拒绝了该操作".into()),
        }
    }
}
