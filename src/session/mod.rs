use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{fs, path::PathBuf};

use crate::error::{DaggerError, Result};
use crate::llm::unified::{Block, Message, Role};

/// 历史记录中的一条数据JSON数据，由 元数据 + 消息 组成
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SessionRecord {
    Meta {
        id: String,
        created_at: DateTime<Utc>,
        provider: String,
        model: String,
        cwd: String,
    },
    Message {
        message: Message,
    },
}

pub struct SessionMeta {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub model: String,
    pub cwd: String,
}

pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    // 会话默认位置：~/.dagger/sessions/
    pub fn default_location() -> Result<Self> {
        //获取home目录
        let home = directories::BaseDirs::new()
            .map(|d| d.home_dir().to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let dir = home.join(".dagger").join("sessions");
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    // 创建文件
    fn file(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.jsonl"))
    }

    pub fn create(&self, provider: &str, model: &str, cwd: &Path) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        let meta = SessionRecord::Meta {
            id: id.clone(),
            created_at: Utc::now(),
            provider: provider.into(),
            model: model.into(),
            cwd: cwd.display().to_string(),
        };
        let line = serde_json::to_string(&meta)?;
        fs::write(self.file(&id), format!("{line}\n"))?;
        Ok(id)
    }

    /// 追加一条消息到历史文件中
    pub fn append_message(&self, id: &str, msg: &Message) -> Result<()> {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(self.file(id))?;
        let line = serde_json::to_string(&SessionRecord::Message {
            message: msg.clone(),
        })?;
        writeln!(f, "{line}")?;
        Ok(())
    }

    ///重新写入
    pub fn rewrite_messages(&self, id: &str, messages: &[Message]) -> Result<()> {
        let path = self.file(id);
        let text = std::fs::read_to_string(&path)
            .map_err(|_| DaggerError::Config(format!("会话 {id} 不存在")))?;
        let meta_line = text.lines().next().unwrap_or_default();

        let mut buf = String::with_capacity(text.len());
        buf.push_str(meta_line);
        buf.push('\n');
        for m in messages {
            let line = serde_json::to_string(&SessionRecord::Message { message: m.clone() })?;
            buf.push_str(&line);
            buf.push('\n');
        }
        fs::write(&path, buf)?;
        Ok(())
    }

    /// 加载会话文件恢复会话
    pub fn load(&self, id: &str) -> Result<Vec<Message>> {
        let text = std::fs::read_to_string(self.file(id))
            .map_err(|_| DaggerError::Config(format!("会话 {id} 不存在")))?;
        let mut messages = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let record: SessionRecord = serde_json::from_str(line)
                .map_err(|e| DaggerError::Parse(format!("会话文件第 {} 行损坏：{e}", i + 1)))?;
            if let SessionRecord::Message { message } = record {
                messages.push(message);
            }
        }
        Ok(repair_orphans(messages))
    }

    // 列出全部会话
    pub fn list(&self) -> Result<Vec<SessionMeta>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.dir)?.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            if let Some(first_line) = std::fs::read_to_string(&path)
                .ok()
                .and_then(|t| t.lines().next().map(String::from))
            {
                if let Ok(SessionRecord::Meta {
                    id,
                    created_at,
                    model,
                    cwd,
                    ..
                }) = serde_json::from_str(&first_line)
                {
                    out.push(SessionMeta {
                        id,
                        created_at,
                        model,
                        cwd,
                    });
                }
            }
        }
        out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(out)
    }
}

/// 协议修复：末尾的 ToolUse 若无对应 ToolResult，补占位结果。
/// 上次会话在工具执行中途崩溃会产生这种孤儿
fn repair_orphans(messages: Vec<Message>) -> Vec<Message> {
    let Some(last) = messages.last() else {
        return messages;
    };
    if last.role != Role::Assistant {
        return messages;
    }
    let orphans: Vec<String> = last
        .tool_uses()
        .iter()
        .map(|(id, _, _)| id.to_string())
        .collect();
    if orphans.is_empty() {
        return messages;
    }
    let mut messages = messages;
    messages.push(Message {
        role: Role::Tool,
        content: orphans
            .into_iter()
            .map(|id| Block::ToolResult {
                tool_use_id: id,
                content: "（上次会话在此中断，该工具未执行完成）".into(),
                is_error: true,
            })
            .collect(),
    });
    messages
}
