use rustyline::error::ReadlineError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DaggerError {
    // 模型响应解析失败（协议变更，非预期字段等）
    #[error("模型响应解析失败：{0}")]
    ParseResponse(String),

    #[error("模型 API 错误 错误（HTTP {status}: {message}）")]
    Api { status: u16, message: String },

    #[error("网络请求失败 {0}")]
    Http(#[from] reqwest::Error),

    // 模型响应解析失败（协议变更，非预期字段等）
    #[error("模型响应解析失败：{0}")]
    Parse(String),

    #[error("JSON 序列化/反序列化错误：{0}")]
    Json(#[from] serde_json::Error),

    // 模型响应解析失败（协议变更，非预期字段等）
    #[error("配置异常：{0}")]
    Config(String),

    #[error("超过最大步数")]
    MaxStepsExceeded(usize),

    #[error("终端读取异常:{0}")]
    Readline(#[from] ReadlineError),
}

pub type Result<T> = std::result::Result<T, DaggerError>;
