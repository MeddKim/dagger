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
}

pub type Result<T> = std::result::Result<T, DaggerError>;
