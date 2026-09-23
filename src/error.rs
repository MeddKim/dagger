use thiserror::Error;

#[derive(Error, Debug)]
pub enum DaggerError {
    // 模型响应解析失败（协议变更，非预期字段等）
    #[error("模型响应解析失败：{0}")]
    ParseResponse(String),
}
