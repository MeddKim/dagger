use async_trait::async_trait;

use crate::error::Result;
use crate::llm::unified::{EventStream, UnifiedRequest, UnifiedResponse};
use crate::llm::{anthropic, openai_chat, openai_responses};

// ===============================================
// Provider trait 定义 + 三种协议的实现
// ================================================
/// provider抽象
#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(&self, req: &UnifiedRequest) -> Result<UnifiedResponse>;
    async fn complete_stream(&self, req: &UnifiedRequest) -> Result<EventStream>;
    fn name(&self) -> String;
}

#[async_trait]
impl Provider for openai_chat::OpenAIChatClient {
    async fn complete(&self, req: &UnifiedRequest) -> Result<UnifiedResponse> {
        self.complete(req).await
    }
    async fn complete_stream(&self, req: &UnifiedRequest) -> Result<EventStream> {
        self.complete_stream(req).await
    }
    fn name(&self) -> String {
        "openai-chat".into()
    }
}

#[async_trait]
impl Provider for openai_responses::OpenAIResponsesClient {
    async fn complete(&self, req: &UnifiedRequest) -> Result<UnifiedResponse> {
        self.complete(req).await
    }
    async fn complete_stream(&self, req: &UnifiedRequest) -> Result<EventStream> {
        self.complete_stream(req).await
    }
    fn name(&self) -> String {
        "openai-responses".into()
    }
}

#[async_trait]
impl Provider for anthropic::AnthropicClient {
    async fn complete(&self, req: &UnifiedRequest) -> Result<UnifiedResponse> {
        self.complete(req).await
    }
    async fn complete_stream(&self, req: &UnifiedRequest) -> Result<EventStream> {
        self.complete_stream(req).await
    }
    fn name(&self) -> String {
        "anthropic".into()
    }
}

// ===============================================
// Provider 工厂，通过配置实例指定Provider
// ================================================
#[derive(Debug, Clone)]
pub enum ProviderKind {
    OpenAIChat,
    OpenAIResponses,
    Anthropic,
}

impl ProviderKind {
    /// 定义规范名称
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::OpenAIResponses => "openai-responses",
            ProviderKind::OpenAIChat => "openai-chat",
        }
    }

    /// 从配置字符串解析（容错：大小写、常见别名）
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "openaichat" | "openai" | "chat" => Ok(Self::OpenAIChat),
            "openairesponses" | "responses" => Ok(Self::OpenAIResponses),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            other => Err(crate::error::DaggerError::Config(format!(
                "未知 provider: {other}（可选：openai-chat / openai-responses / anthropic）"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub api_key: String,
    pub base_url: Option<String>,
    pub model: String,
}

/// 根据配置，构建对应的Provider
pub fn build_provider(cfg: &ProviderConfig) -> Box<dyn Provider> {
    match cfg.kind {
        ProviderKind::OpenAIChat => Box::new(openai_chat::OpenAIChatClient::new(
            cfg.api_key.clone(),
            cfg.base_url.clone(),
        )),
        ProviderKind::OpenAIResponses => Box::new(openai_responses::OpenAIResponsesClient::new(
            cfg.api_key.clone(),
            cfg.base_url.clone(),
        )),
        ProviderKind::Anthropic => Box::new(anthropic::AnthropicClient::new(
            cfg.api_key.clone(),
            cfg.base_url.clone(),
        )),
    }
}

/// 便捷构造：从环境变量组装配置
///
/// 环境变量约定：
///   DAGGER_PROVIDER   openai-chat / openai-responses / anthropic（默认 openai-chat）
///   OPENAI_API_KEY / ANTHROPIC_API_KEY
///   OPENAI_BASE_URL / ANTHROPIC_BASE_URL（可选）
///   DAGGER_MODEL      模型名（按 provider 给默认值）
pub fn provider_from_env() -> Result<ProviderConfig> {
    let kind = std::env::var("DAGGER_PROVIDER")
        .ok()
        .map(|s| ProviderKind::parse(&s))
        .transpose()?
        .unwrap_or(ProviderKind::OpenAIChat);

    let (key_var, url_var, default_model) = match kind {
        ProviderKind::OpenAIChat | ProviderKind::OpenAIResponses => {
            ("OPENAI_API_KEY", "OPENAI_BASE_URL", "deepseek-v4-flash")
        }
        ProviderKind::Anthropic => (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "deepseek-v4-flash",
        ),
    };

    let api_key = std::env::var(key_var)
        .map_err(|_| crate::error::DaggerError::Config(format!("缺少环境变量 {key_var}")))?;
    let base_url = std::env::var(url_var).ok();
    let model = std::env::var("DAGGER_MODEL").unwrap_or_else(|_| default_model.to_string());

    Ok(ProviderConfig {
        kind,
        api_key,
        base_url,
        model,
    })
}
