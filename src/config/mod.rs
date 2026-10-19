//! 配置优先级：默认值 < 环境变量 < os钥匙串 < 用户配置 < 项目配置 < CLI 参数
//!
//!
//!
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::{
    error::{DaggerError, Result},
    provider::{ProviderConfig, ProviderKind},
};

/// 配置中支持的参数
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FileConfig {
    /// 服务商
    pub provider: Option<String>,
    /// 模型名称
    pub model: Option<String>,
    /// 请求地址
    pub base_url: Option<String>,
    /// 秘钥
    pub api_key: Option<String>,
}

/// 配置项来源
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Default,
    Env,
    /// OS 钥匙串（仅用于密钥）
    Keychain,
    UserFile,
    ProjectFile,
    Cli,
}

impl Source {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Default => "默认",
            Self::Env => "环境变量",
            Self::Keychain => "钥匙串",
            Self::UserFile => "用户配置",
            Self::ProjectFile => "项目配置",
            Self::Cli => "命令行",
        }
    }
}

/// 不同来源的值
#[derive(Debug, Clone)]
pub struct Sourced<T> {
    pub value: T,
    pub source: Source,
}

impl<T: Clone> Sourced<T> {
    fn new(value: T, source: Source) -> Self {
        Self { value, source }
    }

    fn override_with(&mut self, candidate: Option<T>, source: Source) {
        // 能够结构表示某层配置是有值的，覆盖
        // 所以调用该方法的地方必须保证是按照各层优先级顺序调用的
        if let Some(v) = candidate {
            self.value = v;
            self.source = source;
        }
    }
}

impl<T: Clone> Sourced<Option<T>> {
    fn override_opt(&mut self, candidate: Option<T>, source: Source) {
        if let Some(v) = candidate {
            self.value = Some(v);
            self.source = source;
        }
    }
}

///完成合并后的最终配置
#[derive(Debug)]
pub struct Config {
    pub provider: Sourced<String>,
    pub model: Sourced<String>,
    pub base_url: Sourced<Option<String>>,
    pub api_key: Sourced<Option<String>>,
}

impl Config {
    pub fn load(project_root: &Path, cli: &crate::cli::GlobalOpts) -> Result<Self> {
        // 默认值 Default
        let mut provider = Sourced::new("open-chat".to_string(), Source::Default);
        let mut model = Sourced::new("gpt-4o-mini".to_string(), Source::Default);
        let mut base_url = Sourced::new(None, Source::Default);
        let mut api_key = Sourced::new(None, Source::Default);

        //环境变量
        // 通用变量优先，provider 专属变量兜底
        provider.override_with(std::env::var("DAGGER_PROVIDER").ok(), Source::Env);
        model.override_with(std::env::var("DAGGER_MODEL").ok(), Source::Env);
        let provider_url = match ProviderKind::parse(&provider.value) {
            Ok(ProviderKind::Anthropic) => std::env::var("ANTHROPIC_BASE_URL").ok(),
            _ => std::env::var("OPENAI_BASE_URL").ok(),
        };
        base_url.override_opt(
            std::env::var("DAGGER_BASE_URL").ok().or(provider_url),
            Source::Env,
        );
        api_key.override_opt(Self::api_key_from_env(&provider.value), Source::Env);

        //钥匙串
        if api_key.value.is_none() {
            api_key.override_opt(keychain_get(&provider.value), Source::Keychain);
        }

        // 用户配置: ~/.dagger/config.toml
        let user_file = Self::user_config_path();
        if let Some(fc) = Self::read_file_config(&user_file)? {
            provider.override_with(fc.provider, Source::UserFile);
            model.override_with(fc.model, Source::UserFile);
            base_url.override_opt(fc.base_url, Source::UserFile);
            api_key.override_opt(fc.api_key, Source::UserFile);
        }

        // 项目配置 {root}/.dagger/config.toml
        let proj_file = project_root.join(".dagger/config.toml");
        if let Some(fc) = Self::read_file_config(&proj_file)? {
            if fc.api_key.is_some() {
                return Err(DaggerError::Config(format!(
                    "{} 含 api_key 字段，已拒绝启动。\n\
                     项目配置可能随代码库一起提交/上传，密钥放在这里有泄漏风险。\n\
                     请把密钥移到以下任一安全位置：\n\
                     \x20 1. 运行 `dagger config set-key` 存入系统钥匙串；\n\
                     \x20 2. 环境变量 DAGGER_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY；\n\
                     \x20 3. 用户级配置 ~/.dagger/config.toml（不入库）。\n\
                     然后删除项目配置中的 api_key 字段。",
                    proj_file.display()
                )));
            }
            provider.override_with(fc.provider, Source::ProjectFile);
            model.override_with(fc.model, Source::ProjectFile);
            base_url.override_opt(fc.base_url, Source::ProjectFile);
        }

        //Cli参数，最高优先级
        provider.override_with(cli.provider.clone(), Source::Cli);
        model.override_with(cli.model.clone(), Source::Cli);
        base_url.override_opt(cli.base_url.clone(), Source::Cli);
        api_key.override_opt(cli.api_key.clone(), Source::Cli);

        Ok(Self {
            provider,
            model,
            base_url,
            api_key,
        })
    }

    //用户配置地址
    fn user_config_path() -> PathBuf {
        directories::BaseDirs::new()
            .map(|d| d.home_dir().join(".dagger/config.toml"))
            .unwrap_or_else(|| PathBuf::from(".dagger/config.toml"))
    }

    ///读取解析toml配置，不存在返回Ok(None)，损坏返回 Err
    fn read_file_config(path: &Path) -> Result<Option<FileConfig>> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text)
                .map(Some)
                .map_err(|e| DaggerError::Config(format!("{} 解析失败：{e}", path.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// provider 对应的专属密钥环境变量名
    fn api_key_from_env(provider: &str) -> Option<String> {
        // 通用变量优先
        if let Ok(k) = std::env::var("DAGGER_API_KEY") {
            return Some(k);
        }
        match ProviderKind::parse(provider).ok()? {
            ProviderKind::Anthropic => std::env::var("ANTHROPIC_API_KEY").ok(),
            _ => std::env::var("OPENAI_API_KEY").ok(),
        }
    }

    /// ProviderConfig
    pub fn to_provider_config(&self) -> Result<ProviderConfig> {
        let kind = ProviderKind::parse(&self.provider.value)?;
        let api_key = self.api_key.value.clone().ok_or_else(|| {
            DaggerError::Config(format!(
                "未找到 API Key。请设置环境变量（{}）、\
                 或运行 dagger config set-key 存入钥匙串",
                match kind {
                    ProviderKind::Anthropic => "ANTHROPIC_API_KEY",
                    _ => "OPENAI_API_KEY",
                }
            ))
        })?;
        Ok(ProviderConfig {
            kind,
            api_key,
            base_url: self.base_url.value.clone(),
            model: self.model.value.clone(),
        })
    }
    /// `config show` 的渲染：值 + 来源，密钥脱敏
    pub fn render(&self) -> String {
        format!(
            "provider  = {}（来自：{}）\nmodel     = {}（来自：{}）\nbase_url  = {}（来自：{}）\napi_key   = {}（来自：{}）",
            self.provider.value,
            self.provider.source.label(),
            self.model.value,
            self.model.source.label(),
            self.base_url.value.as_deref().unwrap_or("（默认端点）"),
            self.base_url.source.label(),
            mask_secret(self.api_key.value.as_deref()),
            self.api_key.source.label(),
        )
    }
}

/// 密钥脱敏：只露末 4 位
pub fn mask_secret(key: Option<&str>) -> String {
    match key {
        None => "（未配置）".to_string(),
        Some(k) => {
            // 按字符取末 4 位，避免按字节切导致非 ASCII panic
            let n = k.chars().count();
            if n > 4 {
                let tail: String = k.chars().skip(n - 4).collect();
                format!("****{tail}")
            } else {
                "****".to_string()
            }
        }
    }
}

// ────────────────────────────────────────────────────────────
// OS 钥匙串
// ────────────────────────────────────────────────────────────

/// 钥匙串里的服务名与条目名约定
const KEYRING_SERVICE: &str = "dev.dagger";

fn keyring_entry(provider: &str) -> Option<keyring::Entry> {
    keyring::Entry::new(KEYRING_SERVICE, provider)
        .map_err(|e| tracing::warn!("钥匙串不可用: {e}"))
        .ok()
}

/// 从钥匙串读取（任何失败都静默降级为 None——钥匙串是增强不是必需）
pub fn keychain_get(provider: &str) -> Option<String> {
    keyring_entry(provider)?.get_password().ok()
}

/// 写入钥匙串（dagger config set-key）
pub fn keychain_set(provider: &str, key: &str) -> Result<()> {
    keyring_entry(provider)
        .ok_or_else(|| DaggerError::Config("钥匙串服务不可用".into()))?
        .set_password(key)
        .map_err(|e| DaggerError::Config(format!("写入钥匙串失败: {e}")))
}

/// 从钥匙串删除（dagger config delete-key）
pub fn keychain_delete(provider: &str) -> Result<()> {
    keyring_entry(provider)
        .ok_or_else(|| DaggerError::Config("钥匙串服务不可用".into()))?
        .delete_credential()
        .map_err(|e| DaggerError::Config(format!("删除失败: {e}")))
}
