use std::time::Duration;

use crate::error::{DaggerError, Result};

/// 重试策略配置
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// 最大尝试次数: 5（含首次）。
    pub max_attempts: usize,
    /// 基础等待时间：500ms
    pub base_delay: Duration,
    /// 等待时间上限 30s
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
        }
    }
}

/// 重试策略（决定了是否重试）
#[derive(Debug)]
pub enum RetryAction {
    /// 直接重试（401/400/协议解析错误）
    Fail,
    /// 等待后重试（after为等待时长）
    Retry { after: Option<Duration> },
}

/// 解析重试策略
///
/// 分类表
/// ┌─────────────────────────────┬─────────┐
/// │ 错误                         │ 决策    │
/// ├─────────────────────────────┼─────────┤
/// │ HTTP 超时/连接错误/重置       │ Retry    │
/// │ 429 Too Many Requests       │ Retry   │
/// │ 500/502/503/529             │ Retry   │
/// │ 401/403/404/400             │ Fail    │
/// │ 响应解析失败                  │ Fail    │
/// │ 工具执行失败/权限拒绝          │ Fail    │
/// └─────────────────────────────┴─────────┘
pub fn classify(err: &DaggerError) -> RetryAction {
    match err {
        DaggerError::Http(e) => {
            if e.is_timeout() || e.is_connect() {
                RetryAction::Retry { after: None }
            } else if e.is_request() {
                RetryAction::Fail
            } else {
                RetryAction::Retry { after: None }
            }
        }
        DaggerError::Api { status, .. } => match status {
            429 => RetryAction::Retry { after: None },
            500 | 502 | 529 => RetryAction::Retry { after: None },
            _ => RetryAction::Fail,
        },
        _ => RetryAction::Fail,
    }
}

// 计算第 n 次 请求需要等待的时间
fn backoff_delay(policy: &RetryPolicy, attempt: usize) -> Duration {
    let exp = policy
        .base_delay
        .saturating_mul(1u32 << attempt.saturating_sub(1).min(10));
    let cap = exp.min(policy.max_delay);
    // 简单的无依赖随机：用系统时间纳秒做伪随机源
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let jittered = nanos % (cap.as_millis() as u64 + 1);
    Duration::from_millis(jittered).max(Duration::from_millis(50))
}

pub async fn with_retry<T, F, Fut>(op_name: &str, mut op: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let policy = RetryPolicy::default();
    let mut last_err: Option<DaggerError> = None;

    for attempt in 1..=policy.max_attempts {
        match op().await {
            Ok(v) => {
                if attempt > 1 {
                    tracing::info!(op = op_name, attempt, "请求重试后成功");
                }
                return Ok(v);
            }
            Err(e) => match classify(&e) {
                RetryAction::Fail => {
                    tracing::warn!(op = op_name, error = %e, "不确定性错误，不予重试");
                    return Err(e);
                }
                RetryAction::Retry { after } => {
                    if attempt == policy.max_attempts {
                        tracing::error!(op = op_name, attempts = attempt, error = %e, "重试次数耗尽");
                        last_err = Some(e);
                        break;
                    }
                    let wait = after.unwrap_or_else(|| backoff_delay(&policy, attempt));
                    tracing::warn!(
                        op = op_name,
                        attempts = attempt,
                        wait_ms = wait.as_millis() as u64,
                        error = %e,
                        "请求错误，稍后重试"
                    );
                    tokio::time::sleep(wait).await;
                    last_err = Some(e);
                }
            },
        }
    }
    Err(last_err.expect("重试次数耗尽"))
}
