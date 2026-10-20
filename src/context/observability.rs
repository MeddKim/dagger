//! 可观测性：日志初始化 + 用量/成本统计。
//! 日志：tracing → stderr（级别由 -v/-vv 或 RUST_LOG 控制），可选写文件。
//! 指标：进程内累计（AgentStats），REPL /cost 与任务结束摘要展示。

use std::{collections::HashMap, sync::Mutex, time::Duration};

use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

use crate::llm::unified::Usage;

pub fn init_logging(
    verbose: u8,
    log_to_file: bool,
) -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let level = match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("dagger={level}")));

    let stderr_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(false)
        .compact();

    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(stderr_layer);

    if log_to_file {
        let dir = directories::BaseDirs::new()
            .map(|d| d.home_dir().join(".dagger").join("logs"))
            .unwrap_or_else(|| std::path::PathBuf::from(".dagger/logs"));

        let appender = std::fs::create_dir_all(&dir)
            .map_err(|e| e.to_string())
            .and_then(|()| {
                tracing_appender::rolling::RollingFileAppender::builder()
                    .rotation(tracing_appender::rolling::Rotation::NEVER)
                    .filename_prefix("dagger.log")
                    .build(&dir)
                    .map_err(|e| e.to_string())
            });

        match appender {
            Ok(appender) => {
                let (writer, guard) = tracing_appender::non_blocking(appender);
                // 文件层：完整格式（含时间戳/target）
                let file_layer = fmt::layer().with_writer(writer).with_ansi(false);
                registry.with(file_layer).init();
                Some(guard)
            }
            Err(e) => {
                eprintln!("[dagger] 日志文件不可用（{e}），仅输出到 stderr");
                registry.init();
                None
            }
        }
    } else {
        registry.init();
        None
    }
}

/// 单次 LLM 调用的记录
#[derive(Debug, Clone)]
pub struct CallRecord {
    pub model: String,
    pub usage: Usage,
    pub latency: Duration,
}

/// 进程级累计
#[derive(Debug, Default)]
pub struct AgentStats {
    inner: Mutex<StatesInner>,
}
#[derive(Debug, Default)]
struct StatesInner {
    // LLM调用记录（一个进程内所有调用的合集）
    calls: Vec<CallRecord>,
    // 工具名称 - (调用次数, 总耗时)
    tools: HashMap<String, (u64, Duration)>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StatsSnapshot {
    pub calls: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub total_latency: Duration,
}

// 时间点累计指标快照
// 在两个时间点各记录一次 使用 now.since(&pre) 可获取区间增量
impl StatsSnapshot {
    pub fn is_empty(&self) -> bool {
        self.calls == 0
    }

    //上一次到当前 这个区间内的增量
    pub fn since(&self, prev: &StatsSnapshot) -> StatsSnapshot {
        StatsSnapshot {
            calls: self.calls.saturating_sub(prev.calls),
            input_tokens: self.input_tokens.saturating_sub(prev.input_tokens),
            output_tokens: self.output_tokens.saturating_sub(prev.output_tokens),
            cache_read_tokens: self
                .cache_read_tokens
                .saturating_sub(prev.cache_read_tokens),
            total_latency: self.total_latency.saturating_sub(prev.total_latency),
        }
    }

    pub fn render_line(&self) -> String {
        format!(
            "· LLM {} 次 | 输入 {} tok（缓存命中 {}）| 输出 {} tok | 耗时 {:.1}s",
            self.calls,
            self.input_tokens,
            self.cache_read_tokens,
            self.output_tokens,
            self.total_latency.as_secs_f64(),
        )
    }
}

impl AgentStats {
    pub fn new() -> Self {
        Self::default()
    }

    // 记录一次LLM调用
    pub fn record_call(&self, model: &String, usage: &Usage, latency: Duration) {
        self.inner.lock().unwrap().calls.push(CallRecord {
            model: model.clone(),
            usage: usage.clone(),
            latency,
        });
    }

    /// 工具名累计“调用次数 + 总耗时”
    pub fn record_tool(&self, name: &str, latency: Duration) {
        let mut g = self.inner.lock().unwrap();
        let entry = g.tools.entry(name.to_string()).or_default();
        entry.0 += 1;
        entry.1 += latency;
    }

    /// 当前累计指标快照
    pub fn snapshot(&self) -> StatsSnapshot {
        let g = self.inner.lock().unwrap();
        let mut s = StatsSnapshot::default();
        for c in &g.calls {
            s.calls += 1;
            s.input_tokens += c.usage.input_tokens;
            s.output_tokens += c.usage.output_tokens;
            s.cache_read_tokens += c.usage.cache_read_tokens;
            s.total_latency += c.latency;
        }
        s
    }

    pub fn render(&self) -> String {
        let snap = self.snapshot();
        if snap.calls == 0 {
            return "尚未发生 LLM 调用".to_string();
        }

        let mut s = format!(
            "LLM 调用 {} 次 | 输入 {} tok（缓存命中 {}）| 输出 {} tok | 总耗时 {:.1}s ",
            snap.calls,
            snap.input_tokens,
            snap.cache_read_tokens,
            snap.output_tokens,
            snap.total_latency.as_secs_f64(),
        );

        let g = self.inner.lock().unwrap();
        if !g.tools.is_empty() {
            s.push_str("\n工具调用：");
            let mut tools: Vec<_> = g.tools.iter().collect();
            tools.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
            for (name, (count, dur)) in tools {
                s.push_str(&format!(
                    "\n  {name}: {count} 次，平均 {:.0}ms",
                    dur.as_secs_f64() * 1000.0 / (*count as f64)
                ));
            }
        }
        s
    }
}
