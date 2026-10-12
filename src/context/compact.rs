//! 对话压缩：把中间部分提交给模型做信息摘要（注意此处可使用便宜模型）
//! 未能正常获取到摘要信息，使用方需要做人工处理
use crate::{
    error::{DaggerError, Result},
    llm::unified::{Message, Role, UnifiedRequest},
    provider::Provider,
};

///摘要提示词，告诉模型做摘要的规则
///这段提示词质量决定后续重建内容的准确性和连贯性
const SUMMARIZE_PROMPT: &str = r#"上面的对话是一个编程 Agent 的工作记录。请将它压缩成一份"接续工作简报"，供下一轮对话的 Agent 无缝接手。

必须保留（按重要性排序）：
1. 【任务目标】用户最初和最新的需求是什么；
2. 【当前进度】进行到哪一步、下一步准备做什么；
3. 【关键发现】代码结构、bug 根因、重要文件路径、约束条件；
4. 【已完成的修改】改动了哪些文件的哪些部分；
5. 【失败的尝试】试过但失败的方案及原因（避免重复踩坑）。

必须丢弃：
- 工具输出的逐字内容（只留结论）；
- 客套话与中间推理过程；
- 已被后续操作覆盖的早期操作。

输出格式：分节的简洁清单，不超过 500 字。直接输出简报内容，不要任何前后缀。"#;

/// 做历史消息摘要
/// [0] 首条（原始任务一定保留）
/// [1-keep_recent.len()] 中间压缩区
/// [keep_recent.len()...] 最近keep_recent条不处理
pub async fn compact(
    provider: &dyn Provider,
    model: &str,
    messages: &[Message],
    keep_recent: usize,
) -> Result<Vec<Message>> {
    //消息过少，压缩无价值
    if messages.len() <= keep_recent + 2 {
        return Ok(messages.to_vec());
    }
    //处理位置
    let split_at = messages.len() - keep_recent;
    //第一条消息
    let first = messages[0].clone();
    //中间要压缩的内容
    let midddle = &messages[1..split_at];
    //最近的recent条不处理
    let recent = &messages[split_at..];

    let mut summarize_msgs = midddle.to_vec();
    summarize_msgs.push(Message::user(SUMMARIZE_PROMPT));

    let req = UnifiedRequest {
        model: model.to_string(),
        system: Some("你是对话压缩器，只做客观摘要，不发挥不评论。".into()),
        tools: vec![],
        temperature: Some(0.0), //做摘要不需要创造性
        max_tokens: 1024,
        messages: summarize_msgs,
        thinking: false,
    };

    let summary = provider.complete(&req).await?.message.text();
    if summary.trim().is_empty() {
        //为正常获取到摘要，返回错误，让上游做人工截断
        return Err(DaggerError::Parse("摘要结果为空".into()));
    }

    //重建历史消息结构为 use(原始任务 + 简报) + assistant + 保留的最近消息
    let mut rebuilt = Vec::with_capacity(keep_recent + 3);
    rebuilt.push(Message::user(format!(
        "{}\n\n═══ 早期工作简报（对话压缩生成）═══\n{}",
        first.text(),
        summary
    )));
    rebuilt.push(Message::assistant(
        "我已了解上述任务背景与前期工作简报，将在此基础上继续。".to_string(),
    ));

    let mut recent = recent.to_vec();
    if let Some(first_recent) = recent.first_mut() {
        if first_recent.role == Role::Tool {
            *first_recent = Message::user("（此处的工具调用细节已随早期对话被压缩）");
        }
    }
    rebuilt.extend(recent);

    Ok(rebuilt)
}
