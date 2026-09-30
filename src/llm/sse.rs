use eventsource_stream::Eventsource;
use futures_util::{Stream, StreamExt};
use reqwest::Response;

use crate::error::{DaggerError, Result};

//SSE事件，单帧的数据
pub struct SseEvent {
    // OpenAI Chat Completions 协议 无该字段 -> None
    // OpenAI Response 有：   response.created / response.in_progress
    //                       response.output_item.added / response.output_item.done
    //                      response.content_part.added / response.content_part.done
    //                      response.reasoning_text.delta / response.reasoning_text.done
    //                      response.function_call_arguments.delta / response.function_call_arguments.done
    //                      response.completed
    // Anthropic 有：    message_start / message_delta / message_stop
    //                  content_block_start / content_block_delta / content_block_stop
    //
    pub event: Option<String>,
    // 每一帧的具体内容
    pub data: String,
}

pub fn into_sse_stream(resp: Response) -> impl Stream<Item = Result<SseEvent>> + Send {
    resp.bytes_stream().eventsource().map(|item| match item {
        Ok(ev) => Ok(SseEvent {
            event: (ev.event != "message").then_some(ev.event),
            data: ev.data,
        }),
        Err(e) => Err(DaggerError::Parse(format!("SSE 流解析失败：{e}"))),
    })
}
