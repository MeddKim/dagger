use serde_json::{Value, json};

use crate::{error::Result, llm::unified::ToolDef};

#[derive(Default)]
pub struct ToolRegistry {}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    ///加载所有定义好的工具
    pub fn load_tools(&self) -> Vec<ToolDef> {
        vec![ToolDef {
            name: "get_weather".into(),
            description: "获取指定城市的天气".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "location": {
                      "type": "string",
                      "description": "城市名称，例如：北京"
                    },
                    "unit": {
                      "type": "string",
                      "enum": ["celsius", "fahrenheit"],
                      "description": "温度单位，默认为摄氏度"
                    }
                },
                "required": ["location"]
            }),
        }]
    }

    /// 执行工具
    pub async fn execute(&self, name: &str, args: &Value) -> Result<String> {
        match name {
            "get_weather" => Ok(format!("天气晴，气温21度")),
            other => Ok(format!("错误：未知工具 {other}")),
        }
    }
}
