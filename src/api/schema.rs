use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(InputContent),
    Multiple(Vec<InputContent>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    Video { video: VideoInput },
}

#[derive(Debug, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum VideoInput {
    Frames(Vec<InputContent>),
    Url { url: String },
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: Option<String>,
    pub instruction: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
    pub code: Option<&'static str>,
}

impl ErrorResponse {
    pub fn new(message: String) -> Self {
        ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: "invalid_request_error",
                code: None,
            },
        }
    }
}
