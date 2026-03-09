use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum EmbeddingInput {
    String(String),
    StringArray(Vec<String>),
    ContentParts(Vec<EmbeddingContentPart>),
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbeddingContentPart {
    Text { text: String },
    ImageUrl { image_url: EmbeddingImageUrl },
    Video { video: EmbeddingVideo },
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingImageUrl {
    pub url: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum EmbeddingVideo {
    Frames(Vec<EmbeddingContentPart>),
    Url { url: String },
}

#[derive(Debug, Deserialize, Clone, Default)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: Option<EncodingFormat>,
    #[serde(default)]
    pub instruction: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: EmbeddingVector,
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
