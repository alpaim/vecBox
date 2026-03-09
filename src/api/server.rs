use std::io::Cursor;
use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
use image::DynamicImage;
use tower_http::cors::{Any, CorsLayer};

use crate::api::schema::{
    EmbeddingData, EmbeddingInput, EmbeddingRequest, EmbeddingResponse, ErrorResponse,
    InputContent, Usage, VideoInput,
};
use crate::models::qwen3::Qwen3VLEmbedding;

pub struct AppState {
    pub embedder: Arc<Qwen3VLEmbedding>,
    pub model_name: String,
}

pub async fn run_server(state: AppState, host: &str, port: u16) -> anyhow::Result<()> {
    let app = axum::Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embedding))
        .route("/health", axum::routing::get(health_check))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(Arc::new(state));

    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("Server listening on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

pub async fn create_embedding(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, AppError> {
    let inputs = match req.input {
        EmbeddingInput::Single(content) => vec![content],
        EmbeddingInput::Multiple(contents) => contents,
    };

    if inputs.is_empty() {
        return Err(AppError(ErrorResponse::new(
            "Input cannot be empty".to_string(),
        )));
    }

    let instruction = req.instruction;
    let mut embeddings = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0usize;

    for (idx, input) in inputs.into_iter().enumerate() {
        let (embedding, tokens) =
            process_single_input(&state.embedder, input, instruction.as_deref())?;
        embeddings.push(EmbeddingData {
            object: "embedding",
            embedding,
            index: idx,
        });
        total_tokens += tokens;
    }

    let response = EmbeddingResponse {
        object: "list",
        data: embeddings,
        model: state.model_name.clone(),
        usage: Usage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    Ok(Json(response))
}

fn process_single_input(
    embedder: &Qwen3VLEmbedding,
    input: InputContent,
    instruction: Option<&str>,
) -> Result<(Vec<f32>, usize), AppError> {
    match input {
        InputContent::Text { text } => {
            let encoding = embedder
                .tokenizer()
                .encode(text.as_str(), true)
                .map_err(|e| AppError(ErrorResponse::new(format!("Tokenization failed: {}", e))))?;
            let token_count = encoding.len();

            let instructions = vec![instruction.map(|s| s.to_string())];
            let embeddings = embedder
                .embed_texts_with_instructions(&[text], &instructions)
                .map_err(|e| AppError(ErrorResponse::new(format!("Embedding failed: {}", e))))?;

            Ok((
                embeddings.into_iter().next().unwrap_or_default(),
                token_count,
            ))
        }
        InputContent::ImageUrl { image_url } => {
            let (bytes, image) = decode_image_url(&image_url.url)?;
            let token_count = estimate_image_tokens(&image, embedder.max_pixels());

            let instructions = vec![instruction.map(|s| s.to_string())];
            let embeddings = embedder
                .embed_image_bytes_with_instructions(&[bytes.as_slice()], &instructions)
                .map_err(|e| AppError(ErrorResponse::new(format!("Embedding failed: {}", e))))?;

            Ok((
                embeddings.into_iter().next().unwrap_or_default(),
                token_count,
            ))
        }
        InputContent::Video { video } => {
            let frames = match video {
                VideoInput::Frames(frame_contents) => {
                    let mut frames = Vec::new();
                    for frame_content in frame_contents {
                        match frame_content {
                            InputContent::ImageUrl { image_url } => {
                                let (_, img) = decode_image_url(&image_url.url)?;
                                frames.push(img);
                            }
                            InputContent::Text { text } => {
                                return Err(AppError(ErrorResponse::new(format!(
                                    "Video frames cannot contain text: {}",
                                    text
                                ))));
                            }
                            InputContent::Video { .. } => {
                                return Err(AppError(ErrorResponse::new(
                                    "Nested video inputs are not supported".to_string(),
                                )));
                            }
                        }
                    }
                    frames
                }
                VideoInput::Url { url: _ } => {
                    return Err(AppError(ErrorResponse::new(
                        "Video URL downloading is not yet implemented. Please provide video as an array of image frames.".to_string(),
                    )));
                }
            };

            if frames.is_empty() {
                return Err(AppError(ErrorResponse::new(
                    "Video must contain at least one frame".to_string(),
                )));
            }

            let token_count =
                frames.len() * estimate_image_tokens(&frames[0], embedder.max_pixels());

            let instructions = vec![instruction.map(|s| s.to_string())];
            let embeddings = embedder
                .embed_video_frames(&[frames], &instructions)
                .map_err(|e| AppError(ErrorResponse::new(format!("Embedding failed: {}", e))))?;

            Ok((
                embeddings.into_iter().next().unwrap_or_default(),
                token_count,
            ))
        }
    }
}

fn decode_image_url(url: &str) -> Result<(Vec<u8>, DynamicImage), AppError> {
    if let Some(base64_data) = url.strip_prefix("data:") {
        if let Some((_, data)) = base64_data.split_once(";base64,") {
            let bytes = BASE64_STANDARD.decode(data).map_err(|e| {
                AppError(ErrorResponse::new(format!("Base64 decode failed: {}", e)))
            })?;
            let image = image::ImageReader::new(Cursor::new(&bytes))
                .with_guessed_format()
                .map_err(|e| {
                    AppError(ErrorResponse::new(format!(
                        "Image format detection failed: {}",
                        e
                    )))
                })?
                .decode()
                .map_err(|e| AppError(ErrorResponse::new(format!("Image decode failed: {}", e))))?;
            return Ok((bytes, image));
        }
    }

    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(AppError(ErrorResponse::new(
            "Remote image URLs are not supported. Please provide images as base64 data URLs."
                .to_string(),
        )));
    }

    let bytes = BASE64_STANDARD
        .decode(url)
        .map_err(|e| AppError(ErrorResponse::new(format!("Base64 decode failed: {}", e))))?;
    let image = image::ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|e| {
            AppError(ErrorResponse::new(format!(
                "Image format detection failed: {}",
                e
            )))
        })?
        .decode()
        .map_err(|e| AppError(ErrorResponse::new(format!("Image decode failed: {}", e))))?;

    Ok((bytes, image))
}

fn estimate_image_tokens(image: &DynamicImage, max_pixels: usize) -> usize {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let pixels = w * h;
    let scale = if pixels > max_pixels {
        (max_pixels as f64 / pixels as f64).sqrt()
    } else {
        1.0
    };
    let scaled_w = (w as f64 * scale) as usize;
    let scaled_h = (h as f64 * scale) as usize;
    let patch_size = 14usize;
    let patches_h = (scaled_h + patch_size - 1) / patch_size;
    let patches_w = (scaled_w + patch_size - 1) / patch_size;
    patches_h * patches_w
}

#[derive(Debug)]
pub struct AppError(ErrorResponse);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(self.0)).into_response()
    }
}
