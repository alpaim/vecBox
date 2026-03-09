use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
use image::DynamicImage;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};

use crate::api::schema::{
    EmbeddingContentPart, EmbeddingData, EmbeddingInput, EmbeddingRequest, EmbeddingResponse,
    EmbeddingVector, EncodingFormat, ErrorResponse, Usage,
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
    info!("Server listening on {}", addr);

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
    let start = Instant::now();

    let use_base64 = matches!(req.encoding_format, Some(EncodingFormat::Base64));

    let input_count = match &req.input {
        EmbeddingInput::String(_) => 1,
        EmbeddingInput::StringArray(v) => v.len(),
        EmbeddingInput::ContentParts(v) => v.len(),
    };

    info!(
        "Embedding request: {} input(s), model: {}, base64: {}",
        input_count, state.model_name, use_base64
    );

    let instruction = req.instruction;

    match &req.input {
        EmbeddingInput::String(text) => {
            let (embedding, tokens) = process_text(&state.embedder, text, instruction.as_deref())?;
            let embedding = encode_embedding(embedding, use_base64);
            let response = EmbeddingResponse {
                object: "list",
                data: vec![EmbeddingData {
                    object: "embedding",
                    embedding,
                    index: 0,
                }],
                model: state.model_name.clone(),
                usage: Usage {
                    prompt_tokens: tokens,
                    total_tokens: tokens,
                },
            };
            info!(
                "Embedding request completed: {} tokens, {}ms",
                tokens,
                start.elapsed().as_millis()
            );
            Ok(Json(response))
        }
        EmbeddingInput::StringArray(texts) => {
            if texts.is_empty() {
                warn!("Empty embedding request received");
                return Err(AppError(ErrorResponse::new(
                    "Input cannot be empty".to_string(),
                )));
            }
            let mut embeddings = Vec::with_capacity(texts.len());
            let mut total_tokens = 0usize;

            for (idx, text) in texts.iter().enumerate() {
                let (embedding, tokens) =
                    process_text(&state.embedder, text, instruction.as_deref())?;
                embeddings.push(EmbeddingData {
                    object: "embedding",
                    embedding: encode_embedding(embedding, use_base64),
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

            info!(
                "Embedding request completed: {} inputs, {} tokens, {}ms",
                texts.len(),
                total_tokens,
                start.elapsed().as_millis()
            );
            Ok(Json(response))
        }
        EmbeddingInput::ContentParts(parts) => {
            if parts.is_empty() {
                warn!("Empty embedding request received");
                return Err(AppError(ErrorResponse::new(
                    "Input cannot be empty".to_string(),
                )));
            }
            let (embedding, tokens) =
                process_content_parts(&state.embedder, parts, instruction.as_deref())?;
            let embedding = encode_embedding(embedding, use_base64);
            let response = EmbeddingResponse {
                object: "list",
                data: vec![EmbeddingData {
                    object: "embedding",
                    embedding,
                    index: 0,
                }],
                model: state.model_name.clone(),
                usage: Usage {
                    prompt_tokens: tokens,
                    total_tokens: tokens,
                },
            };
            info!(
                "Embedding request completed: {} parts, {} tokens, {}ms",
                parts.len(),
                tokens,
                start.elapsed().as_millis()
            );
            Ok(Json(response))
        }
    }
}

fn encode_embedding(embedding: Vec<f32>, use_base64: bool) -> EmbeddingVector {
    if use_base64 {
        let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        EmbeddingVector::Base64(BASE64_STANDARD.encode(&bytes))
    } else {
        EmbeddingVector::Float(embedding)
    }
}

fn process_text(
    embedder: &Qwen3VLEmbedding,
    text: &str,
    instruction: Option<&str>,
) -> Result<(Vec<f32>, usize), AppError> {
    let encoding = embedder
        .tokenizer()
        .encode(text, true)
        .map_err(|e| AppError(ErrorResponse::new(format!("Tokenization failed: {}", e))))?;
    let token_count = encoding.len();

    let instructions = vec![instruction.map(|s| s.to_string())];
    let embeddings = embedder
        .embed_texts_with_instructions(&[text.to_string()], &instructions)
        .map_err(|e| AppError(ErrorResponse::new(format!("Embedding failed: {}", e))))?;

    Ok((
        embeddings.into_iter().next().unwrap_or_default(),
        token_count,
    ))
}

fn process_content_parts(
    embedder: &Qwen3VLEmbedding,
    parts: &[EmbeddingContentPart],
    instruction: Option<&str>,
) -> Result<(Vec<f32>, usize), AppError> {
    let mut text_parts = Vec::new();
    let mut image_bytes = Vec::new();
    let mut video_count = 0;
    let mut video_frames: Option<Vec<DynamicImage>> = None;
    let mut total_tokens = 0usize;

    for part in parts {
        match part {
            EmbeddingContentPart::Text { text } => {
                let encoding = embedder
                    .tokenizer()
                    .encode(text.to_string(), true)
                    .map_err(|e| {
                        AppError(ErrorResponse::new(format!("Tokenization failed: {}", e)))
                    })?;
                total_tokens += encoding.len();
                text_parts.push(text.clone());
            }
            EmbeddingContentPart::ImageUrl { image_url } => {
                let (bytes, image) = decode_image_url(&image_url.url)?;
                total_tokens += estimate_image_tokens(&image, embedder.max_pixels());
                image_bytes.push(bytes);
            }
            EmbeddingContentPart::Video { video } => {
                let frames = match video {
                    crate::api::schema::EmbeddingVideo::Frames(frame_parts) => {
                        let mut frames = Vec::new();
                        for frame_part in frame_parts {
                            match frame_part {
                                EmbeddingContentPart::ImageUrl { image_url } => {
                                    let (_, img) = decode_image_url(&image_url.url)?;
                                    frames.push(img);
                                }
                                _ => {
                                    return Err(AppError(ErrorResponse::new(
                                        "Video frames must be image URLs".to_string(),
                                    )));
                                }
                            }
                        }
                        frames
                    }
                    crate::api::schema::EmbeddingVideo::Url { url: _ } => {
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

                total_tokens +=
                    frames.len() * estimate_image_tokens(&frames[0], embedder.max_pixels());
                if video_count > 0 {
                    return Err(AppError(ErrorResponse::new(
                        "Multiple videos in a single request are not supported. Please send one video at a time.".to_string(),
                    )));
                }
                video_count += 1;
                video_frames = Some(frames);
            }
        }
    }

    let instructions = vec![instruction.map(|s| s.to_string())];
    let embedding: Vec<f32>;

    if !text_parts.is_empty() && image_bytes.is_empty() && video_frames.is_none() {
        let embeddings = embedder
            .embed_texts_with_instructions(&text_parts, &instructions)
            .map_err(|e| AppError(ErrorResponse::new(format!("Embedding failed: {}", e))))?;
        embedding = embeddings.into_iter().next().unwrap_or_default();
    } else if !image_bytes.is_empty() && text_parts.is_empty() && video_frames.is_none() {
        let image_refs: Vec<&[u8]> = image_bytes.iter().map(|v| v.as_slice()).collect();
        let embeddings = embedder
            .embed_image_bytes_with_instructions(&image_refs, &instructions)
            .map_err(|e| AppError(ErrorResponse::new(format!("Embedding failed: {}", e))))?;
        embedding = embeddings.into_iter().next().unwrap_or_default();
    } else if let Some(frames) = video_frames {
        let embeddings = embedder
            .embed_video_frames(&[frames], &instructions)
            .map_err(|e| AppError(ErrorResponse::new(format!("Embedding failed: {}", e))))?;
        embedding = embeddings.into_iter().next().unwrap_or_default();
    } else {
        return Err(AppError(ErrorResponse::new(
            "Mixed multimodal inputs are not yet supported. Use either text, images, or video frames.".to_string(),
        )));
    }

    Ok((embedding, total_tokens))
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
        let msg = self.0.error.message.clone();
        error!("Embedding request failed: {}", msg);
        (StatusCode::BAD_REQUEST, Json(self.0)).into_response()
    }
}
