mod api;
mod cli;
mod logging;
mod metrics;

use vecbox_core::{download, models, utils};

use clap::Parser;
use cli::{Cli, Commands};
use metrics::{InputType, MetricsBuilder, ModelInfo};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::try_parse()?;

    let _guard = logging::init_logging(&cli.log_level, cli.log_file.clone());

    match &cli.command {
        Commands::Server(args) => {
            run_server(args).await?;
        }
        _ => {
            run_embedding_command(&cli)?;
        }
    }

    Ok(())
}

fn run_embedding_command(cli: &Cli) -> anyhow::Result<()> {
    let (repo, quant, show_metrics) = match &cli.command {
        Commands::TextEmbedding(args) => (&args.repo, &args.quant, args.metrics),
        Commands::ImageEmbedding(args) => (&args.repo, &args.quant, args.metrics),
        Commands::Server(_) => unreachable!(),
    };

    info!("Downloading model from Hugging Face...");
    let load_start = Instant::now();
    let downloaded = download::download_model(repo, quant)?;
    info!("Model downloaded successfully!");

    let device = utils::get_device()?;
    let dtype = utils::get_device_dtype(&device)?;

    let mut embedder = models::qwen3::Qwen3VLEmbedding::from_gguf_and_mmproj(
        &downloaded.gguf_path,
        &downloaded.mmproj_path,
        &downloaded.config_dir,
        &device,
        dtype,
    )?;

    let model_load_time_ms = load_start.elapsed().as_millis() as u64;

    info!("Model loaded successfully!");

    let model_info = ModelInfo::new(
        repo.to_string(),
        quant.to_string(),
        format!("{:?}", device),
        format!("{:?}", dtype),
        embedder.config().hidden_size,
        &downloaded.gguf_path,
        &downloaded.mmproj_path,
        embedder.max_pixels(),
        embedder.min_pixels(),
    );
    model_info.print();

    match &cli.command {
        Commands::TextEmbedding(args) => {
            let input = utils::resolve_input(&args.input);
            let instruction = args.instruction.as_ref().map(|i| utils::resolve_input(i));

            let encoding = embedder
                .tokenizer()
                .encode(input.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
            let token_count = encoding.len();

            let builder =
                MetricsBuilder::new(InputType::Text).with_model_load_time(model_load_time_ms);

            let embeddings =
                embedder.embed_texts_with_instructions(&[&input], &[instruction.as_deref()])?;

            let metrics = builder.finish_with_tokens(embeddings.len(), token_count);

            for (i, emb) in embeddings.iter().enumerate() {
                println!(
                    "Text {}: Embedded into vector of size {} (First 5 values: {:.4}, {:.4}, {:.4}, {:.4}, {:.4})",
                    i,
                    emb.len(),
                    emb[0],
                    emb[1],
                    emb[2],
                    emb[3],
                    emb[4]
                );
            }

            if show_metrics {
                print!("{}", metrics.format_human());
            }
        }
        Commands::ImageEmbedding(args) => {
            if let Some(max_pixels) = args.max_pixels {
                embedder.set_max_pixels(max_pixels);
                info!("Using max_pixels: {}", max_pixels);
            }
            if let Some(min_pixels) = args.min_pixels {
                embedder.set_min_pixels(min_pixels);
                info!("Using min_pixels: {}", min_pixels);
            }

            let instruction = args.instruction.as_ref().map(|i| utils::resolve_input(i));

            let files = utils::get_files_from_directory(&args.input);

            for file in &files {
                info!("Processing: {}", file);
            }

            let instructions: Vec<Option<&str>> =
                files.iter().map(|_| instruction.as_deref()).collect();

            let builder =
                MetricsBuilder::new(InputType::Image).with_model_load_time(model_load_time_ms);

            let embeddings = embedder.embed_images_with_instructions(&files, &instructions)?;

            let metrics = builder.finish(embeddings.len());

            for (i, emb) in embeddings.iter().enumerate() {
                println!(
                    "Image {}: Embedded into vector of size {} (First 5 values: {:.4}, {:.4}, {:.4}, {:.4}, {:.4})",
                    i,
                    emb.len(),
                    emb[0],
                    emb[1],
                    emb[2],
                    emb[3],
                    emb[4]
                );
            }

            if show_metrics {
                print!("{}", metrics.format_human());
            }
        }
        Commands::Server(_) => unreachable!(),
    }

    Ok(())
}

async fn run_server(args: &cli::ServerArgs) -> anyhow::Result<()> {
    info!("Downloading model from Hugging Face...");
    let downloaded = download::download_model(&args.repo, &args.quant)?;
    info!("Model downloaded successfully!");

    let device = utils::get_device()?;
    let dtype = utils::get_device_dtype(&device)?;

    let mut embedder = models::qwen3::Qwen3VLEmbedding::from_gguf_and_mmproj(
        &downloaded.gguf_path,
        &downloaded.mmproj_path,
        &downloaded.config_dir,
        &device,
        dtype,
    )?;

    if let Some(max_pixels) = args.max_pixels {
        embedder.set_max_pixels(max_pixels);
        info!("Using max_pixels: {}", max_pixels);
    }
    if let Some(min_pixels) = args.min_pixels {
        embedder.set_min_pixels(min_pixels);
        info!("Using min_pixels: {}", min_pixels);
    }

    info!("Model loaded successfully!");
    info!(
        "Model: {} (hidden_size: {})",
        args.repo,
        embedder.config().hidden_size
    );

    let model_name = format!("{}-{}", args.repo, args.quant);
    let state = api::AppState {
        embedder: Arc::new(embedder),
        model_name,
    };

    api::run_server(state, &args.host, args.port).await
}
