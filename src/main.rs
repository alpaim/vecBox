mod cli;
mod download;
mod models;
mod utils;

use clap::Parser;
use cli::{Cli, Commands};

fn main() -> anyhow::Result<()> {
    let cli = Cli::try_parse()?;

    let (repo, quant) = match &cli.command {
        Commands::TextEmbedding(args) => (&args.repo, &args.quant),
        Commands::ImageEmbedding(args) => (&args.repo, &args.quant),
    };

    println!("Downloading model from Hugging Face...");
    let downloaded = download::download_model(repo, quant)?;
    println!("Model downloaded successfully!");

    let device = utils::get_device()?;
    let dtype = utils::get_device_dtype(&device)?;

    let mut embedder = models::qwen3::Qwen3VLEmbedding::from_gguf_and_mmproj(
        downloaded.gguf_path,
        downloaded.mmproj_path,
        downloaded.config_dir,
        &device,
        dtype,
    )?;

    println!("Model loaded successfully!");
    println!("Default max_pixels: {}, min_pixels: {}", embedder.max_pixels(), embedder.min_pixels());

    match cli.command {
        Commands::TextEmbedding(args) => {
            let input = utils::resolve_input(&args.input);
            let instruction = args.instruction.map(|i| utils::resolve_input(&i));

            let embeddings =
                embedder.embed_texts_with_instructions(&[input], &[instruction.as_deref()])?;

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
        }
        Commands::ImageEmbedding(args) => {
            // Apply custom pixel settings if provided
            if let Some(max_pixels) = args.max_pixels {
                embedder.set_max_pixels(max_pixels);
                println!("Using max_pixels: {}", max_pixels);
            }
            if let Some(min_pixels) = args.min_pixels {
                embedder.set_min_pixels(min_pixels);
                println!("Using min_pixels: {}", min_pixels);
            }

            let instruction = args.instruction.map(|i| utils::resolve_input(&i));

            let files = utils::get_files_from_directory(&args.input);

            for file in &files {
                println!("Processing: {}", file);
            }

            let instructions: Vec<Option<String>> =
                files.iter().map(|_| instruction.clone()).collect();

            let embeddings = embedder.embed_images_with_instructions(&files, &instructions)?;

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
        }
    }

    Ok(())
}
