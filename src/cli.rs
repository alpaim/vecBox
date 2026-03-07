use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "vecbox")]
#[command(about = "Embedding tool for text and images using Qwen3-VL", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    #[command(name = "text-embedding")]
    TextEmbedding(TextEmbeddingArgs),
    #[command(name = "image-embedding")]
    ImageEmbedding(ImageEmbeddingArgs),
}

#[derive(clap::Args)]
#[command(args_conflicts_with_subcommands = true)]
pub struct TextEmbeddingArgs {
    #[arg(
        long,
        help = "Hugging Face repo ID (e.g., 'alpaim/Qwen3-VL-Embedding-2B-GGUF-vecBox')"
    )]
    pub repo: String,

    #[arg(
        long,
        default_value = "Q4_K_M",
        help = "GGUF quant to download (e.g., Q4_K_M, Q8_0)"
    )]
    pub quant: String,

    #[arg(long, help = "Input text or path to text file")]
    pub input: String,

    #[arg(long, help = "Instruction text or path to instruction file")]
    pub instruction: Option<String>,
}

#[derive(clap::Args)]
#[command(args_conflicts_with_subcommands = true)]
pub struct ImageEmbeddingArgs {
    #[arg(
        long,
        help = "Hugging Face repo ID (e.g., 'alpaim/Qwen3-VL-Embedding-2B-GGUF-vecBox')"
    )]
    pub repo: String,

    #[arg(
        long,
        default_value = "Q4_K_M",
        help = "GGUF quant to download (e.g., Q4_K_M, Q8_0)"
    )]
    pub quant: String,

    #[arg(long, help = "Path to image file or directory of images")]
    pub input: String,

    #[arg(long, help = "Instruction text or path to instruction file")]
    pub instruction: Option<String>,
}
