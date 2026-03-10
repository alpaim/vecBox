# vecBox

Multimodal embedding inference using Qwen3-VL.

## About

Part of the **vec\*** family - powers [vecDir](https://github.com/alpaim/vecDir) and other privacy-focused local AI projects.

Built on [vecbox-core](https://github.com/alpaim/vecbox-core) - a Rust + Candle inference library.

OpenAI-compatible API for **text and image embeddings**. Since OpenAI doesn't provide multimodal embeddings (yet), here's my improvisation.

## Quick Start

```bash
# Text embedding
cargo run -- text-embedding \
  --repo alpaim/Qwen3-VL-Embedding-2B-GGUF-vecBox \
  --input "your text here"

# Image embedding  
cargo run -- image-embedding \
  --repo alpaim/Qwen3-VL-Embedding-2B-GGUF-vecBox \
  --input /path/to/image.jpg

# REST API server
cargo run -- server \
  --repo alpaim/Qwen3-VL-Embedding-2B-GGUF-vecBox

# Terminal UI
cargo run -- tui
```

## Platforms

| Feature | Platform |
|---------|----------|
| cpu | All  |
| cuda | Linux/Windows  |
| metal | macOS (Apple Silicon) |
| accelerate | macOS (Intel) |
| mkl | Linux/Windows |

Build: `cargo build --features cuda` (or cpu/metal/accelerate/mkl)

## API

```
POST /embed/text   - Text embeddings
POST /embed/image  - Image embeddings
```

Response format mimics OpenAI's embedding API with custom multimodal fields.