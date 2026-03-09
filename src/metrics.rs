use serde::Serialize;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum InputType {
    Text,
    Image,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub repo: String,
    pub quant: String,
    pub device: String,
    pub dtype: String,
    pub embedding_dim: usize,
    pub text_model_name: String,
    pub vision_model_name: String,
    pub text_model_size_mb: f64,
    pub vision_model_size_mb: f64,
    pub max_pixels: usize,
    pub min_pixels: usize,
}

impl ModelInfo {
    pub fn new(
        repo: String,
        quant: String,
        device: String,
        dtype: String,
        embedding_dim: usize,
        text_model_path: &Path,
        vision_model_path: &Path,
        max_pixels: usize,
        min_pixels: usize,
    ) -> Self {
        let text_model_name = text_model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let vision_model_name = vision_model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let text_model_size_mb = std::fs::metadata(text_model_path)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0);
        let vision_model_size_mb = std::fs::metadata(vision_model_path)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0);

        Self {
            repo,
            quant,
            device,
            dtype,
            embedding_dim,
            text_model_name,
            vision_model_name,
            text_model_size_mb,
            vision_model_size_mb,
            max_pixels,
            min_pixels,
        }
    }

    pub fn print(&self) {
        println!("\n--- Model Info ---");
        println!("Repo:           {}", self.repo);
        println!("Quantization:   {}", self.quant);
        println!("Device:         {}", self.device);
        println!("Dtype:          {}", self.dtype);
        println!("Embedding dim:  {}", self.embedding_dim);
        println!(
            "Text model:     {} ({:.1} MB)",
            self.text_model_name, self.text_model_size_mb
        );
        println!(
            "Vision model:   {} ({:.1} MB)",
            self.vision_model_name, self.vision_model_size_mb
        );
        println!("Pixels range:   {} - {}", self.min_pixels, self.max_pixels);
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub model_load_time_ms: u64,
    pub total_time_ms: u64,
    pub samples_processed: usize,
    pub samples_per_second: f64,
    pub avg_time_per_sample_ms: f64,
    pub input_type: InputType,
    pub total_tokens: Option<usize>,
    pub tokens_per_second: Option<f64>,
    pub avg_tokens_per_sample: Option<f64>,
    pub total_images: Option<usize>,
    pub avg_time_per_image_ms: Option<f64>,
}

pub struct MetricsBuilder {
    start: Instant,
    model_load_time_ms: u64,
    input_type: InputType,
}

impl MetricsBuilder {
    pub fn new(input_type: InputType) -> Self {
        Self {
            start: Instant::now(),
            model_load_time_ms: 0,
            input_type,
        }
    }

    pub fn with_model_load_time(mut self, ms: u64) -> Self {
        self.model_load_time_ms = ms;
        self
    }

    pub fn finish(&self, samples: usize) -> Metrics {
        let elapsed_ms = self.start.elapsed().as_millis() as u64;
        let samples_per_second = if elapsed_ms > 0 {
            1000.0 * samples as f64 / elapsed_ms as f64
        } else {
            0.0
        };
        let avg_time_per_sample_ms = if samples > 0 {
            elapsed_ms as f64 / samples as f64
        } else {
            0.0
        };

        Metrics {
            model_load_time_ms: self.model_load_time_ms,
            total_time_ms: elapsed_ms,
            samples_processed: samples,
            samples_per_second,
            avg_time_per_sample_ms,
            input_type: self.input_type,
            total_tokens: None,
            tokens_per_second: None,
            avg_tokens_per_sample: None,
            total_images: if self.input_type == InputType::Image {
                Some(samples)
            } else {
                None
            },
            avg_time_per_image_ms: if self.input_type == InputType::Image {
                Some(avg_time_per_sample_ms)
            } else {
                None
            },
        }
    }

    pub fn finish_with_tokens(&self, samples: usize, total_tokens: usize) -> Metrics {
        let elapsed_ms = self.start.elapsed().as_millis() as u64;
        let elapsed_secs = elapsed_ms as f64 / 1000.0;
        let samples_per_second = if elapsed_ms > 0 {
            1000.0 * samples as f64 / elapsed_ms as f64
        } else {
            0.0
        };
        let avg_time_per_sample_ms = if samples > 0 {
            elapsed_ms as f64 / samples as f64
        } else {
            0.0
        };
        let tokens_per_second = if elapsed_secs > 0.0 {
            total_tokens as f64 / elapsed_secs
        } else {
            0.0
        };
        let avg_tokens_per_sample = if samples > 0 {
            total_tokens as f64 / samples as f64
        } else {
            0.0
        };

        Metrics {
            model_load_time_ms: self.model_load_time_ms,
            total_time_ms: elapsed_ms,
            samples_processed: samples,
            samples_per_second,
            avg_time_per_sample_ms,
            input_type: self.input_type,
            total_tokens: Some(total_tokens),
            tokens_per_second: Some(tokens_per_second),
            avg_tokens_per_sample: Some(avg_tokens_per_sample),
            total_images: None,
            avg_time_per_image_ms: None,
        }
    }
}

impl Metrics {
    pub fn format_human(&self) -> String {
        let mut output = String::new();
        output.push_str("\n--- Embedding Metrics ---\n");

        let type_str = match self.input_type {
            InputType::Text => "Text",
            InputType::Image => "Image",
        };
        output.push_str(&format!("Type:           {}\n", type_str));
        output.push_str(&format!("Samples:        {}\n", self.samples_processed));
        output.push_str(&format!("Total time:     {} ms\n", self.total_time_ms));
        output.push_str(&format!(
            "Throughput:     {:.2} samples/sec\n",
            self.samples_per_second
        ));
        output.push_str(&format!(
            "Avg/sample:     {:.2} ms\n",
            self.avg_time_per_sample_ms
        ));

        if let (Some(total_tokens), Some(toks_per_sec), Some(avg_toks)) = (
            self.total_tokens,
            self.tokens_per_second,
            self.avg_tokens_per_sample,
        ) {
            output.push_str(&format!("\nTokens:         {} total\n", total_tokens));
            output.push_str(&format!("Tokens/sec:     {:.2}\n", toks_per_sec));
            output.push_str(&format!("Avg tokens:     {:.2}/sample\n", avg_toks));
        }

        if let (Some(total_images), Some(avg_img_time)) =
            (self.total_images, self.avg_time_per_image_ms)
        {
            output.push_str(&format!("\nImages:         {} total\n", total_images));
            output.push_str(&format!("Avg/image:      {:.2} ms\n", avg_img_time));
        }

        output
    }

    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}
