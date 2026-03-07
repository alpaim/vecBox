use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;

pub struct DownloadedModel {
    pub gguf_path: PathBuf,
    pub mmproj_path: PathBuf,
    pub config_dir: PathBuf,
}

pub fn download_model(repo_id: &str, quant: &str) -> anyhow::Result<DownloadedModel> {
    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create HF API: {}", e))?;

    let repo = api.model(repo_id.to_string());

    let info = repo
        .info()
        .map_err(|e| anyhow::anyhow!("Failed to get repo info: {}", e))?;

    let mut gguf_file: Option<PathBuf> = None;
    let mut mmproj_file: Option<PathBuf> = None;
    let mut config_files: Vec<PathBuf> = Vec::new();

    for file in info.siblings {
        let filename = file.rfilename;

        let should_download = if filename.ends_with(".gguf") {
            if filename.contains("mmproj") {
                true
            } else if filename.contains(quant) {
                true
            } else {
                false
            }
        } else {
            true
        };

        if should_download {
            let local_path = repo
                .get(&filename)
                .map_err(|e| anyhow::anyhow!("Failed to download {}: {}", filename, e))?;

            if filename.ends_with(".gguf") {
                if filename.contains("mmproj") {
                    mmproj_file = Some(local_path.clone());
                } else if filename.contains(quant) {
                    gguf_file = Some(local_path.clone());
                }
            } else {
                config_files.push(local_path);
            }
        }
    }

    let gguf_path =
        gguf_file.ok_or_else(|| anyhow::anyhow!("No GGUF file found for quant: {}", quant))?;

    let mmproj_path = mmproj_file.ok_or_else(|| anyhow::anyhow!("No mmproj file found"))?;

    let config_dir = if let Some(first_config) = config_files.first() {
        first_config
            .parent()
            .map(|p| p.to_path_buf())
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory"))?
    } else {
        gguf_path
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory"))?
    };

    Ok(DownloadedModel {
        gguf_path,
        mmproj_path,
        config_dir,
    })
}
