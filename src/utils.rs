use std::fs;
use std::path::Path;

pub fn resolve_input(s: &str) -> String {
    let path = Path::new(s);
    if path.exists() && path.is_file() {
        fs::read_to_string(path).unwrap_or_else(|_| s.to_string())
    } else {
        s.to_string()
    }
}

pub fn resolve_path(s: &str) -> String {
    let path = Path::new(s);
    if path.exists() {
        s.to_string()
    } else {
        panic!("File not found: {}", s);
    }
}

pub fn is_directory(s: &str) -> bool {
    Path::new(s).is_dir()
}

pub fn get_files_from_directory(dir: &str) -> Vec<String> {
    let path = Path::new(dir);
    if !path.is_dir() {
        return vec![dir.to_string()];
    }

    let mut files: Vec<String> = fs::read_dir(path)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .map(|entry| entry.path().to_string_lossy().to_string())
        .collect();

    files.sort();
    files
}
