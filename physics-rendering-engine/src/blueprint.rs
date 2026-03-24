/// Blueprint system — save/load structure blueprints as JSON files.

use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize)]
pub struct Blueprint {
    pub name: String,
    pub blocks: Vec<BlockEntry>,
    /// Baked groups — each group is a merged set of blocks rendered as one object.
    #[serde(default)]
    pub groups: Vec<Vec<BlockEntry>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BlockEntry {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub block_type: u8,
    pub rotation: u8,
    pub sub_blocks: u64,
    pub color: [f32; 3],
}

pub fn blueprints_dir() -> PathBuf {
    let mut p = if let Ok(home) = std::env::var("HOME") {
        let mut p = PathBuf::from(home);
        p.push(".config");
        p
    } else {
        PathBuf::from(".")
    };
    p.push("physics-rpg");
    p.push("blueprints");
    p
}

pub fn save_blueprint(blueprint: &Blueprint) -> Result<PathBuf, String> {
    let dir = blueprints_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {}", e))?;
    let filename = format!("{}.json", blueprint.name);
    let path = dir.join(&filename);
    let json = serde_json::to_string_pretty(blueprint).map_err(|e| format!("serialize: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("write: {}", e))?;
    Ok(path)
}

pub fn load_blueprint(path: &Path) -> Result<Blueprint, String> {
    let data = std::fs::read_to_string(path).map_err(|e| format!("read: {}", e))?;
    serde_json::from_str(&data).map_err(|e| format!("deserialize: {}", e))
}

pub fn list_blueprints() -> Vec<PathBuf> {
    let dir = blueprints_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "json"))
        .collect();
    files.sort();
    files
}
