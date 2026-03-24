/// Save/Load system — serializes player state to binary file.

use serde::{Serialize, Deserialize};

use crate::game::inventory::Inventory;
use crate::game::equipment::EquipmentSlots;
use crate::game::quest::{self, Quest, QuestObjective, QuestState};
use crate::game::stats::StatBlock;

/// Serializable snapshot of game state.
#[derive(Serialize, Deserialize)]
pub struct SaveData {
    pub player_x: f32,
    pub player_y: f32,
    pub player_z: f32,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub stats: StatBlock,
    pub inventory: Inventory,
    pub equipment: EquipmentSlots,
    pub quest_states: Vec<QuestSave>,
    pub time_of_day: f32,
    #[serde(default)]
    pub buildings: Vec<BuildingSave>,
}

/// Serializable building cell.
#[derive(Serialize, Deserialize)]
pub struct BuildingSave {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub block_type: u8,
    pub rotation: u8,
    pub sub_blocks: u64,
    pub color: [f32; 3],
}

/// Minimal quest save — just mutable state per quest ID.
#[derive(Serialize, Deserialize)]
pub struct QuestSave {
    pub id: u8,
    pub state: QuestState,
    pub objective: QuestObjective,
}

/// Save directory path.
fn save_dir() -> std::path::PathBuf {
    let mut p = dirs_fallback();
    p.push("physics-rpg");
    p
}

fn dirs_fallback() -> std::path::PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        let mut p = std::path::PathBuf::from(home);
        p.push(".config");
        p
    } else {
        std::path::PathBuf::from(".")
    }
}

fn save_path() -> std::path::PathBuf {
    let mut p = save_dir();
    p.push("save.bin");
    p
}

pub fn save(data: &SaveData) -> Result<(), String> {
    let dir = save_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {}", e))?;
    let bytes = bincode::serialize(data).map_err(|e| format!("serialize: {}", e))?;
    std::fs::write(save_path(), bytes).map_err(|e| format!("write: {}", e))?;
    Ok(())
}

pub fn load() -> Result<SaveData, String> {
    let bytes = std::fs::read(save_path()).map_err(|e| format!("read: {}", e))?;
    bincode::deserialize(&bytes).map_err(|e| format!("deserialize: {}", e))
}

/// Extract quest save data from full quest list.
pub fn quests_to_save(quests: &[Quest]) -> Vec<QuestSave> {
    quests.iter().map(|q| QuestSave {
        id: q.id,
        state: q.state,
        objective: q.objective.clone(),
    }).collect()
}

/// Apply saved quest state onto a fresh quest list.
pub fn apply_quest_saves(quests: &mut [Quest], saves: &[QuestSave]) {
    for qs in saves {
        if let Some(q) = quests.iter_mut().find(|q| q.id == qs.id) {
            q.state = qs.state;
            q.objective = qs.objective.clone();
        }
    }
}
