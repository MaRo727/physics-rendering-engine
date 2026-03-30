use serde::{Deserialize, Serialize};

/// Persisted game settings (graphics, audio, controls).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameSettings {
    /// Internal render resolution scale (0.25 – 1.0).
    pub render_scale: f32,
    /// Terrain draw distance in world units (200 – 2000).
    pub terrain_draw_distance: f32,
    /// Tree draw distance in world units (500 – 5200).
    pub tree_draw_distance: f32,
    /// Grass draw distance in world units (50 – 400).
    pub grass_draw_distance: f32,
    /// Grass density multiplier (0.25 – 1.0).
    pub grass_density: f32,
    /// Vertical field of view in degrees (50 – 120).
    pub fov: f32,
    /// Mouse sensitivity multiplier (0.1 – 3.0).
    pub mouse_sensitivity: f32,
    /// Music volume (0.0 – 1.0).
    pub music_volume: f32,
    /// SFX volume (0.0 – 1.0).
    pub sfx_volume: f32,
}

impl Default for GameSettings {
    fn default() -> Self {
        Self {
            render_scale: 1.0,
            terrain_draw_distance: 1500.0,
            tree_draw_distance: 5200.0,
            grass_draw_distance: 250.0,
            grass_density: 1.0,
            fov: 75.0,
            mouse_sensitivity: 1.0,
            music_volume: 0.5,
            sfx_volume: 0.5,
        }
    }
}

/// A single setting entry for the settings menu.
pub struct SettingEntry {
    pub label: &'static str,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    /// Format the value for display.
    pub format: fn(f32) -> String,
    pub get: fn(&GameSettings) -> f32,
    pub set: fn(&mut GameSettings, f32),
}

fn fmt_percent(v: f32) -> String {
    format!("{}%", (v * 100.0) as i32)
}

fn fmt_distance(v: f32) -> String {
    format!("{}m", v as i32)
}

fn fmt_degrees(v: f32) -> String {
    format!("{}°", v as i32)
}

fn fmt_sensitivity(v: f32) -> String {
    format!("{:.1}x", v)
}

pub const SETTINGS_ENTRIES: &[SettingEntry] = &[
    SettingEntry {
        label: "Render Scale",
        min: 0.25, max: 1.0, step: 0.25,
        format: fmt_percent,
        get: |s| s.render_scale,
        set: |s, v| s.render_scale = v,
    },
    SettingEntry {
        label: "Terrain Distance",
        min: 200.0, max: 2000.0, step: 100.0,
        format: fmt_distance,
        get: |s| s.terrain_draw_distance,
        set: |s, v| s.terrain_draw_distance = v,
    },
    SettingEntry {
        label: "Tree Distance",
        min: 500.0, max: 5200.0, step: 100.0,
        format: fmt_distance,
        get: |s| s.tree_draw_distance,
        set: |s, v| s.tree_draw_distance = v,
    },
    SettingEntry {
        label: "Grass Distance",
        min: 50.0, max: 400.0, step: 25.0,
        format: fmt_distance,
        get: |s| s.grass_draw_distance,
        set: |s, v| s.grass_draw_distance = v,
    },
    SettingEntry {
        label: "Grass Density",
        min: 0.25, max: 1.0, step: 0.25,
        format: fmt_percent,
        get: |s| s.grass_density,
        set: |s, v| s.grass_density = v,
    },
    SettingEntry {
        label: "Field of View",
        min: 50.0, max: 120.0, step: 5.0,
        format: fmt_degrees,
        get: |s| s.fov,
        set: |s, v| s.fov = v,
    },
    SettingEntry {
        label: "Mouse Sensitivity",
        min: 0.1, max: 3.0, step: 0.1,
        format: fmt_sensitivity,
        get: |s| s.mouse_sensitivity,
        set: |s, v| s.mouse_sensitivity = v,
    },
    SettingEntry {
        label: "Music Volume",
        min: 0.0, max: 1.0, step: 0.05,
        format: fmt_percent,
        get: |s| s.music_volume,
        set: |s, v| s.music_volume = v,
    },
    SettingEntry {
        label: "SFX Volume",
        min: 0.0, max: 1.0, step: 0.05,
        format: fmt_percent,
        get: |s| s.sfx_volume,
        set: |s, v| s.sfx_volume = v,
    },
];

fn config_path() -> std::path::PathBuf {
    let base = if let Ok(home) = std::env::var("HOME") {
        let mut p = std::path::PathBuf::from(home);
        p.push(".config");
        p
    } else {
        std::path::PathBuf::from(".")
    };
    base.join("physics-rpg").join("settings.json")
}

impl GameSettings {
    pub fn load() -> Self {
        let path = config_path();
        match std::fs::read_to_string(&path) {
            Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) {
        let path = config_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(data) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(&path, data);
        }
    }

    /// FOV in radians for use in projection matrices.
    pub fn fov_radians(&self) -> f32 {
        self.fov.to_radians()
    }
}
