/// NPC definitions — types, dialogue, spawn positions.

use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpcKind {
    Garrison = 0,
    Merchant = 1,
    Hermit   = 2,
    Smith    = 3,
    Oracle   = 4,
}

pub struct NpcDef {
    pub kind: NpcKind,
    pub name: &'static str,
    pub dialogue: &'static [&'static str],
    pub world_x: f32,
    pub world_z: f32,
    pub scale: Vec3,
    pub color: Vec3,
}

/// All NPC spawn definitions.
pub fn npc_defs() -> Vec<NpcDef> {
    vec![
        NpcDef {
            kind: NpcKind::Garrison,
            name: "Old Garrison",
            dialogue: &[
                "You look freshly woken. Careful --",
                "slimes nest in the low grass east.",
                "The forest north hides a broken shrine.",
                "Bring me proof of goblins and I'll",
                "make it worth your while.",
            ],
            world_x: 18.0,
            world_z: 22.0,
            scale: Vec3::new(0.9, 1.1, 0.9),
            color: Vec3::new(0.45, 0.55, 0.40),
        },
        NpcDef {
            kind: NpcKind::Merchant,
            name: "Mira",
            dialogue: &[
                "Trade routes? Ha. I go where the",
                "coin goes. Today, that's here.",
                "Desert sand plays havoc with my stock,",
                "but the relics fetch a good price.",
            ],
            world_x: -80.0,
            world_z: -60.0,
            scale: Vec3::new(0.85, 1.0, 0.85),
            color: Vec3::new(0.65, 0.45, 0.30),
        },
        NpcDef {
            kind: NpcKind::Hermit,
            name: "Brother Aldric",
            dialogue: &[
                "The forest keeps its own time.",
                "I stopped counting years ago.",
                "That staff was carved from the oldest",
                "oak I could find. It hums when you cast.",
            ],
            world_x: -220.0,
            world_z: -350.0,
            scale: Vec3::new(0.8, 1.15, 0.8),
            color: Vec3::new(0.30, 0.35, 0.50),
        },
        NpcDef {
            kind: NpcKind::Smith,
            name: "Dura Stonesong",
            dialogue: &[
                "You want quality iron? Earn it.",
                "Golems drop chunks worth smelting.",
                "Bring me stone and wood and I'll",
                "make something that won't bend.",
            ],
            world_x: 380.0,
            world_z: -820.0,
            scale: Vec3::new(0.75, 0.9, 0.75),
            color: Vec3::new(0.70, 0.40, 0.25),
        },
        NpcDef {
            kind: NpcKind::Oracle,
            name: "The Hollow Voice",
            dialogue: &[
                "The rock remembers everything.",
                "Skeletons do not guard treasure.",
                "They guard what was once theirs.",
                "Find the vault where the golem was",
                "first made. Your answer is there.",
            ],
            world_x: 640.0,
            world_z: 900.0,
            scale: Vec3::new(0.7, 1.3, 0.7),
            color: Vec3::new(0.20, 0.15, 0.25),
        },
    ]
}

/// Active dialogue state when the player is talking to an NPC.
pub struct ActiveDialogue {
    pub npc_entity_id: u32,
    pub npc_kind: NpcKind,
    pub npc_name: &'static str,
    pub lines: &'static [&'static str],
    pub current_line: usize,
}

impl ActiveDialogue {
    pub fn current_text(&self) -> &[&str] {
        // Show up to 3 lines starting at current_line.
        let end = (self.current_line + 3).min(self.lines.len());
        &self.lines[self.current_line..end]
    }

    /// Advance to the next page of lines. Returns true if dialogue is finished.
    pub fn advance(&mut self) -> bool {
        self.current_line += 3;
        self.current_line >= self.lines.len()
    }
}
