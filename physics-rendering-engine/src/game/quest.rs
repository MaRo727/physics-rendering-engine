/// Quest system — kill/collect/reach objectives with XP and item rewards.

use crate::game::items::ItemId;

pub type QuestId = u8;

#[derive(Debug, Clone)]
pub enum QuestObjective {
    /// Kill N enemies of a specific type.  `enemy_kind` matches `EnemyType` discriminant.
    Kill { enemy_kind: u8, needed: u32, done: u32 },
    /// Have N of an item in inventory (not consumed until turn-in).
    Collect { item_id: ItemId, needed: u16 },
    /// Walk within `radius` of a world location.
    Reach { x: f32, z: f32, radius: f32, reached: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestState {
    /// Not yet offered (prerequisite not met).
    Locked,
    /// Available to accept from NPC.
    Available,
    /// In progress.
    Active,
    /// Objective met — talk to NPC to turn in.
    Complete,
    /// Turned in, rewards given.
    TurnedIn,
}

#[derive(Debug, Clone)]
pub struct Quest {
    pub id: QuestId,
    pub name: &'static str,
    pub description: &'static str,
    pub objective: QuestObjective,
    pub state: QuestState,
    pub reward_xp: u32,
    pub reward_gold: u32,
    pub reward_items: &'static [(ItemId, u16)],
    /// Which NPC kind (index) gives/completes this quest.
    pub giver_npc: u8,
    /// Quest ID that must be TurnedIn before this one becomes Available.
    pub prerequisite: Option<QuestId>,
}

// Quest IDs.
pub const QUEST_FIRST_BLOOD: QuestId = 0;
pub const QUEST_HERMIT_HERBS: QuestId = 1;
pub const QUEST_GOBLIN_PATROL: QuestId = 2;
pub const QUEST_STONE_FOR_SMITH: QuestId = 3;

// NPC kind indices (matches NpcKind discriminant order).
pub const NPC_GARRISON: u8 = 0;
pub const NPC_MERCHANT: u8 = 1;
pub const NPC_HERMIT: u8 = 2;
pub const NPC_SMITH: u8 = 3;
pub const NPC_ORACLE: u8 = 4;

// Enemy type discriminants (must match EnemyType enum order).
pub const ENEMY_SLIME: u8 = 0;
pub const ENEMY_SKELETON: u8 = 1;
pub const ENEMY_GOBLIN: u8 = 2;
pub const ENEMY_GOLEM: u8 = 3;

use crate::game::items::{ITEM_HEALTH_POTION, ITEM_OAK_STAFF, ITEM_MANA_POTION,
    ITEM_LEATHER_VEST, ITEM_IRON_SWORD, ITEM_IRON_AXE, ITEM_WARHAMMER,
    ITEM_WOOD, ITEM_STONE};

pub fn create_quests() -> Vec<Quest> {
    vec![
        Quest {
            id: QUEST_FIRST_BLOOD,
            name: "First Blood",
            description: "Kill 5 Slimes near the plains.",
            objective: QuestObjective::Kill { enemy_kind: ENEMY_SLIME, needed: 5, done: 0 },
            state: QuestState::Available,
            reward_xp: 150,
            reward_gold: 10,
            reward_items: &[(ITEM_HEALTH_POTION, 3)],
            giver_npc: NPC_GARRISON,
            prerequisite: None,
        },
        Quest {
            id: QUEST_HERMIT_HERBS,
            name: "The Hermit's Herbs",
            description: "Bring 10 Wood to Brother Aldric.",
            objective: QuestObjective::Collect { item_id: ITEM_WOOD, needed: 10 },
            state: QuestState::Available,
            reward_xp: 200,
            reward_gold: 15,
            reward_items: &[(ITEM_OAK_STAFF, 1), (ITEM_MANA_POTION, 5)],
            giver_npc: NPC_HERMIT,
            prerequisite: None,
        },
        Quest {
            id: QUEST_GOBLIN_PATROL,
            name: "Goblin Patrol",
            description: "Defeat 3 Goblin Archers.",
            objective: QuestObjective::Kill { enemy_kind: ENEMY_GOBLIN, needed: 3, done: 0 },
            state: QuestState::Locked,
            reward_xp: 300,
            reward_gold: 25,
            reward_items: &[(ITEM_LEATHER_VEST, 1), (ITEM_IRON_SWORD, 1)],
            giver_npc: NPC_GARRISON,
            prerequisite: Some(QUEST_FIRST_BLOOD),
        },
        Quest {
            id: QUEST_STONE_FOR_SMITH,
            name: "Stone for the Smith",
            description: "Bring 15 Stone to Dura Stonesong.",
            objective: QuestObjective::Collect { item_id: ITEM_STONE, needed: 15 },
            state: QuestState::Available,
            reward_xp: 400,
            reward_gold: 40,
            reward_items: &[(ITEM_IRON_AXE, 1), (ITEM_WARHAMMER, 1)],
            giver_npc: NPC_SMITH,
            prerequisite: None,
        },
    ]
}

/// Check if a kill quest's counter should increment for this enemy type.
pub fn notify_kill(quests: &mut [Quest], enemy_kind: u8) {
    for q in quests.iter_mut() {
        if q.state != QuestState::Active { continue; }
        if let QuestObjective::Kill { enemy_kind: ek, needed, ref mut done } = q.objective {
            if ek == enemy_kind && *done < needed {
                *done += 1;
                if *done >= needed {
                    q.state = QuestState::Complete;
                }
            }
        }
    }
}

/// Check collect quests against current inventory counts.
pub fn check_collect_quests(quests: &mut [Quest], inventory: &crate::game::inventory::Inventory) {
    for q in quests.iter_mut() {
        if q.state != QuestState::Active { continue; }
        if let QuestObjective::Collect { item_id, needed } = q.objective {
            if inventory.count(item_id) >= needed {
                q.state = QuestState::Complete;
            }
        }
    }
}

/// Check reach quests against player position.
pub fn check_reach_quests(quests: &mut [Quest], px: f32, pz: f32) {
    for q in quests.iter_mut() {
        if q.state != QuestState::Active { continue; }
        if let QuestObjective::Reach { x, z, radius, ref mut reached } = q.objective {
            let dx = px - x;
            let dz = pz - z;
            if dx * dx + dz * dz <= radius * radius {
                *reached = true;
                q.state = QuestState::Complete;
            }
        }
    }
}

/// Unlock quests whose prerequisites have been turned in.
pub fn unlock_quests(quests: &mut [Quest]) {
    // Collect turned-in quest IDs first.
    let turned_in: Vec<QuestId> = quests.iter()
        .filter(|q| q.state == QuestState::TurnedIn)
        .map(|q| q.id)
        .collect();

    for q in quests.iter_mut() {
        if q.state == QuestState::Locked {
            if let Some(pre) = q.prerequisite {
                if turned_in.contains(&pre) {
                    q.state = QuestState::Available;
                }
            }
        }
    }
}
