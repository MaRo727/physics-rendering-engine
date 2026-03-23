/// Game world: owns all entities and game state.

use crate::game::entity::{Entity, EntityId, EntityKind};
use crate::game::stats::StatBonuses;
use crate::game::items::item_by_id;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameState {
    Playing,
    InventoryOpen,
    Paused,
}

pub struct World {
    pub entities: Vec<Entity>,
    pub player_id: EntityId,
    pub state: GameState,
    pub game_time: f32,
    pub next_entity_id: EntityId,
}

impl World {
    pub fn new(entities: Vec<Entity>, player_id: EntityId, next_id: EntityId) -> Self {
        Self {
            entities,
            player_id,
            state: GameState::Playing,
            game_time: 0.0,
            next_entity_id: next_id,
        }
    }

    pub fn alloc_id(&mut self) -> EntityId {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        id
    }

    pub fn next_entity_id(&self) -> EntityId {
        self.next_entity_id
    }

    /// Find entity by id.
    pub fn entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }

    pub fn entity_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.entities.iter_mut().find(|e| e.id == id)
    }

    /// Get the player entity.
    pub fn player(&self) -> &Entity {
        self.entity(self.player_id).expect("Player entity missing")
    }

    pub fn player_mut(&mut self) -> &mut Entity {
        let id = self.player_id;
        self.entity_mut(id).expect("Player entity missing")
    }

    /// Tick game logic: regen, etc.
    pub fn game_tick(&mut self, dt: f32) {
        self.game_time += dt;

        // Regen for all entities with stats (using cached derived to avoid recomputation)
        for entity in &mut self.entities {
            if let (Some(stats), Some(derived)) = (&mut entity.stats, &entity.cached_derived) {
                stats.regen_tick(derived, dt);
            }
        }
    }

    /// Try to pick up a nearby item drop for the player.
    pub fn pickup_item(&mut self, drop_entity_id: EntityId) -> bool {
        // Find the drop
        let drop_idx = match self.entities.iter().position(|e| e.id == drop_entity_id && e.kind == EntityKind::ItemDrop) {
            Some(idx) => idx,
            None => return false,
        };

        let (item_id, count) = match self.entities[drop_idx].drop_item {
            Some(d) => d,
            None => return false,
        };

        // Find the player's inventory
        let player_idx = self.entities.iter().position(|e| e.id == self.player_id).unwrap();

        // Check if player has inventory space
        let leftover = {
            let inv = self.entities[player_idx].inventory.as_mut().unwrap();
            inv.add(item_id, count)
        };

        if leftover == 0 {
            // Fully picked up — remove the drop entity
            let name = item_by_id(item_id).map_or("???", |d| d.name);
            println!("Picked up {}x {}", count, name);
            // We'll return true and let engine handle physics removal
            true
        } else {
            // Partially picked up — update remaining count
            self.entities[drop_idx].drop_item = Some((item_id, leftover));
            false
        }
    }

    /// Log player stats to console (debug).
    pub fn log_player_stats(&self) {
        let player = self.player();
        if let Some(ref stats) = player.stats {
            let bonuses = player.equipment.as_ref()
                .map(|eq| eq.total_bonuses())
                .unwrap_or(StatBonuses::default());
            let derived = stats.compute_derived(&bonuses);
            println!("=== Player Stats (Lv.{}) ===", stats.level);
            println!("  HP: {:.0}/{:.0}  MP: {:.0}/{:.0}  SP: {:.0}/{:.0}",
                stats.health, derived.max_health,
                stats.mana, derived.max_mana,
                stats.stamina, derived.max_stamina);
            println!("  STR:{} INT:{} DEX:{} VIT:{} END:{}",
                stats.strength, stats.intelligence, stats.dexterity,
                stats.vitality, stats.endurance);
            println!("  XP: {} / {}", stats.xp, crate::game::progression::xp_to_next(stats.level));
            println!("  Melee mult: {:.2}  Magic mult: {:.2}",
                derived.melee_damage_mult, derived.magic_damage_mult);

            if let Some(ref eq) = player.equipment {
                if let Some(w) = eq.weapon_data() {
                    println!("  Weapon: dmg={:.0} spd={:.1}", w.base_damage, w.attack_speed);
                }
                println!("  Armor: {:.0}", eq.total_armor());
            }
        }
        if let Some(ref inv) = player.inventory {
            let occupied = inv.slots().iter().filter(|s| s.is_some()).count();
            println!("  Inventory: {}/{} slots used", occupied, crate::game::inventory::INVENTORY_SIZE);
            for (i, slot) in inv.slots().iter().enumerate() {
                if let Some(stack) = slot {
                    let name = item_by_id(stack.item_id).map_or("???", |d| d.name);
                    println!("    [{}] {}x {}", i, stack.count, name);
                }
            }
        }
    }
}
