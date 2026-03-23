/// Core RPG stat block and derived stats.

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatBlock {
    // Primary attributes
    pub strength: u32,
    pub intelligence: u32,
    pub dexterity: u32,
    pub vitality: u32,
    pub endurance: u32,

    // Resource pools (current / max derived from attributes)
    pub health: f32,
    pub mana: f32,
    pub stamina: f32,

    pub level: u32,
    pub xp: u32,
    pub gold: u32,
}

#[derive(Debug, Clone)]
pub struct DerivedStats {
    pub max_health: f32,
    pub max_mana: f32,
    pub max_stamina: f32,
    pub health_regen: f32,  // per second
    pub mana_regen: f32,
    pub stamina_regen: f32,
    pub melee_damage_mult: f32,
    pub magic_damage_mult: f32,
    pub move_speed_mult: f32,
}

/// Additive bonuses from equipment, buffs, etc.
#[derive(Debug, Clone, Default)]
pub struct StatBonuses {
    pub strength: i32,
    pub intelligence: i32,
    pub dexterity: i32,
    pub vitality: i32,
    pub endurance: i32,
    pub flat_armor: f32,
    pub flat_health: f32,
    pub flat_mana: f32,
}

impl StatBlock {
    pub fn new_player() -> Self {
        let mut s = Self {
            strength: 10,
            intelligence: 10,
            dexterity: 10,
            vitality: 10,
            endurance: 10,
            health: 0.0,
            mana: 0.0,
            stamina: 0.0,
            level: 1,
            xp: 0,
            gold: 0,
        };
        let derived = s.compute_derived(&StatBonuses::default());
        s.health = derived.max_health;
        s.mana = derived.max_mana;
        s.stamina = derived.max_stamina;
        s
    }

    pub fn new_enemy(level: u32, str_: u32, int: u32, dex: u32, vit: u32, end: u32) -> Self {
        let mut s = Self {
            strength: str_,
            intelligence: int,
            dexterity: dex,
            vitality: vit,
            endurance: end,
            health: 0.0,
            mana: 0.0,
            stamina: 0.0,
            level,
            xp: 0,
            gold: 0,
        };
        let derived = s.compute_derived(&StatBonuses::default());
        s.health = derived.max_health;
        s.mana = derived.max_mana;
        s.stamina = derived.max_stamina;
        s
    }

    /// Compute derived stats from base attributes + bonuses.
    pub fn compute_derived(&self, bonuses: &StatBonuses) -> DerivedStats {
        let str_total = (self.strength as i32 + bonuses.strength).max(1) as f32;
        let int_total = (self.intelligence as i32 + bonuses.intelligence).max(1) as f32;
        let dex_total = (self.dexterity as i32 + bonuses.dexterity).max(1) as f32;
        let vit_total = (self.vitality as i32 + bonuses.vitality).max(1) as f32;
        let end_total = (self.endurance as i32 + bonuses.endurance).max(1) as f32;

        DerivedStats {
            max_health: 50.0 + vit_total * 10.0 + bonuses.flat_health,
            max_mana: 20.0 + int_total * 5.0 + bonuses.flat_mana,
            max_stamina: 30.0 + end_total * 5.0,
            health_regen: 0.5 + vit_total * 0.1,
            mana_regen: 0.3 + int_total * 0.1,
            stamina_regen: 3.0 + end_total * 0.2,
            melee_damage_mult: 1.0 + str_total / 100.0,
            magic_damage_mult: 1.0 + int_total / 100.0,
            move_speed_mult: 1.0 + dex_total * 0.005,
        }
    }

    /// Tick resource regeneration. Call once per frame with dt.
    pub fn regen_tick(&mut self, derived: &DerivedStats, dt: f32) {
        self.health = (self.health + derived.health_regen * dt).min(derived.max_health);
        self.mana = (self.mana + derived.mana_regen * dt).min(derived.max_mana);
        self.stamina = (self.stamina + derived.stamina_regen * dt).min(derived.max_stamina);
    }

    /// Apply damage. Returns actual damage dealt after clamping.
    pub fn take_damage(&mut self, amount: f32) -> f32 {
        let actual = amount.min(self.health);
        self.health -= actual;
        actual
    }

    pub fn is_dead(&self) -> bool {
        self.health <= 0.0
    }
}

/// Compute melee damage: weapon_base * melee_damage_mult.
pub fn melee_damage(weapon_base: f32, derived: &DerivedStats) -> f32 {
    weapon_base * derived.melee_damage_mult
}

/// Compute magic damage: spell_base * magic_damage_mult.
pub fn magic_damage(spell_base: f32, derived: &DerivedStats) -> f32 {
    spell_base * derived.magic_damage_mult
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_starts_full_health() {
        let s = StatBlock::new_player();
        let d = s.compute_derived(&StatBonuses::default());
        assert!((s.health - d.max_health).abs() < 0.01);
    }

    #[test]
    fn bonuses_increase_stats() {
        let s = StatBlock::new_player();
        let base = s.compute_derived(&StatBonuses::default());
        let boosted = s.compute_derived(&StatBonuses { strength: 10, ..Default::default() });
        assert!(boosted.melee_damage_mult > base.melee_damage_mult);
    }

    #[test]
    fn damage_reduces_health() {
        let mut s = StatBlock::new_player();
        let initial = s.health;
        s.take_damage(20.0);
        assert!((initial - s.health - 20.0).abs() < 0.01);
    }

    #[test]
    fn regen_restores_health() {
        let mut s = StatBlock::new_player();
        let d = s.compute_derived(&StatBonuses::default());
        s.take_damage(30.0);
        let after_dmg = s.health;
        s.regen_tick(&d, 1.0);
        assert!(s.health > after_dmg);
    }
}
