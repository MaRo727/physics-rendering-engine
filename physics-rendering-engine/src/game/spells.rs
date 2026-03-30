/// Spell definitions, cooldown tracking, projectile management, and casting.

use std::collections::HashSet;

use glam::Vec3;
use crate::physics::body::ColliderHandle;
use crate::renderer::{MESH_FIREBALL, MESH_ICESHARD};

use crate::game::entity::{EntityId, EntityKind};
use crate::game::stats::{self, DerivedStats};
use crate::game::world::World;
use crate::physics::world::PhysicsWorld;

// ---------------------------------------------------------------------------
// Spell definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpellId {
    Fireball,
    IceShard,
    Heal,
}

impl SpellId {
    pub const ALL: [SpellId; 3] = [SpellId::Fireball, SpellId::IceShard, SpellId::Heal];

    pub fn index(self) -> usize {
        match self {
            SpellId::Fireball => 0,
            SpellId::IceShard => 1,
            SpellId::Heal => 2,
        }
    }

    pub fn next(self) -> Self {
        match self {
            SpellId::Fireball => SpellId::IceShard,
            SpellId::IceShard => SpellId::Heal,
            SpellId::Heal => SpellId::Fireball,
        }
    }

    pub fn def(self) -> &'static SpellDef {
        &SPELLS[self.index()]
    }

    pub fn name(self) -> &'static str {
        self.def().name
    }
}

pub struct SpellDef {
    pub name: &'static str,
    pub mana_cost: f32,
    pub cooldown: f32,
    pub base_damage: f32,
    pub base_heal: f32,
    pub range: f32,
}

static SPELLS: [SpellDef; 3] = [
    // Fireball: high damage projectile
    SpellDef {
        name: "Fireball",
        mana_cost: 20.0,
        cooldown: 2.0,
        base_damage: 25.0,
        base_heal: 0.0,
        range: 30.0,
    },
    // Ice Shard: instant raycast, lower damage, fast cooldown
    SpellDef {
        name: "Ice Shard",
        mana_cost: 12.0,
        cooldown: 1.0,
        base_damage: 15.0,
        base_heal: 0.0,
        range: 40.0,
    },
    // Heal: self-target heal-over-time
    SpellDef {
        name: "Heal",
        mana_cost: 15.0,
        cooldown: 3.0,
        base_damage: 0.0,
        base_heal: 30.0,
        range: 0.0,
    },
];

// ---------------------------------------------------------------------------
// Projectile
// ---------------------------------------------------------------------------

const FIREBALL_SPEED: f32 = 20.0;
const FIREBALL_RADIUS: f32 = 0.5;
const FIREBALL_AOE_RADIUS: f32 = 4.0;
const ICESHARD_SPEED: f32 = 60.0;

pub struct Projectile {
    pub position: Vec3,
    pub velocity: Vec3,
    pub lifetime: f32,
    pub base_damage: f32,
    pub magic_mult: f32,
    pub is_fireball: bool, // true = fireball (AoE + collide), false = visual-only (ice shard trail)
    pub scale: f32,
    pub mesh_type: u32,
}

// ---------------------------------------------------------------------------
// Spell hit result
// ---------------------------------------------------------------------------

pub struct SpellHit {
    pub entity_id: EntityId,
    pub damage: f32,
    pub knockback_dir: Vec3,
}

// ---------------------------------------------------------------------------
// Spell system
// ---------------------------------------------------------------------------

pub struct SpellSystem {
    cooldowns: [f32; 3],
    pub active_spell: SpellId,
    pub projectiles: Vec<Projectile>,
    cycle_prev: bool,
}

impl SpellSystem {
    pub fn new() -> Self {
        Self {
            cooldowns: [0.0; 3],
            active_spell: SpellId::Fireball,
            projectiles: Vec::new(),
            cycle_prev: false,
        }
    }

    /// Cycle to the next spell. Returns true if changed.
    pub fn cycle_spell(&mut self, pressed: bool) -> bool {
        let edge = pressed && !self.cycle_prev;
        self.cycle_prev = pressed;
        if edge {
            self.active_spell = self.active_spell.next();
            println!("Active spell: {}", self.active_spell.name());
            true
        } else {
            false
        }
    }

    /// Try to cast the active spell. Returns the result + mana cost if successful.
    /// Caller is responsible for deducting mana.
    pub fn try_cast(
        &mut self,
        current_mana: f32,
        derived: &DerivedStats,
        eye: Vec3,
        look_dir: Vec3,
        physics: &PhysicsWorld,
        world: &World,
        player_col: ColliderHandle,
    ) -> Option<(CastResult, f32)> {
        let spell = self.active_spell;
        let def = spell.def();
        let idx = spell.index();

        if self.cooldowns[idx] > 0.0 {
            return None;
        }
        if current_mana < def.mana_cost {
            return None;
        }

        let mana_cost = def.mana_cost;
        self.cooldowns[idx] = def.cooldown;

        match spell {
            SpellId::Fireball => {
                let damage = stats::magic_damage(def.base_damage, derived);
                self.projectiles.push(Projectile {
                    position: eye + look_dir * 1.0,
                    velocity: look_dir * FIREBALL_SPEED,
                    lifetime: def.range / FIREBALL_SPEED,
                    base_damage: damage,
                    magic_mult: derived.magic_damage_mult,
                    is_fireball: true,
                    scale: 0.3,
                    mesh_type: MESH_FIREBALL,
                });
                println!("Cast Fireball! (damage: {:.0})", damage);
                Some((CastResult::Projectile, mana_cost))
            }
            SpellId::IceShard => {
                let damage = stats::magic_damage(def.base_damage, derived);
                // Spawn a fast visual projectile so the shard is visible.
                self.projectiles.push(Projectile {
                    position: eye + look_dir * 1.0,
                    velocity: look_dir * ICESHARD_SPEED,
                    lifetime: def.range / ICESHARD_SPEED,
                    base_damage: 0.0, // damage handled by raycast below
                    magic_mult: 1.0,
                    is_fireball: false,
                    scale: 0.15,
                    mesh_type: MESH_ICESHARD,
                });
                // Instant raycast for actual damage.
                if let Some((body_handle, hit_pos, _)) =
                    physics.cast_ray_full(eye, look_dir, def.range, player_col)
                {
                    if let Some(entity) = world.entity_by_rb(body_handle) {
                        if entity.kind == EntityKind::Enemy {
                            let dir = (hit_pos - eye).normalize_or_zero();
                            println!("Ice Shard hits enemy {} for {:.0}!", entity.id, damage);
                            return Some((CastResult::Hit(SpellHit {
                                entity_id: entity.id,
                                damage,
                                knockback_dir: dir,
                            }), mana_cost));
                        }
                    }
                }
                println!("Ice Shard cast (missed)");
                Some((CastResult::Miss, mana_cost))
            }
            SpellId::Heal => {
                let heal = def.base_heal * derived.magic_damage_mult;
                println!("Heal for {:.0} HP!", heal);
                Some((CastResult::Heal(heal), mana_cost))
            }
        }
    }

    /// Tick cooldowns and update projectiles. Fills `hits` with projectile collisions.
    pub fn update(
        &mut self,
        dt: f32,
        physics: &PhysicsWorld,
        world: &World,
        player_col: ColliderHandle,
        hits: &mut Vec<SpellHit>,
    ) {
        // Tick cooldowns.
        for cd in &mut self.cooldowns {
            *cd = (*cd - dt).max(0.0);
        }

        hits.clear();
        let mut i = 0;
        while i < self.projectiles.len() {
            let proj = &mut self.projectiles[i];
            let step = proj.velocity * dt;
            let step_len = step.length();
            let dir = step.normalize_or_zero();

            // Fireball projectiles collide and deal AoE damage.
            // Ice shard visuals just fly through (damage already applied on cast).
            let mut hit_something = false;
            if proj.is_fireball && step_len > 0.001 {
                if let Some((body_handle, hit_pos, _)) =
                    physics.cast_ray_full(proj.position, dir, step_len + FIREBALL_RADIUS, player_col)
                {
                    // Direct hit — check if enemy.
                    if let Some(entity) = world.entity_by_rb(body_handle) {
                        if entity.kind == EntityKind::Enemy {
                            hits.push(SpellHit {
                                entity_id: entity.id,
                                damage: proj.base_damage,
                                knockback_dir: dir,
                            });
                        }
                    }
                    // AoE: damage nearby enemies.
                    // Collect already-hit IDs into a HashSet for O(1) dedup lookups.
                    let already_hit: HashSet<EntityId> = hits.iter().map(|h| h.entity_id).collect();
                    let aoe_sq = FIREBALL_AOE_RADIUS * FIREBALL_AOE_RADIUS;
                    for entity in world.entities.iter().filter(|e| e.kind == EntityKind::Enemy) {
                        if already_hit.contains(&entity.id) {
                            continue;
                        }
                        let epos = physics.body_position(entity.body.rigid_body);
                        let diff = epos - hit_pos;
                        let dist_sq = diff.length_squared();
                        if dist_sq < aoe_sq && dist_sq > 0.01 {
                            let dist = dist_sq.sqrt();
                            let falloff = 1.0 - (dist / FIREBALL_AOE_RADIUS);
                            hits.push(SpellHit {
                                entity_id: entity.id,
                                damage: proj.base_damage * falloff * 0.5,
                                knockback_dir: diff.normalize_or_zero(),
                            });
                        }
                    }
                    hit_something = true;
                }
            }

            proj.position += step;
            proj.lifetime -= dt;

            if hit_something || proj.lifetime <= 0.0 {
                self.projectiles.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn cooldown_remaining(&self, spell: SpellId) -> f32 {
        self.cooldowns[spell.index()]
    }
}

pub enum CastResult {
    Projectile,
    Hit(SpellHit),
    Heal(f32),
    Miss,
}
