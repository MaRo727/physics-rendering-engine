/// Combat state machine: handles melee attack timing, hit detection, and damage.

use glam::Vec3;
use crate::game::entity::{EntityId, EntityKind};
use crate::game::items::{WeaponData, WeaponType};
use crate::game::world::World;
use crate::physics::body::ColliderHandle;
use crate::physics::world::PhysicsWorld;

const BARE_FIST_DAMAGE: f32 = 5.0;
const BARE_FIST_SPEED: f32 = 2.0;
const BARE_FIST_RANGE: f32 = 3.0;
const BARE_FIST_KNOCKBACK: f32 = 3.0;

// Base phase durations (scaled by weapon speed).
const BASE_WINDUP: f32 = 0.1;
const BASE_ACTIVE: f32 = 0.15;
const BASE_RECOVERY: f32 = 0.2;

fn knockback_for_weapon(weapon: Option<&WeaponData>) -> f32 {
    match weapon {
        None => BARE_FIST_KNOCKBACK,
        Some(w) => match w.weapon_type {
            WeaponType::Hammer => 5.0,
            WeaponType::Sword => 3.0,
            WeaponType::Axe => 3.5,
            WeaponType::Staff => 2.0,
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AttackPhase {
    Windup,
    Active,
    Recovery,
}

#[derive(Debug, Clone)]
enum CombatState {
    Idle,
    Attacking {
        phase: AttackPhase,
        timer: f32,
        has_hit: bool,
        // Per-attack parameters from weapon.
        base_damage: f32,
        attack_speed: f32,
        range: f32,
        knockback: f32,
    },
}

pub struct CombatHit {
    pub entity_id: EntityId,
    pub damage: f32,
    pub knockback_dir: Vec3,
    pub knockback_force: f32,
}

pub struct CombatSystem {
    state: CombatState,
    attack_cooldown: f32,
    /// True on the first frame the Active phase is entered (for env hit sync).
    entered_active: bool,
}

impl CombatSystem {
    pub fn new() -> Self {
        Self {
            state: CombatState::Idle,
            attack_cooldown: 0.0,
            entered_active: false,
        }
    }

    /// Try to start an attack. Accepts optional weapon data for damage/speed/range.
    pub fn try_attack(&mut self, weapon: Option<&WeaponData>) -> bool {
        if self.attack_cooldown > 0.0 {
            return false;
        }
        if matches!(self.state, CombatState::Idle) {
            self.state = CombatState::Attacking {
                phase: AttackPhase::Windup,
                timer: 0.0,
                has_hit: false,
                base_damage: weapon.map_or(BARE_FIST_DAMAGE, |w| w.base_damage),
                attack_speed: weapon.map_or(BARE_FIST_SPEED, |w| w.attack_speed),
                range: weapon.map_or(BARE_FIST_RANGE, |w| w.range),
                knockback: knockback_for_weapon(weapon),
            };
            true
        } else {
            false
        }
    }

    /// Tick the combat state machine. Returns a hit if an enemy was struck this frame.
    /// `melee_mult` is the player's melee damage multiplier from derived stats.
    pub fn update(
        &mut self,
        dt: f32,
        physics: &PhysicsWorld,
        world: &World,
        eye: Vec3,
        look_dir: Vec3,
        player_col: ColliderHandle,
        melee_mult: f32,
    ) -> Option<CombatHit> {
        self.attack_cooldown = (self.attack_cooldown - dt).max(0.0);
        self.entered_active = false;

        let mut hit_result = None;

        match &mut self.state {
            CombatState::Idle => {}
            CombatState::Attacking {
                phase, timer, has_hit,
                base_damage, attack_speed, range, knockback,
            } => {
                // Scale phase durations by weapon speed (faster weapon = shorter phases).
                let speed_scale = 1.0 / *attack_speed;
                let windup = BASE_WINDUP * speed_scale;
                let active = BASE_ACTIVE * speed_scale;
                let recovery = BASE_RECOVERY * speed_scale;
                let total = windup + active + recovery;

                *timer += dt;

                match *phase {
                    AttackPhase::Windup => {
                        if *timer >= windup {
                            *timer -= windup;
                            *phase = AttackPhase::Active;
                            self.entered_active = true;
                        }
                    }
                    AttackPhase::Active => {
                        if !*has_hit {
                            if let Some((body_handle, hit_pos, _normal)) =
                                physics.cast_ray_full(eye, look_dir, *range, player_col)
                            {
                                if let Some(entity) = world.entity_by_rb(body_handle) {
                                    if entity.kind == EntityKind::Enemy {
                                        let damage = *base_damage * melee_mult;
                                        let knockback_dir = (hit_pos - eye).normalize_or_zero();
                                        hit_result = Some(CombatHit {
                                            entity_id: entity.id,
                                            damage,
                                            knockback_dir,
                                            knockback_force: *knockback,
                                        });
                                        *has_hit = true;
                                    }
                                }
                            }
                        }

                        if *timer >= active {
                            *timer -= active;
                            *phase = AttackPhase::Recovery;
                        }
                    }
                    AttackPhase::Recovery => {
                        if *timer >= recovery {
                            let cooldown = 1.0 / *attack_speed;
                            self.state = CombatState::Idle;
                            self.attack_cooldown = cooldown;
                            // Store total for animation_progress doesn't need it after idle.
                            let _ = total;
                        }
                    }
                }
            }
        }

        hit_result
    }

    /// Animation progress: 0.0 when idle, 0.0..1.0 during attack cycle.
    pub fn animation_progress(&self) -> f32 {
        match &self.state {
            CombatState::Idle => 0.0,
            CombatState::Attacking { phase, timer, attack_speed, .. } => {
                let speed_scale = 1.0 / *attack_speed;
                let windup = BASE_WINDUP * speed_scale;
                let active = BASE_ACTIVE * speed_scale;
                let recovery = BASE_RECOVERY * speed_scale;
                let total = windup + active + recovery;

                let elapsed = match phase {
                    AttackPhase::Windup => *timer,
                    AttackPhase::Active => windup + *timer,
                    AttackPhase::Recovery => windup + active + *timer,
                };
                (elapsed / total).clamp(0.0, 1.0)
            }
        }
    }

    /// True on the frame the attack enters the Active (hit) phase.
    pub fn entered_active_phase(&self) -> bool {
        self.entered_active
    }

    pub fn is_attacking(&self) -> bool {
        matches!(self.state, CombatState::Attacking { .. })
    }
}
