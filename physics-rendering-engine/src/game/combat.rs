/// Combat state machine: handles melee attack timing, hit detection, and damage.

use glam::Vec3;
use rapier3d::prelude::ColliderHandle;

use crate::game::entity::{Entity, EntityId, EntityKind};
use crate::physics::world::PhysicsWorld;

const BARE_FIST_DAMAGE: f32 = 5.0;
pub const PUNCH_KNOCKBACK: f32 = 3.0;
const PUNCH_RANGE: f32 = 3.0;

const WINDUP_DURATION: f32 = 0.1;
const ACTIVE_DURATION: f32 = 0.15;
const RECOVERY_DURATION: f32 = 0.2;
const TOTAL_DURATION: f32 = WINDUP_DURATION + ACTIVE_DURATION + RECOVERY_DURATION;

// Default attacks per second when bare-fisted.
const BARE_FIST_SPEED: f32 = 2.0;

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
    },
}

pub struct CombatHit {
    pub entity_id: EntityId,
    pub damage: f32,
    pub knockback_dir: Vec3,
}

pub struct CombatSystem {
    state: CombatState,
    attack_cooldown: f32,
}

impl CombatSystem {
    pub fn new() -> Self {
        Self {
            state: CombatState::Idle,
            attack_cooldown: 0.0,
        }
    }

    /// Try to start an attack. Returns true if attack was initiated.
    pub fn try_attack(&mut self) -> bool {
        if self.attack_cooldown > 0.0 {
            return false;
        }
        if matches!(self.state, CombatState::Idle) {
            self.state = CombatState::Attacking {
                phase: AttackPhase::Windup,
                timer: 0.0,
                has_hit: false,
            };
            true
        } else {
            false
        }
    }

    /// Tick the combat state machine. Returns a hit if an enemy was struck this frame.
    pub fn update(
        &mut self,
        dt: f32,
        physics: &PhysicsWorld,
        entities: &[Entity],
        eye: Vec3,
        look_dir: Vec3,
        player_col: ColliderHandle,
    ) -> Option<CombatHit> {
        self.attack_cooldown = (self.attack_cooldown - dt).max(0.0);

        let mut hit_result = None;

        match &mut self.state {
            CombatState::Idle => {}
            CombatState::Attacking { phase, timer, has_hit } => {
                *timer += dt;

                match *phase {
                    AttackPhase::Windup => {
                        if *timer >= WINDUP_DURATION {
                            *timer -= WINDUP_DURATION;
                            *phase = AttackPhase::Active;
                        }
                    }
                    AttackPhase::Active => {
                        // Raycast for enemy hit during active window.
                        if !*has_hit {
                            if let Some((body_handle, hit_pos, _normal)) =
                                physics.cast_ray_full(eye, look_dir, PUNCH_RANGE, player_col)
                            {
                                // Find which entity was hit.
                                if let Some(entity) = entities.iter().find(|e| e.body.rigid_body == body_handle) {
                                    if entity.kind == EntityKind::Enemy {
                                        let damage = BARE_FIST_DAMAGE;
                                        // TODO: use equipped weapon damage + derived stats when available
                                        let knockback_dir = (hit_pos - eye).normalize_or_zero();
                                        hit_result = Some(CombatHit {
                                            entity_id: entity.id,
                                            damage,
                                            knockback_dir,
                                        });
                                        *has_hit = true;
                                    }
                                }
                            }
                        }

                        if *timer >= ACTIVE_DURATION {
                            *timer -= ACTIVE_DURATION;
                            *phase = AttackPhase::Recovery;
                        }
                    }
                    AttackPhase::Recovery => {
                        if *timer >= RECOVERY_DURATION {
                            self.state = CombatState::Idle;
                            self.attack_cooldown = 1.0 / BARE_FIST_SPEED;
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
            CombatState::Attacking { phase, timer, .. } => {
                let elapsed = match phase {
                    AttackPhase::Windup => *timer,
                    AttackPhase::Active => WINDUP_DURATION + *timer,
                    AttackPhase::Recovery => WINDUP_DURATION + ACTIVE_DURATION + *timer,
                };
                (elapsed / TOTAL_DURATION).clamp(0.0, 1.0)
            }
        }
    }

    pub fn is_attacking(&self) -> bool {
        matches!(self.state, CombatState::Attacking { .. })
    }
}
