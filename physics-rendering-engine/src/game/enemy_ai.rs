/// Simple AI for slime enemies: idle and hop toward the player.

use std::collections::HashMap;

use glam::Vec3;

use crate::game::entity::{Entity, EntityId, EntityKind};
use crate::physics::world::PhysicsWorld;

const AGGRO_RANGE: f32 = 15.0;
const HOP_HORIZONTAL: f32 = 4.0;
const HOP_VERTICAL: f32 = 5.0;
const HOP_DURATION: f32 = 0.6;
const IDLE_MIN: f32 = 0.8;
const IDLE_MAX: f32 = 2.0;

#[derive(Debug, Clone, Copy)]
enum SlimeState {
    Idle,
    Hopping,
}

pub struct SlimeAi {
    state: SlimeState,
    timer: f32,
    seed: u32,
}

impl SlimeAi {
    pub fn new(seed: u32) -> Self {
        Self {
            state: SlimeState::Idle,
            timer: 0.5 + (seed % 100) as f32 * 0.02, // stagger initial idle
            seed,
        }
    }
}

/// Simple pseudo-random float [0, 1) from a seed, advancing the seed.
fn cheap_rand(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((*seed >> 16) & 0x7FFF) as f32 / 32768.0
}

/// Update all slime AIs. Call once per frame.
pub fn update_all(
    ais: &mut HashMap<EntityId, SlimeAi>,
    physics: &mut PhysicsWorld,
    entities: &[Entity],
    player_pos: Vec3,
    dt: f32,
) {
    for entity in entities {
        if entity.kind != EntityKind::Enemy {
            continue;
        }
        let ai = match ais.get_mut(&entity.id) {
            Some(a) => a,
            None => continue,
        };

        let pos = physics.body_position(entity.body.rigid_body);
        let to_player = player_pos - pos;
        let dist = to_player.length();

        match ai.state {
            SlimeState::Idle => {
                ai.timer -= dt;
                if ai.timer <= 0.0 {
                    // Hop toward player if in range, otherwise random hop.
                    let dir = if dist < AGGRO_RANGE && dist > 0.1 {
                        Vec3::new(to_player.x, 0.0, to_player.z).normalize_or_zero()
                    } else {
                        let angle = cheap_rand(&mut ai.seed) * std::f32::consts::TAU;
                        Vec3::new(angle.cos(), 0.0, angle.sin())
                    };

                    let impulse = dir * HOP_HORIZONTAL + Vec3::Y * HOP_VERTICAL;
                    let mass = physics.body_mass(entity.body.rigid_body);
                    physics.apply_impulse(entity.body.rigid_body, impulse * mass);

                    ai.state = SlimeState::Hopping;
                    ai.timer = HOP_DURATION;
                }
            }
            SlimeState::Hopping => {
                ai.timer -= dt;
                if ai.timer <= 0.0 {
                    ai.state = SlimeState::Idle;
                    ai.timer = IDLE_MIN + cheap_rand(&mut ai.seed) * (IDLE_MAX - IDLE_MIN);
                }
            }
        }
    }
}
