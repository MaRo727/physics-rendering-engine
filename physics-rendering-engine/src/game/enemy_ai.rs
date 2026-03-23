// Scalable enemy AI: data-driven state machine supporting multiple enemy types.
// To add a new enemy type: add a variant to EnemyType, add a match arm in params().

use std::collections::HashMap;

use glam::Vec3;

use crate::game::entity::{Entity, EntityId, EntityKind};
use crate::physics::body::{ColliderHandle, WeightClass};
use crate::physics::world::PhysicsWorld;
use crate::renderer::{MESH_SLIME, MESH_SKELETON, MESH_GOBLIN, MESH_GOLEM, MESH_ARROW};
use crate::terrain::Biome;

// ---------------------------------------------------------------------------
// Enemy type definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnemyType {
    Slime,
    Skeleton,
    GoblinArcher,
    Golem,
}

/// All per-type parameters. The single source of truth for enemy behavior.
pub struct EnemyParams {
    // Movement
    pub move_speed: f32,
    pub chase_speed: f32,
    // Perception
    pub aggro_range: f32,
    pub deaggro_range: f32,
    pub patrol_radius: f32,
    // Combat
    pub attack_range: f32,
    pub attack_damage: f32,
    pub attack_cooldown: f32,
    pub attack_windup: f32,
    pub knockback_force: f32,
    // Flee
    pub flee_threshold: f32,
    pub flee_speed: f32,
    // Spawn / visual
    pub mesh: u32,
    pub render_scale: Vec3,
    pub bounding_radius: f32,
    pub physics_radius: f32,
    pub weight_class: WeightClass,
    // Base stats
    pub level: u32,
    pub str_: u32,
    pub int: u32,
    pub dex: u32,
    pub vit: u32,
    pub end: u32,
    // Behavior flags
    pub is_ranged: bool,
    pub preferred_range: f32,
    pub hop_movement: bool,
    // Projectile (ranged enemies only)
    pub projectile_speed: f32,
    pub projectile_inaccuracy: f32, // radians of random spread
}

impl EnemyType {
    pub fn params(self) -> EnemyParams {
        match self {
            EnemyType::Slime => EnemyParams {
                move_speed: 4.0, chase_speed: 5.0,
                aggro_range: 15.0, deaggro_range: 20.0, patrol_radius: 6.0,
                attack_range: 1.5, attack_damage: 8.0, attack_cooldown: 1.5,
                attack_windup: 0.0, knockback_force: 3.0,
                flee_threshold: 0.0, flee_speed: 4.0,
                mesh: MESH_SLIME, render_scale: Vec3::new(1.0, 0.7, 1.0),
                bounding_radius: 0.5, physics_radius: 0.5,
                weight_class: WeightClass::Light,
                level: 1, str_: 5, int: 1, dex: 3, vit: 8, end: 3,
                is_ranged: false, preferred_range: 0.0, hop_movement: true,
                projectile_speed: 0.0, projectile_inaccuracy: 0.0,
            },
            EnemyType::Skeleton => EnemyParams {
                move_speed: 3.5, chase_speed: 5.5,
                aggro_range: 18.0, deaggro_range: 25.0, patrol_radius: 8.0,
                attack_range: 2.5, attack_damage: 12.0, attack_cooldown: 1.2,
                attack_windup: 0.3, knockback_force: 4.0,
                flee_threshold: 0.15, flee_speed: 4.0,
                mesh: MESH_SKELETON, render_scale: Vec3::ONE,
                bounding_radius: 0.5, physics_radius: 0.4,
                weight_class: WeightClass::Medium,
                level: 2, str_: 8, int: 2, dex: 6, vit: 10, end: 5,
                is_ranged: false, preferred_range: 0.0, hop_movement: false,
                projectile_speed: 0.0, projectile_inaccuracy: 0.0,
            },
            EnemyType::GoblinArcher => EnemyParams {
                move_speed: 4.0, chase_speed: 4.5,
                aggro_range: 25.0, deaggro_range: 30.0, patrol_radius: 10.0,
                attack_range: 15.0, attack_damage: 10.0, attack_cooldown: 2.0,
                attack_windup: 0.5, knockback_force: 2.0,
                flee_threshold: 0.3, flee_speed: 5.0,
                mesh: MESH_GOBLIN, render_scale: Vec3::splat(0.8),
                bounding_radius: 0.4, physics_radius: 0.35,
                weight_class: WeightClass::Light,
                level: 3, str_: 4, int: 6, dex: 10, vit: 7, end: 4,
                is_ranged: true, preferred_range: 10.0, hop_movement: false,
                projectile_speed: 18.0, projectile_inaccuracy: 0.08,
            },
            EnemyType::Golem => EnemyParams {
                move_speed: 2.0, chase_speed: 3.0,
                aggro_range: 12.0, deaggro_range: 18.0, patrol_radius: 5.0,
                attack_range: 3.0, attack_damage: 25.0, attack_cooldown: 2.5,
                attack_windup: 0.6, knockback_force: 8.0,
                flee_threshold: 0.0, flee_speed: 2.0,
                mesh: MESH_GOLEM, render_scale: Vec3::splat(1.5),
                bounding_radius: 0.8, physics_radius: 0.7,
                weight_class: WeightClass::Heavy,
                level: 5, str_: 15, int: 2, dex: 3, vit: 20, end: 10,
                is_ranged: false, preferred_range: 0.0, hop_movement: false,
                projectile_speed: 0.0, projectile_inaccuracy: 0.0,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Biome-based spawn weights
// ---------------------------------------------------------------------------

// (EnemyType, weight) per biome. Higher weight = more likely to spawn.
// To add a new enemy: add entries here for each biome it can appear in.
fn spawn_table(biome: Biome) -> &'static [(EnemyType, u32)] {
    match biome {
        Biome::Plains => &[
            (EnemyType::Slime, 50),
            (EnemyType::Skeleton, 30),
            (EnemyType::GoblinArcher, 15),
            (EnemyType::Golem, 5),
        ],
        Biome::Forest => &[
            (EnemyType::Slime, 25),
            (EnemyType::Skeleton, 30),
            (EnemyType::GoblinArcher, 35),
            (EnemyType::Golem, 10),
        ],
        Biome::Desert => &[
            (EnemyType::Slime, 10),
            (EnemyType::Skeleton, 50),
            (EnemyType::GoblinArcher, 25),
            (EnemyType::Golem, 15),
        ],
        Biome::Mountains => &[
            (EnemyType::Slime, 10),
            (EnemyType::Skeleton, 15),
            (EnemyType::GoblinArcher, 20),
            (EnemyType::Golem, 55),
        ],
        Biome::Dungeon => &[
            (EnemyType::Skeleton, 40),
            (EnemyType::GoblinArcher, 30),
            (EnemyType::Golem, 30),
        ],
    }
}

/// Pick a random enemy type for a biome using weighted selection.
pub fn pick_enemy_for_biome(biome: Biome, seed: &mut u32) -> EnemyType {
    let table = spawn_table(biome);
    let total: u32 = table.iter().map(|(_, w)| *w).sum();
    let roll = (cheap_rand(seed) * total as f32) as u32;
    let mut acc = 0;
    for &(enemy_type, weight) in table {
        acc += weight;
        if roll < acc {
            return enemy_type;
        }
    }
    table.last().unwrap().0
}

/// Maximum number of enemies alive at once.
pub const MAX_ENEMIES: usize = 30;

/// Minimum distance from the player to spawn an enemy.
pub const MIN_SPAWN_DIST: f32 = 40.0;
/// Maximum distance from the player to spawn an enemy.
pub const MAX_SPAWN_DIST: f32 = 120.0;

/// Whether it's night (enemies spawn). Night: 0.75 (sunset) through 0.0 to 0.25 (sunrise).
pub fn is_night(time_of_day: f32) -> bool {
    time_of_day >= 0.75 || time_of_day < 0.25
}

// ---------------------------------------------------------------------------
// AI state machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum AiState {
    Idle,
    Patrol,
    Chase,
    /// Windup before dealing damage.
    Attack { windup_remaining: f32, has_hit: bool },
    Flee,
}

pub struct EnemyAi {
    pub enemy_type: EnemyType,
    state: AiState,
    timer: f32,
    attack_cooldown: f32,
    seed: u32,
    spawn_pos: Vec3,
    patrol_target: Vec3,
}

impl EnemyAi {
    pub fn new(enemy_type: EnemyType, spawn_pos: Vec3, seed: u32) -> Self {
        Self {
            enemy_type,
            state: AiState::Idle,
            timer: 0.5 + (seed % 100) as f32 * 0.02,
            attack_cooldown: 0.0,
            seed,
            spawn_pos,
            patrol_target: spawn_pos,
        }
    }
}

/// Damage dealt by an enemy to the player this frame.
pub struct EnemyAttackHit {
    pub damage: f32,
    pub knockback_dir: Vec3,
    pub knockback_force: f32,
}

// ---------------------------------------------------------------------------
// Enemy projectiles (arrows, etc.)
// ---------------------------------------------------------------------------

pub struct EnemyProjectile {
    pub position: Vec3,
    pub velocity: Vec3,
    pub lifetime: f32,
    pub damage: f32,
    pub knockback_force: f32,
    pub mesh: u32,
    pub scale: f32,
}

/// Pseudo-random float [0, 1) from a seed, advancing the seed (public for spawn system).
pub fn cheap_rand_pub(seed: &mut u32) -> f32 {
    cheap_rand(seed)
}

/// Pseudo-random float [0, 1) from a seed, advancing the seed.
fn cheap_rand(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((*seed >> 16) & 0x7FFF) as f32 / 32768.0
}

/// Update all enemy AIs. Returns hits that should be applied to the player.
pub fn update_all(
    ais: &mut HashMap<EntityId, EnemyAi>,
    projectiles: &mut Vec<EnemyProjectile>,
    physics: &mut PhysicsWorld,
    entities: &[Entity],
    player_pos: Vec3,
    player_collider: ColliderHandle,
    dt: f32,
) -> Vec<EnemyAttackHit> {
    let mut hits = Vec::new();

    for entity in entities {
        if entity.kind != EntityKind::Enemy {
            continue;
        }
        let ai = match ais.get_mut(&entity.id) {
            Some(a) => a,
            None => continue,
        };

        let params = ai.enemy_type.params();
        let pos = physics.body_position(entity.body.rigid_body);
        let to_player = player_pos - pos;
        let dist = to_player.length();
        let dir_to_player = if dist > 0.1 {
            Vec3::new(to_player.x, 0.0, to_player.z).normalize_or_zero()
        } else {
            Vec3::ZERO
        };

        // Health fraction for flee check.
        let health_frac = entity.stats.as_ref().map_or(1.0, |s| {
            let max = s.compute_derived(&Default::default()).max_health;
            if max > 0.0 { s.health / max } else { 1.0 }
        });

        ai.attack_cooldown = (ai.attack_cooldown - dt).max(0.0);

        match ai.state {
            // ---------------------------------------------------------------
            AiState::Idle => {
                ai.timer -= dt;

                // Priority: flee > chase > patrol
                if params.flee_threshold > 0.0 && health_frac < params.flee_threshold && dist < params.deaggro_range {
                    ai.state = AiState::Flee;
                    ai.timer = 3.0 + cheap_rand(&mut ai.seed) * 2.0;
                } else if dist < params.aggro_range {
                    ai.state = AiState::Chase;
                } else if ai.timer <= 0.0 {
                    // Pick a random patrol target near spawn.
                    let angle = cheap_rand(&mut ai.seed) * std::f32::consts::TAU;
                    let r = cheap_rand(&mut ai.seed) * params.patrol_radius;
                    ai.patrol_target = ai.spawn_pos + Vec3::new(angle.cos() * r, 0.0, angle.sin() * r);
                    ai.state = AiState::Patrol;
                    ai.timer = 3.0 + cheap_rand(&mut ai.seed) * 2.0;
                }
            }

            // ---------------------------------------------------------------
            AiState::Patrol => {
                ai.timer -= dt;
                let to_target = ai.patrol_target - pos;
                let target_dist = Vec3::new(to_target.x, 0.0, to_target.z).length();

                // Check aggro.
                if dist < params.aggro_range {
                    ai.state = AiState::Chase;
                } else if target_dist < 1.0 || ai.timer <= 0.0 {
                    ai.state = AiState::Idle;
                    ai.timer = 1.0 + cheap_rand(&mut ai.seed) * 2.0;
                } else {
                    let dir = Vec3::new(to_target.x, 0.0, to_target.z).normalize_or_zero();
                    apply_movement(physics, entity, dir, params.move_speed, params.hop_movement, &mut ai.seed, dt);
                }
            }

            // ---------------------------------------------------------------
            AiState::Chase => {
                // Check flee.
                if params.flee_threshold > 0.0 && health_frac < params.flee_threshold {
                    ai.state = AiState::Flee;
                    ai.timer = 3.0 + cheap_rand(&mut ai.seed) * 2.0;
                } else if dist > params.deaggro_range {
                    ai.state = AiState::Idle;
                    ai.timer = 1.0 + cheap_rand(&mut ai.seed) * 1.0;
                } else if dist < params.attack_range && ai.attack_cooldown <= 0.0 {
                    ai.state = AiState::Attack { windup_remaining: params.attack_windup, has_hit: false };
                } else {
                    // Move toward player, but ranged enemies maintain preferred_range.
                    let dir = if params.is_ranged && dist < params.preferred_range {
                        -dir_to_player // back away
                    } else {
                        dir_to_player
                    };
                    apply_movement(physics, entity, dir, params.chase_speed, params.hop_movement, &mut ai.seed, dt);
                }
            }

            // ---------------------------------------------------------------
            AiState::Attack { ref mut windup_remaining, ref mut has_hit } => {
                // Check flee interrupt.
                if params.flee_threshold > 0.0 && health_frac < params.flee_threshold {
                    ai.state = AiState::Flee;
                    ai.timer = 3.0 + cheap_rand(&mut ai.seed) * 2.0;
                } else if *windup_remaining > 0.0 {
                    *windup_remaining -= dt;
                } else if !*has_hit {
                    if params.is_ranged {
                        // Spawn a physical projectile toward the player with inaccuracy.
                        let eye = pos + Vec3::Y * 0.5;
                        let base_dir = (player_pos + Vec3::Y * 0.5 - eye).normalize_or_zero();
                        // Apply random spread.
                        let yaw_off = (cheap_rand(&mut ai.seed) - 0.5) * 2.0 * params.projectile_inaccuracy;
                        let pitch_off = (cheap_rand(&mut ai.seed) - 0.5) * 2.0 * params.projectile_inaccuracy;
                        let spread = Vec3::new(
                            base_dir.x * pitch_off.cos() - base_dir.z * yaw_off.sin(),
                            base_dir.y + pitch_off,
                            base_dir.z * pitch_off.cos() + base_dir.x * yaw_off.sin(),
                        );
                        let dir = (base_dir + spread).normalize_or_zero();
                        projectiles.push(EnemyProjectile {
                            position: eye + dir * 0.5,
                            velocity: dir * params.projectile_speed,
                            lifetime: params.attack_range / params.projectile_speed + 0.5,
                            damage: params.attack_damage,
                            knockback_force: params.knockback_force,
                            mesh: MESH_ARROW,
                            scale: 0.5,
                        });
                    } else {
                        // Melee / contact hit check.
                        let did_hit = if params.hop_movement {
                            physics.are_colliders_in_contact(entity.body.collider, player_collider)
                        } else {
                            dist < params.attack_range
                        };
                        if did_hit {
                            hits.push(EnemyAttackHit {
                                damage: params.attack_damage,
                                knockback_dir: dir_to_player,
                                knockback_force: params.knockback_force,
                            });
                        }
                    }
                    *has_hit = true;

                    // Transition back to chase after attack.
                    ai.attack_cooldown = params.attack_cooldown;
                    ai.state = AiState::Chase;
                }
            }

            // ---------------------------------------------------------------
            AiState::Flee => {
                ai.timer -= dt;
                if ai.timer <= 0.0 || dist > params.deaggro_range * 1.5 {
                    ai.state = AiState::Idle;
                    ai.timer = 1.0 + cheap_rand(&mut ai.seed) * 1.0;
                } else {
                    let dir = -dir_to_player;
                    apply_movement(physics, entity, dir, params.flee_speed, params.hop_movement, &mut ai.seed, dt);
                }
            }
        }
    }

    hits
}

/// Tick enemy projectiles: move, check collision with player, expire.
pub fn update_projectiles(
    projectiles: &mut Vec<EnemyProjectile>,
    physics: &PhysicsWorld,
    entities: &[Entity],
    dt: f32,
) -> Vec<EnemyAttackHit> {
    let mut hits = Vec::new();
    let mut i = 0;
    while i < projectiles.len() {
        let proj = &mut projectiles[i];
        proj.lifetime -= dt;
        if proj.lifetime <= 0.0 {
            projectiles.swap_remove(i);
            continue;
        }

        let step = proj.velocity * dt;
        let step_len = step.length();
        let dir = step.normalize_or_zero();

        // Raycast along the step to check for collision.
        if step_len > 0.001 {
            if let Some((body_handle, _hit_pos, _normal)) =
                physics.cast_ray_unfiltered(proj.position, dir, step_len + 0.15)
            {
                // Check if we hit the player.
                let hit_player = entities.iter().any(|e| {
                    e.kind == EntityKind::Player && e.body.rigid_body == body_handle
                });
                if hit_player {
                    hits.push(EnemyAttackHit {
                        damage: proj.damage,
                        knockback_dir: dir,
                        knockback_force: proj.knockback_force,
                    });
                }
                // Arrow hit something (player, terrain, or enemy) — remove it.
                projectiles.swap_remove(i);
                continue;
            }
        }

        proj.position += step;
        // Apply gravity to arrows for a slight arc.
        proj.velocity.y -= 4.0 * dt;
        i += 1;
    }
    hits
}

/// Apply movement to an enemy: hop-based (Slime) or walk-based (others).
fn apply_movement(
    physics: &mut PhysicsWorld,
    entity: &Entity,
    dir: Vec3,
    speed: f32,
    hop: bool,
    seed: &mut u32,
    _dt: f32,
) {
    if hop {
        // Only hop if on ground (avoid double-hops).
        if physics.is_on_ground(entity.body.collider) {
            let impulse = dir * speed + Vec3::Y * 5.0;
            let mass = physics.body_mass(entity.body.rigid_body);
            physics.apply_impulse(entity.body.rigid_body, impulse * mass);
            let _ = cheap_rand(seed); // consume seed for consistency
        }
    } else {
        // Walk: set horizontal velocity, preserve vertical for gravity.
        let current_vel_y = physics.body_linvel_y(entity.body.rigid_body);
        let vel = dir * speed + Vec3::Y * current_vel_y;
        physics.set_body_linvel(entity.body.rigid_body, vel);
    }
}
