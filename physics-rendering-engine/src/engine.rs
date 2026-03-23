use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use glam::{Mat4, Vec3, Vec4};
use winit::window::Window;

use crate::audio::AudioManager;
use crate::building::{self, BuildingGrid};
use crate::game::camera::ThirdPersonCamera;
use crate::game::combat::CombatSystem;
use crate::game::enemy_ai::{self, EnemyAi, EnemyProjectile, EnemyType};
use crate::game::entity::{Entity, EntityKind};
use crate::game::items::ITEM_IRON_SWORD;
use crate::game::player_model::{PlayerModel, BODY_PART_COUNT};
use crate::game::progression;
use crate::game::spells::{SpellSystem, CastResult};
use crate::game::stats::{StatBlock, StatBonuses};
use crate::game::world::World;
use crate::input::InputState;
use crate::interaction::{Interaction, ToolType, PickaxeHit, HammerHit};
use crate::mining::MiningSystem;
use crate::physics::body::{PhysicsBody, WeightClass};
use crate::physics::world::PhysicsWorld;
use crate::player::{GhostCamera, extract_frustum_planes, is_sphere_in_frustum};
use crate::renderer::{Renderer, pack_instance_id, MESH_CUBE, MESH_WATER, MESH_TERRAIN_BASE};
use crate::scene::{self, UNIT_BOUNDING_RADIUS};
use crate::structures::{StructureGrid, GrassGrid};
use crate::terrain::{TerrainGrid, TerrainChunkInfo, TERRAIN_HALF, CHUNKS_PER_SIDE};

const PLACE_RANGE: f32 = 8.0;
const BUILDING_OBJECT_ID: u32 = 0xFFF0;
const PLAYER_MODEL_OBJECT_ID: u32 = 0xFFE0;
const WATER_OBJECT_ID: u32 = 0xFFD0;
const PLAYER_SPEED: f32 = 5.0;
const FAST_SPEED: f32 = 40.0;
const JUMP_VELOCITY: f32 = 6.0;
const WATER_LEVEL: f32 = 5.0;
const BUOYANCY_FORCE: f32 = 25.0;
const WATER_DRAG: f32 = 12.0;

/// Minimum interval between terrain GPU/physics rebuilds (seconds).
const TERRAIN_REBUILD_INTERVAL: f32 = 0.1;
/// Radius of terrain deformation per pickaxe hit (world units).
const DEFORM_RADIUS: f32 = 3.0;
/// Height lowered at the center of a pickaxe hit.
const DEFORM_AMOUNT: f32 = 0.5;

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct EngineConfig {
    pub window_width: u32,
    pub window_height: u32,
    pub window_title: String,
    pub gravity: Vec3,
}

pub struct Engine {
    pub config: EngineConfig,
    physics: PhysicsWorld,
    world: World,
    player_rb: rapier3d::prelude::RigidBodyHandle,
    player_col: rapier3d::prelude::ColliderHandle,
    camera: ThirdPersonCamera,
    player_model: PlayerModel,
    renderer: Renderer,
    interaction: Interaction,
    ghost: GhostCamera,
    terrain: TerrainGrid,
    terrain_chunks: Vec<TerrainChunkInfo>,
    terrain_object_id: u32,
    terrain_rbs: std::collections::HashSet<rapier3d::prelude::RigidBodyHandle>,
    terrain_chunk_cols: Vec<rapier3d::prelude::ColliderHandle>,
    terrain_rebuild_timer: f32,
    mesh_building_id: u32,
    building: BuildingGrid,
    mining: MiningSystem,
    structures: StructureGrid,
    grass: GrassGrid,
    place_prev: bool,
    spawn_prev: bool,
    debug_stats_prev: bool,
    fast_prev: bool,
    fast_mode: bool,
    show_debug_ui: bool,
    debug_ui_prev: bool,
    mute_prev: bool,
    fast_time: bool,
    fast_time_prev: bool,
    water_time: f32,
    time_of_day: f32, // 0..1 where 0=midnight, 0.25=sunrise, 0.5=noon, 0.75=sunset
    surface_width: u32,
    surface_height: u32,
    audio: Option<AudioManager>,
    combat: CombatSystem,
    spells: SpellSystem,
    cast_prev: bool,
    enemy_ais: HashMap<u32, EnemyAi>,
    enemy_projectiles: Vec<EnemyProjectile>,
    player_visual_yaw: f32,
    snow_intensity: f32,
    snow_time: f32,
}

impl Engine {
    pub fn new(config: EngineConfig, window: &Arc<Window>) -> Result<Self> {
        let surface_width = config.window_width;
        let surface_height = config.window_height;

        let mut physics = PhysicsWorld::new(config.gravity);
        let (entities, player_id, next_id) = scene::build_scene(&mut physics);

        // Generate terrain chunks.
        let terrain = TerrainGrid::generate(42);
        let (chunk_meshes, terrain_chunks, _full_mesh) =
            terrain.generate_chunks(MESH_TERRAIN_BASE);
        let num_terrain_chunks = chunk_meshes.len();

        // Add per-chunk heightfield colliders for terrain.
        let chunk_world_size = (TERRAIN_HALF * 2) as f32 / CHUNKS_PER_SIDE as f32;
        let chunk_scale = Vec3::new(chunk_world_size, 1.0, chunk_world_size);
        let mut terrain_rbs = std::collections::HashSet::new();
        let mut terrain_chunk_cols = Vec::with_capacity(terrain.chunk_count());
        for i in 0..terrain.chunk_count() {
            let (heights, nrows, ncols, cx, cz) = terrain.chunk_heightfield_data(i);
            let (rb, col) = physics.add_heightfield_chunk(&heights, nrows, ncols, chunk_scale, cx, cz);
            terrain_rbs.insert(rb);
            terrain_chunk_cols.push(col);
        }

        // Build game world.
        let world = World::new(entities, player_id, next_id);

        // Extract player physics handles.
        let player_rb = world.player().body.rigid_body;
        let player_col = world.player().body.collider;

        // Spawn player on terrain surface.
        let spawn_h = terrain.get_height(0, 4) + 1.0 + 0.9;
        physics.set_body_position(player_rb, Vec3::new(0.0, spawn_h, 4.0));

        let camera = ThirdPersonCamera::new();
        let player_model = PlayerModel::new();

        let terrain_object_id = 0xFFF1;

        // Generate structures (trees, ruins) and grass on terrain.
        let structures = StructureGrid::generate(42, &terrain);
        let grass = GrassGrid::generate(42, &terrain);

        // Add tree trunk colliders (compound collider per chunk).
        for (_, trunks) in structures.trunk_colliders() {
            if !trunks.is_empty() {
                physics.add_compound_static(&trunks);
            }
        }

        // Extra headroom: base entities + terrain + building + player model parts + trees + grass + dynamic.
        let max_instances = (world.entities.len() + num_terrain_chunks + 3 + BODY_PART_COUNT + 32768 + 16384 + 512) as u32;
        let renderer = Renderer::new(window, max_instances, chunk_meshes)?;
        let mesh_building_id = renderer.mesh_building_id();

        // Spawn initial mining nodes on terrain.
        let mining = MiningSystem::new();
        let mut engine = Self {
            config,
            physics,
            world,
            player_rb,
            player_col,
            camera,
            player_model,
            renderer,
            interaction: Interaction::default(),
            ghost: GhostCamera::default(),
            terrain,
            terrain_chunks,
            terrain_object_id,
            terrain_rbs,
            terrain_chunk_cols,
            terrain_rebuild_timer: 0.0,
            mesh_building_id,
            building: BuildingGrid::new(),
            mining,
            structures,
            grass,
            place_prev: false,
            spawn_prev: false,
            debug_stats_prev: false,
            fast_prev: false,
            fast_mode: false,
            show_debug_ui: false,
            debug_ui_prev: false,
            mute_prev: false,
            fast_time: false,
            fast_time_prev: false,
            water_time: 0.0,
            time_of_day: 0.35, // start at mid-morning
            surface_width,
            surface_height,
            audio: AudioManager::new(std::path::Path::new("../assets")),
            combat: CombatSystem::new(),
            spells: SpellSystem::new(),
            cast_prev: false,
            enemy_ais: HashMap::new(),
            enemy_projectiles: Vec::new(),
            player_visual_yaw: 0.0,
            snow_intensity: 0.0,
            snow_time: 0.0,
        };

        // Spawn a few mining nodes scattered on the terrain.
        engine.spawn_mining_nodes();

        // Spawn enemies on the terrain.
        engine.spawn_enemies();

        // Equip player with starting weapon.
        {
            let player = engine.world.player_mut();
            if let Some(ref mut eq) = player.equipment {
                let _ = eq.equip(ITEM_IRON_SWORD);
            }
        }

        Ok(engine)
    }

    fn spawn_mining_nodes(&mut self) {
        let positions = [
            Vec3::new(15.0, 0.0, 10.0),
            Vec3::new(-20.0, 0.0, 25.0),
            Vec3::new(30.0, 0.0, -15.0),
            Vec3::new(-10.0, 0.0, -30.0),
            Vec3::new(40.0, 0.0, 5.0),
        ];
        let sizes = [
            (3, 2, 3),
            (4, 3, 4),
            (2, 2, 2),
            (3, 3, 3),
            (4, 2, 3),
        ];
        for (&pos, &size) in positions.iter().zip(sizes.iter()) {
            let next_id = &mut self.world.next_entity_id;
            let new_entities = self.mining.spawn_node(
                &mut self.physics,
                &self.terrain,
                pos,
                size,
                &mut || {
                    let id = *next_id;
                    *next_id += 1;
                    id
                },
            );
            self.world.entities.extend(new_entities);
        }
    }

    fn spawn_enemies(&mut self) {
        let spawns: &[(Vec3, EnemyType)] = &[
            // Slimes
            (Vec3::new(12.0, 0.0, 8.0), EnemyType::Slime),
            (Vec3::new(-10.0, 0.0, 15.0), EnemyType::Slime),
            (Vec3::new(20.0, 0.0, -10.0), EnemyType::Slime),
            (Vec3::new(-18.0, 0.0, -12.0), EnemyType::Slime),
            (Vec3::new(8.0, 0.0, 25.0), EnemyType::Slime),
            (Vec3::new(-5.0, 0.0, -20.0), EnemyType::Slime),
            // Skeletons
            (Vec3::new(30.0, 0.0, 5.0), EnemyType::Skeleton),
            (Vec3::new(-25.0, 0.0, 20.0), EnemyType::Skeleton),
            // Goblin Archers
            (Vec3::new(15.0, 0.0, -25.0), EnemyType::GoblinArcher),
            (Vec3::new(-30.0, 0.0, -5.0), EnemyType::GoblinArcher),
            // Golem
            (Vec3::new(0.0, 0.0, 35.0), EnemyType::Golem),
        ];

        for (i, &(pos, enemy_type)) in spawns.iter().enumerate() {
            let params = enemy_type.params();
            let terrain_y = self.terrain.height_at_world(pos.x, pos.z);
            let spawn_pos = Vec3::new(pos.x, terrain_y + 1.0, pos.z);

            let id = self.world.alloc_id();
            let body = PhysicsBody::new_enemy_ball(
                &mut self.physics,
                spawn_pos,
                params.physics_radius,
                params.weight_class,
            );
            let stats = StatBlock::new_enemy(
                params.level, params.str_, params.int,
                params.dex, params.vit, params.end,
            );
            let entity = Entity::enemy(
                id, body, params.mesh, params.render_scale,
                params.bounding_radius, stats,
            );
            self.world.entities.push(entity);
            self.enemy_ais.insert(id, EnemyAi::new(enemy_type, spawn_pos, i as u32 * 7 + 13));
        }
    }

    pub fn update(&mut self, dt: f32, input: &InputState) {
        self.water_time += dt;
        // Day/night cycle: ~4 minutes per full day, F4 speeds up 10x.
        if input.fast_time && !self.fast_time_prev {
            self.fast_time = !self.fast_time;
        }
        self.fast_time_prev = input.fast_time;
        const DAY_DURATION: f32 = 240.0;
        let time_speed = if self.fast_time { 10.0 } else { 1.0 };
        self.time_of_day = (self.time_of_day + dt * time_speed / DAY_DURATION) % 1.0;
        self.terrain_rebuild_timer += dt;

        // --- Fast mode toggle (F2) ---
        if input.toggle_fast && !self.fast_prev {
            self.fast_mode = !self.fast_mode;
        }
        self.fast_prev = input.toggle_fast;

        // --- Debug UI toggle (F3) ---
        if input.toggle_debug_ui && !self.debug_ui_prev {
            self.show_debug_ui = !self.show_debug_ui;
        }
        self.debug_ui_prev = input.toggle_debug_ui;

        // --- Mute toggle (M) ---
        if input.toggle_mute && !self.mute_prev {
            if let Some(audio) = &mut self.audio {
                audio.toggle_mute();
            }
        }
        self.mute_prev = input.toggle_mute;

        // --- Debug stats (F1) ---
        let debug_edge = input.debug_stats && !self.debug_stats_prev;
        self.debug_stats_prev = input.debug_stats;
        if debug_edge {
            self.world.log_player_stats();
        }

        // --- Ghost mode toggle ---
        let just_entered_ghost = self.ghost.handle_toggle(input);

        if just_entered_ghost {
            let aspect = self.surface_width as f32 / self.surface_height.max(1) as f32;
            let (view, proj) = self.camera.camera_matrices(aspect);
            self.ghost.activate_from_camera(&self.camera, view, proj);
            self.interaction.drop_held(&mut self.physics);
        }

        // --- Tool cycling (Tab) ---
        self.interaction.cycle_tool(input.cycle_tool);

        if self.ghost.active {
            self.ghost.update(dt, input);
            self.physics.set_body_linvel(self.player_rb, Vec3::ZERO);
            self.apply_buoyancy(dt);
            self.physics.step(dt);
        } else {
            // --- Third-person camera ---
            self.camera.look(input);
            let player_pos = self.physics.body_position(self.player_rb);
            self.camera.update(player_pos, &self.physics, self.player_col);

            let cam_eye = self.camera.eye;
            let look_dir = self.camera.look_dir();
            let player_eye = player_pos + Vec3::new(0.0, 1.5, 0.0);

            // Determine if crosshair is aimed at a building cell (for pry logic).
            let building_cell_aimed_at = self.physics
                .cast_ray_detailed(cam_eye, look_dir, PLACE_RANGE, self.player_col)
                .and_then(|(hit_pos, normal)| {
                    let target = hit_pos - normal * 0.01;
                    let coords = building::snap_to_grid(target);
                    if self.building.is_occupied(coords.0, coords.1, coords.2) {
                        Some(coords)
                    } else {
                        None
                    }
                });

            let interaction_result = self.interaction.update(
                &mut self.physics,
                &self.world.entities,
                player_eye,
                look_dir,
                input.interact,
                input.throw,
                self.player_col,
                dt,
                building_cell_aimed_at,
                &self.terrain_rbs,
            );

            // --- Handle item pickup ---
            if let Some(drop_id) = interaction_result.picked_up_item {
                if self.world.pickup_item(drop_id) {
                    if let Some(idx) = self.world.entities.iter().position(|e| e.id == drop_id) {
                        let entity = self.world.entities.swap_remove(idx);
                        self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                    }
                }
            }

            // --- Handle pried building cube ---
            if let Some((cx, cy, cz)) = interaction_result.pried_cell {
                self.building.remove(&mut self.physics, cx, cy, cz);

                let center = building::cell_center(cx, cy, cz);
                let obj_id = self.world.alloc_id();

                let body = PhysicsBody::new_dynamic_box(
                    &mut self.physics,
                    center,
                    Vec3::splat(0.5),
                    WeightClass::Medium,
                );
                self.physics.set_gravity_enabled(body.rigid_body, false);
                self.interaction.held_body = Some(body.rigid_body);

                self.world.entities.push(Entity::prop(
                    obj_id,
                    body,
                    MESH_CUBE,
                    Vec3::ONE,
                    UNIT_BOUNDING_RADIUS,
                ));
            }

            // --- Handle axe split ---
            if let Some(target_body) = interaction_result.axe_hit {
                self.split_cube(target_body, player_eye, look_dir);
            }

            // --- Handle pickaxe hit (terrain only) ---
            if let Some(pickaxe_hit) = interaction_result.pickaxe_hit {
                match pickaxe_hit {
                    PickaxeHit::Terrain(hit_pos) => {
                        self.terrain.deform_ground(hit_pos, DEFORM_RADIUS, DEFORM_AMOUNT);
                    }
                }
            }

            // --- Handle hammer hit (structures only) ---
            if let Some(hammer_hit) = interaction_result.hammer_hit {
                match hammer_hit {
                    HammerHit::Body(rb) => {
                        self.damage_mining_chunk(rb);
                    }
                    HammerHit::Static(rb, hit_pos) => {
                        if self.building.has_body(rb) {
                            self.building.mine_at(&mut self.physics, hit_pos);
                        } else if self.mining.is_mining_chunk(rb) {
                            self.damage_mining_chunk(rb);
                        }
                    }
                }
            }

            // --- RMB: place held cube into building grid ---
            let place_pressed = input.place && !self.place_prev;
            self.place_prev = input.place;

            if place_pressed {
                if let Some(held_handle) = self.interaction.held_body.take() {
                    let pos = self.physics.body_position(held_handle);
                    let (cx, cy, cz) = building::snap_to_grid(pos);

                    if self.building.place(&mut self.physics, cx, cy, cz) {
                        if let Some(idx) = self.world.entities.iter().position(|o| o.body.rigid_body == held_handle) {
                            let entity = self.world.entities.swap_remove(idx);
                            self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                        }
                    } else {
                        self.interaction.held_body = Some(held_handle);
                    }
                }
            }

            // --- F: spawn a new block ---
            let spawn_pressed = input.spawn && !self.spawn_prev;
            self.spawn_prev = input.spawn;

            if spawn_pressed && self.interaction.held_body.is_none() {
                let spawn_pos = player_pos + Vec3::new(0.0, 1.5, 0.0) + look_dir * 3.0;
                let obj_id = self.world.alloc_id();

                let body = PhysicsBody::new_dynamic_box(
                    &mut self.physics,
                    spawn_pos,
                    Vec3::splat(0.5),
                    WeightClass::Medium,
                );
                self.physics.set_gravity_enabled(body.rigid_body, false);
                self.interaction.held_body = Some(body.rigid_body);

                self.world.entities.push(Entity::prop(
                    obj_id,
                    body,
                    MESH_CUBE,
                    Vec3::ONE,
                    UNIT_BOUNDING_RADIUS,
                ));
            }

            // --- Compute player derived stats (used by combat + spells) ---
            let player_melee_mult = {
                let p = self.world.player();
                let bonuses = p.equipment.as_ref()
                    .map(|eq| eq.total_bonuses())
                    .unwrap_or_default();
                p.stats.as_ref()
                    .map(|s| s.compute_derived(&bonuses).melee_damage_mult)
                    .unwrap_or(1.0)
            };

            // --- Combat: melee attack ---
            if input.throw
                && self.interaction.held_body.is_none()
                && self.interaction.equipped_tool == ToolType::Hands
            {
                let weapon_data = self.world.player().equipment.as_ref()
                    .and_then(|eq| eq.weapon_data());
                self.combat.try_attack(weapon_data);
            }
            if let Some(hit) = self.combat.update(
                dt,
                &self.physics,
                &self.world.entities,
                player_eye,
                look_dir,
                self.player_col,
                player_melee_mult,
            ) {
                // Apply damage to hit enemy.
                if let Some(entity) = self.world.entities.iter_mut().find(|e| e.id == hit.entity_id) {
                    if let Some(ref mut stats) = entity.stats {
                        let dealt = stats.take_damage(hit.damage);
                        println!("Hit enemy {} for {:.0} damage! HP: {:.0}", hit.entity_id, dealt, stats.health);
                    }
                    self.physics.apply_impulse(
                        entity.body.rigid_body,
                        hit.knockback_dir * hit.knockback_force * self.physics.body_mass(entity.body.rigid_body),
                    );
                }
            }

            // --- Spell cycling (Q) ---
            self.spells.cycle_spell(input.cycle_spell);

            // --- Spell casting (R) ---
            let cast_edge = input.cast_spell && !self.cast_prev;
            self.cast_prev = input.cast_spell;
            if cast_edge {
                // Read player stats immutably first.
                let (current_mana, derived) = {
                    let player = self.world.player();
                    let bonuses = player.equipment.as_ref()
                        .map(|eq| eq.total_bonuses())
                        .unwrap_or_default();
                    let mana = player.stats.as_ref().map_or(0.0, |s| s.mana);
                    let derived = player.stats.as_ref()
                        .map(|s| s.compute_derived(&bonuses))
                        .unwrap_or_else(|| StatBlock::new_player().compute_derived(&StatBonuses::default()));
                    (mana, derived)
                };

                if let Some((result, mana_cost)) = self.spells.try_cast(
                    current_mana,
                    &derived,
                    player_eye,
                    look_dir,
                    &self.physics,
                    &self.world.entities,
                    self.player_col,
                ) {
                    // Deduct mana.
                    if let Some(ref mut stats) = self.world.player_mut().stats {
                        stats.mana -= mana_cost;
                    }

                    match result {
                        CastResult::Hit(spell_hit) => {
                            // Ice Shard direct hit.
                            if let Some(entity) = self.world.entities.iter_mut().find(|e| e.id == spell_hit.entity_id) {
                                if let Some(ref mut stats) = entity.stats {
                                    stats.take_damage(spell_hit.damage);
                                }
                                self.physics.apply_impulse(
                                    entity.body.rigid_body,
                                    spell_hit.knockback_dir * 4.0 * self.physics.body_mass(entity.body.rigid_body),
                                );
                            }
                        }
                        CastResult::Heal(amount) => {
                            if let Some(ref mut stats) = self.world.player_mut().stats {
                                stats.health = (stats.health + amount).min(derived.max_health);
                            }
                        }
                        CastResult::Projectile | CastResult::Miss => {}
                    }
                }
            }

            // --- Update spell projectiles ---
            let spell_hits = self.spells.update(
                dt,
                &self.physics,
                &self.world.entities,
                self.player_col,
            );
            for spell_hit in spell_hits {
                if let Some(entity) = self.world.entities.iter_mut().find(|e| e.id == spell_hit.entity_id) {
                    if let Some(ref mut stats) = entity.stats {
                        let dealt = stats.take_damage(spell_hit.damage);
                        println!("Fireball hits enemy {} for {:.0}! HP: {:.0}", spell_hit.entity_id, dealt, stats.health);
                    }
                    self.physics.apply_impulse(
                        entity.body.rigid_body,
                        spell_hit.knockback_dir * 4.0 * self.physics.body_mass(entity.body.rigid_body),
                    );
                }
            }

            // --- Remove dead enemies + award XP ---
            let dead_ids: Vec<(u32, u32)> = self.world.entities.iter()
                .filter(|e| e.kind == EntityKind::Enemy)
                .filter(|e| e.stats.as_ref().map_or(false, |s| s.is_dead()))
                .map(|e| {
                    let level = e.stats.as_ref().map_or(1, |s| s.level);
                    (e.id, level)
                })
                .collect();
            for &(dead_id, enemy_level) in &dead_ids {
                if let Some(idx) = self.world.entities.iter().position(|e| e.id == dead_id) {
                    let entity = self.world.entities.swap_remove(idx);
                    self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                    self.enemy_ais.remove(&dead_id);

                    // Award XP to player.
                    let xp = progression::xp_for_kill(enemy_level);
                    if let Some(ref mut stats) = self.world.player_mut().stats {
                        let levels = progression::award_xp(stats, xp);
                        println!("Enemy defeated! +{} XP", xp);
                        if levels > 0 {
                            println!("Level up! Now level {}", stats.level);
                        }
                    }
                }
            }

            // --- Enemy AI ---
            let player_col_handle = self.world.player().body.collider;
            let enemy_hits = enemy_ai::update_all(
                &mut self.enemy_ais,
                &mut self.enemy_projectiles,
                &mut self.physics,
                &self.world.entities,
                player_pos,
                player_col_handle,
                dt,
            );
            // Tick enemy projectiles (arrows).
            let arrow_hits = enemy_ai::update_projectiles(
                &mut self.enemy_projectiles,
                &self.physics,
                &self.world.entities,
                dt,
            );
            // Apply enemy damage to player (melee + projectile).
            for hit in enemy_hits.into_iter().chain(arrow_hits) {
                if let Some(ref mut stats) = self.world.player_mut().stats {
                    let dealt = stats.take_damage(hit.damage);
                    println!("Enemy hit you for {:.0} damage! HP: {:.0}", dealt, stats.health);
                }
                let player_rb = self.world.player().body.rigid_body;
                let mass = self.physics.body_mass(player_rb);
                self.physics.apply_impulse(player_rb, hit.knockback_dir * hit.knockback_force * mass);
            }

            // --- Camera-relative player movement ---
            self.apply_player_movement(input);

            // --- Water buoyancy ---
            self.apply_buoyancy(dt);

            self.physics.step(dt);

            // --- Update player model animation ---
            self.player_model.set_attack_progress(self.combat.animation_progress());
            let vel = self.physics.body_linvel_xz(self.player_rb);
            let horiz_speed = Vec3::new(vel.x, 0.0, vel.z).length();
            self.player_model.update(dt, horiz_speed);

            // --- Smooth player facing yaw ---
            let target_yaw = self.player_facing_yaw();
            let mut delta = target_yaw - self.player_visual_yaw;
            // Wrap to [-PI, PI] for shortest-path rotation.
            if delta > std::f32::consts::PI { delta -= std::f32::consts::TAU; }
            if delta < -std::f32::consts::PI { delta += std::f32::consts::TAU; }
            let turn_speed = 12.0; // radians per second
            let max_step = turn_speed * dt;
            self.player_visual_yaw += delta.clamp(-max_step, max_step);
        }

        // --- Blizzard intensity + snow particles ---
        self.snow_time += dt;
        {
            let player_pos = self.physics.body_position(self.player_rb);
            let in_snow = self.terrain.is_snow_zone(player_pos.x, player_pos.z);

            // Smooth snow intensity transition.
            let target = if in_snow { 1.0 } else { 0.0 };
            let ramp_speed = if in_snow { 0.6 } else { 1.5 }; // fade in slower, fade out faster
            if self.snow_intensity < target {
                self.snow_intensity = (self.snow_intensity + ramp_speed * dt).min(target);
            } else {
                self.snow_intensity = (self.snow_intensity - ramp_speed * dt).max(target);
            }

        }

        // --- Batched terrain rebuild (mesh + physics) ---
        if self.terrain.has_dirty_chunks()
            && self.terrain_rebuild_timer >= TERRAIN_REBUILD_INTERVAL
        {
            self.terrain_rebuild_timer = 0.0;
            self.rebuild_dirty_terrain();
        }

        // --- Game tick: regen, etc. ---
        self.world.game_tick(dt);

        // --- Audio: update music and footsteps based on player biome ---
        if let Some(audio) = &mut self.audio {
            let player_pos = self.physics.body_position(self.player_rb);
            let biome = self.terrain.biome_at_world(player_pos.x, player_pos.z);
            audio.update(dt, biome, None);

            let vel = self.physics.body_linvel_xz(self.player_rb);
            let horizontal_speed = (vel.x * vel.x + vel.z * vel.z).sqrt();
            let on_ground = self.physics.is_on_ground(self.player_col);
            audio.update_footsteps(dt, biome, horizontal_speed, on_ground);
        }
    }

    /// Rebuild terrain chunk meshes, BLASes, and physics heightfield for dirty chunks.
    fn rebuild_dirty_terrain(&mut self) {
        let dirty = self.terrain.take_dirty_chunks();
        if dirty.is_empty() {
            return;
        }

        // Regenerate chunk meshes.
        let updates: Vec<(usize, Vec<crate::renderer::mesh::Vertex>, Vec<u32>)> = dirty
            .iter()
            .map(|&idx| {
                let (verts, indices) = self.terrain.regenerate_chunk(idx);
                (idx, verts, indices)
            })
            .collect();

        // Update renderer (GPU mesh + BLASes).
        if let Err(e) = self.renderer.update_terrain_chunks(&updates) {
            log::error!("Failed to update terrain chunks: {}", e);
        }

        // Update only the dirty chunks' physics heightfields.
        let chunk_world_size = (TERRAIN_HALF * 2) as f32 / CHUNKS_PER_SIDE as f32;
        let chunk_scale = Vec3::new(chunk_world_size, 1.0, chunk_world_size);
        for &idx in &dirty {
            let (heights, nrows, ncols, _cx, _cz) = self.terrain.chunk_heightfield_data(idx);
            self.physics.update_heightfield_chunk(
                self.terrain_chunk_cols[idx],
                &heights,
                nrows,
                ncols,
                chunk_scale,
            );
        }
    }

    /// Apply camera-relative movement to the player rigid body.
    fn apply_player_movement(&mut self, input: &InputState) {
        let forward = self.camera.forward_flat();
        let right = self.camera.right_flat();

        let mut move_vel = Vec3::ZERO;
        if input.forward  { move_vel += forward; }
        if input.backward { move_vel -= forward; }
        if input.right    { move_vel += right; }
        if input.left     { move_vel -= right; }
        let speed = if self.fast_mode { FAST_SPEED } else { PLAYER_SPEED };
        if move_vel.length_squared() > 0.0 {
            move_vel = move_vel.normalize() * speed;
        }

        // Slide along walls: remove the velocity component pushing into each wall.
        for wall_n in self.physics.wall_normals(self.player_col) {
            let into_wall = move_vel.dot(wall_n);
            if into_wall < 0.0 {
                move_vel -= wall_n * into_wall;
            }
        }

        let mut vy = self.physics.body_linvel_y(self.player_rb);
        if input.jump && self.physics.is_on_ground(self.player_col) {
            vy = JUMP_VELOCITY;
        }
        self.physics.set_body_linvel(
            self.player_rb,
            Vec3::new(move_vel.x, vy, move_vel.z),
        );
    }

    /// Apply buoyancy and drag to a single rigid body using impulses (not forces,
    /// which accumulate across frames in Rapier 0.22).
    fn apply_body_buoyancy(&mut self, rb: rapier3d::prelude::RigidBodyHandle, size: f32, dt: f32) {
        let pos = self.physics.body_position(rb);
        let depth = WATER_LEVEL - pos.y;
        if depth > 0.0 {
            let mass = self.physics.body_mass(rb);
            let submerged = (depth / size).min(1.0);

            // Impulse = mass * acceleration * dt.
            let buoyancy = Vec3::new(0.0, mass * BUOYANCY_FORCE * submerged * dt, 0.0);
            self.physics.apply_impulse(rb, buoyancy);

            // Drag on all axes.
            let vel = self.physics.body_linvel_xz(rb);
            let drag = vel * (-mass * WATER_DRAG * submerged * dt);
            self.physics.apply_impulse(rb, drag);
        }
    }

    /// Apply buoyancy and drag to all dynamic bodies submerged below the water level.
    fn apply_buoyancy(&mut self, dt: f32) {
        self.apply_body_buoyancy(self.player_rb, 1.8, dt);

        let bodies: Vec<_> = self.world.entities.iter()
            .filter(|e| e.kind != EntityKind::Player && self.physics.is_dynamic(e.body.rigid_body))
            .map(|e| (e.body.rigid_body, e.render_scale.max_element()))
            .collect();

        for (rb, size) in bodies {
            self.apply_body_buoyancy(rb, size, dt);
        }
    }

    /// Damage a mining chunk and handle destruction + stability collapse.
    fn damage_mining_chunk(&mut self, rb: rapier3d::prelude::RigidBodyHandle) {
        if let Some(destroyed_id) = self.mining.damage_chunk(rb, 1) {
            if let Some(idx) = self.world.entities.iter().position(|e| e.id == destroyed_id) {
                let entity = self.world.entities.swap_remove(idx);
                self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
            }
            let impact_pos = self.physics.body_position(rb);
            let collapsed = self.mining.check_stability(&self.physics, impact_pos);
            for eid in collapsed {
                if let Some(idx) = self.world.entities.iter().position(|e| e.id == eid) {
                    let entity = self.world.entities.swap_remove(idx);
                    self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                }
            }
        }
    }

    /// Split a cube object into two halves along the axis most aligned with the hit.
    fn split_cube(&mut self, target_body: rapier3d::prelude::RigidBodyHandle, _eye: Vec3, look_dir: Vec3) {
        let obj_idx = match self.world.entities.iter().position(|o| o.body.rigid_body == target_body) {
            Some(idx) => idx,
            None => return,
        };

        if self.world.entities[obj_idx].mesh_type != MESH_CUBE {
            let wc = self.world.entities[obj_idx].body.weight_class;
            let force = look_dir * 8.0 * wc.punch_knockback();
            self.physics.apply_impulse(target_body, force);
            return;
        }

        let entity = self.world.entities.swap_remove(obj_idx);
        let pos = self.physics.body_position(entity.body.rigid_body);
        let transform = self.physics.body_transform(entity.body.rigid_body);
        let scale = entity.render_scale;

        let inv_rot = transform.inverse();
        let local_dir = inv_rot.transform_vector3(look_dir);
        let abs = local_dir.abs();

        let split_axis = if abs.x >= abs.y && abs.x >= abs.z {
            0
        } else if abs.y >= abs.x && abs.y >= abs.z {
            1
        } else {
            2
        };

        self.physics.remove_body(entity.body.rigid_body, entity.body.collider);

        let mut half_scale = scale;
        match split_axis {
            0 => half_scale.x *= 0.5,
            1 => half_scale.y *= 0.5,
            _ => half_scale.z *= 0.5,
        }

        let mut offset = Vec3::ZERO;
        match split_axis {
            0 => offset.x = scale.x * 0.25,
            1 => offset.y = scale.y * 0.25,
            _ => offset.z = scale.z * 0.25,
        }

        let world_offset = transform.transform_vector3(offset);
        let half_extents = half_scale * 0.5;
        let wc = entity.body.weight_class;

        for sign in [-1.0_f32, 1.0_f32] {
            let half_pos = pos + world_offset * sign;
            let obj_id = self.world.alloc_id();

            let body = PhysicsBody::new_dynamic_box(
                &mut self.physics,
                half_pos,
                half_extents,
                wc,
            );

            let separation_impulse = world_offset.normalize_or_zero() * sign * 2.0;
            self.physics.apply_impulse(body.rigid_body, separation_impulse);

            self.world.entities.push(Entity::prop(
                obj_id,
                body,
                MESH_CUBE,
                half_scale,
                half_scale.max_element() * UNIT_BOUNDING_RADIUS,
            ));
        }
    }

    pub fn render(&mut self) -> Result<()> {
        // Rebuild building mesh on GPU if grid changed.
        if self.building.is_dirty() {
            let (verts, indices) = self.building.generate_mesh();
            self.renderer.update_building_mesh(&verts, &indices)?;
            self.building.clear_dirty();
        }

        let aspect = self.surface_width as f32 / self.surface_height.max(1) as f32;

        let (cull_view, cull_proj) = if self.ghost.active {
            (self.ghost.frozen_view, self.ghost.frozen_proj)
        } else {
            self.camera.camera_matrices(aspect)
        };

        let (render_view, render_proj) = if self.ghost.active {
            self.ghost.camera_matrices(aspect)
        } else {
            (cull_view, cull_proj)
        };

        let mut transforms = Vec::new();
        let mut instance_ids = Vec::new();

        // In ghost mode, frustum-cull to the frozen camera so only visible
        // geometry appears.  In normal mode, skip culling so off-screen
        // entities can still cast shadows.
        let ghost_frustum = if self.ghost.active {
            Some(extract_frustum_planes(cull_proj * cull_view))
        } else {
            None
        };

        // World entities (skip the player entity — we render the model instead).
        for entity in &self.world.entities {
            if entity.kind == EntityKind::Player { continue; }

            let pos = self.physics.body_position(entity.body.rigid_body);
            if let Some(ref planes) = ghost_frustum {
                if !is_sphere_in_frustum(planes, pos, entity.bounding_radius) {
                    continue;
                }
            }

            let t = self.physics.body_transform(entity.body.rigid_body)
                * Mat4::from_scale(entity.render_scale);
            transforms.push(t);
            instance_ids.push(pack_instance_id(entity.mesh_type, entity.id));
        }

        // Player model body parts.
        let player_pos = self.physics.body_position(self.player_rb);
        let player_yaw = self.player_visual_yaw;
        let parts = self.player_model.compute_transforms(player_pos, player_yaw);
        let part_meshes = PlayerModel::mesh_types();
        for (i, (transform, _scale)) in parts.iter().enumerate() {
            transforms.push(*transform);
            instance_ids.push(pack_instance_id(part_meshes[i], PLAYER_MODEL_OBJECT_ID + i as u32));
        }

        // Spell projectiles.
        let projectile_object_base: u32 = 0xFFA0;
        for (i, proj) in self.spells.projectiles.iter().enumerate() {
            let t = Mat4::from_translation(proj.position) * Mat4::from_scale(Vec3::splat(proj.scale));
            transforms.push(t);
            instance_ids.push(pack_instance_id(proj.mesh_type, projectile_object_base + i as u32));
        }

        // Enemy projectiles (arrows).
        let enemy_proj_object_base: u32 = 0xFF90;
        for (i, proj) in self.enemy_projectiles.iter().enumerate() {
            // Orient arrow along its velocity.
            let dir = proj.velocity.normalize_or_zero();
            let rot = if dir.length_squared() > 0.001 {
                let up = Vec3::Y;
                let right = up.cross(dir).normalize_or_zero();
                let corrected_up = dir.cross(right);
                Mat4::from_cols(
                    Vec4::new(right.x, right.y, right.z, 0.0),
                    Vec4::new(corrected_up.x, corrected_up.y, corrected_up.z, 0.0),
                    Vec4::new(dir.x, dir.y, dir.z, 0.0),
                    Vec4::new(0.0, 0.0, 0.0, 1.0),
                )
            } else {
                Mat4::IDENTITY
            };
            let t = Mat4::from_translation(proj.position) * rot * Mat4::from_scale(Vec3::splat(proj.scale));
            transforms.push(t);
            instance_ids.push(pack_instance_id(proj.mesh, enemy_proj_object_base + i as u32));
        }

        // Terrain chunks — cull in ghost mode, include all otherwise for shadows.
        for chunk in &self.terrain_chunks {
            if let Some(ref planes) = ghost_frustum {
                if !is_sphere_in_frustum(planes, chunk.center, chunk.radius) {
                    continue;
                }
            }
            transforms.push(Mat4::IDENTITY);
            instance_ids.push(pack_instance_id(chunk.mesh_type, self.terrain_object_id));
        }

        // Trees near the player, frustum-culled to the player camera
        // (in ghost mode, use the frozen player frustum like terrain chunks).
        let tree_frustum = extract_frustum_planes(cull_proj * cull_view);
        self.structures.render_nearby(player_pos, &tree_frustum, &mut transforms, &mut instance_ids);

        // Grass patches near the player, frustum-culled.
        self.grass.render_nearby(player_pos, &tree_frustum, &mut transforms, &mut instance_ids);

        // Water plane at WATER_LEVEL, drifting slowly for animation.
        // Wave period ~52.36 (2*PI/0.12), so wrap offset to stay seamless.
        let wave_period = std::f32::consts::TAU / 0.12;
        let water_offset = (self.water_time * 2.0) % wave_period;
        transforms.push(Mat4::from_translation(Vec3::new(water_offset, WATER_LEVEL, water_offset * 0.6)));
        instance_ids.push(pack_instance_id(MESH_WATER, WATER_OBJECT_ID));

        // Building mesh.
        if !self.building.is_empty() && self.renderer.has_building_blas() {
            transforms.push(Mat4::IDENTITY);
            instance_ids.push(pack_instance_id(self.mesh_building_id, BUILDING_OBJECT_ID));
        }

        let player_vp = cull_proj * cull_view;

        let pry_progress = self.interaction.pry_progress();
        let tool_type = match self.interaction.equipped_tool {
            crate::interaction::ToolType::Hands => 0.0,
            crate::interaction::ToolType::Axe => 1.0,
            crate::interaction::ToolType::Pickaxe => 2.0,
            crate::interaction::ToolType::Hammer => 3.0,
        };

        // Debug overlay data.
        let biome_id = match self.terrain.biome_at_world(player_pos.x, player_pos.z) {
            crate::terrain::Biome::Plains => 0.0,
            crate::terrain::Biome::Forest => 1.0,
            crate::terrain::Biome::Desert => 2.0,
            crate::terrain::Biome::Mountains => 3.0,
            crate::terrain::Biome::Dungeon => 4.0,
        };
        let (hp_frac, mana_frac, stam_frac, level) = if let Some(stats) = &self.world.player().stats {
            let derived = stats.compute_derived(&crate::game::stats::StatBonuses::default());
            (
                stats.health / derived.max_health,
                stats.mana / derived.max_mana,
                stats.stamina / derived.max_stamina,
                stats.level as f32,
            )
        } else {
            (1.0, 1.0, 1.0, 1.0)
        };
        let debug_info = Vec4::new(
            if self.show_debug_ui { 1.0 } else { 0.0 },
            biome_id,
            hp_frac,
            mana_frac,
        );
        let debug_info2 = Vec4::new(
            level,
            stam_frac,
            player_pos.x,
            player_pos.z,
        );

        // Compute sun/moon positions from time of day.
        // -cos gives: midnight(0)=-1, sunrise(0.25)=0, noon(0.5)=+1, sunset(0.75)=0
        let phase = self.time_of_day * std::f32::consts::TAU;
        let sun_altitude = -(phase.cos()); // -1 at midnight, +1 at noon
        // Sun east-west sweep: sin gives +1 at sunrise, 0 at noon, -1 at sunset
        let sun_x = phase.sin();
        // Sun direction in world space — allow going below horizon so it visually sets.
        let sun_dir = Vec3::new(sun_x, sun_altitude, 0.3).normalize();

        // Moon is opposite the sun.
        let moon_altitude = -sun_altitude; // +1 at midnight, -1 at noon
        let moon_dir = Vec3::new(-sun_x, moon_altitude, -0.3).normalize();

        // Light: blend between sun and moon based on sun altitude.
        // Sun fades out as it dips below horizon, moon fades in.
        // Fade over a wide range: 0 at altitude -0.25, full at 0.5 — slow ramp up as they rise.
        let sun_fade = ((sun_altitude + 0.25) / 0.75).clamp(0.0, 1.0);
        let moon_fade = ((moon_altitude + 0.25) / 0.75).clamp(0.0, 1.0);

        let sun_height = sun_altitude.max(0.0);
        let sunset_factor = (1.0 - sun_height * 2.0).max(0.0);
        let r = 1.0;
        let g = 0.95 - sunset_factor * 0.35;
        let b = 0.9 - sunset_factor * 0.6;
        let sun_intensity = (0.3 + sun_height.min(1.0) * 0.7) * sun_fade;
        let sun_color = Vec4::new(r, g, b, sun_intensity);

        let moon_height = moon_altitude.max(0.0);
        let moon_intensity = (0.08 + moon_height * 0.12) * moon_fade;
        let moon_color = Vec4::new(0.6, 0.7, 1.0, moon_intensity);

        // Blend light direction and color smoothly between sun and moon.
        let total = sun_intensity + moon_intensity;
        let (light_dir, light_color) = if total < 0.001 {
            // Both below horizon — use a dim upward fill light.
            (Vec3::new(0.0, 1.0, 0.0), Vec4::new(0.5, 0.5, 0.7, 0.05))
        } else {
            let sun_weight = sun_intensity / total;
            let blended_dir = (sun_dir * sun_weight + moon_dir * (1.0 - sun_weight)).normalize();
            let blended_color = Vec4::new(
                sun_color.x * sun_weight + moon_color.x * (1.0 - sun_weight),
                sun_color.y * sun_weight + moon_color.y * (1.0 - sun_weight),
                sun_color.z * sun_weight + moon_color.z * (1.0 - sun_weight),
                total,
            );
            (blended_dir, blended_color)
        };

        // Pack sun and moon directions for shader disc rendering.
        // sunMoon.xyz = sun direction, sunMoon.w = sun altitude for sky color.
        let sun_moon = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, sun_altitude);
        // Repurpose ghostMode.w (was unused 0.0) for moon direction packing won't work cleanly.
        // Instead pass moon dir via lightDir.w (was 0.0) — but lightDir is already used.
        // Simplest: add a second vec4 for moon.
        let moon_info = Vec4::new(moon_dir.x, moon_dir.y, moon_dir.z, moon_altitude);

        let blizzard_info = Vec4::new(self.snow_intensity, self.snow_time, 0.0, 0.0);

        self.renderer.draw_frame(
            &transforms,
            &instance_ids,
            render_view,
            render_proj,
            Vec4::from((light_dir, 0.0)),
            light_color,
            player_vp,
            self.ghost.active,
            pry_progress,
            tool_type,
            debug_info,
            debug_info2,
            sun_moon,
            moon_info,
            blizzard_info,
        )
    }

    /// Compute the direction the player character should face.
    /// Uses movement direction if moving, otherwise faces camera forward.
    fn player_facing_yaw(&self) -> f32 {
        let vel = self.physics.body_linvel_xz(self.player_rb);
        let horiz = Vec3::new(vel.x, 0.0, vel.z);
        if horiz.length_squared() > 0.5 {
            horiz.x.atan2(horiz.z)
        } else {
            // Keep current facing when stopped.
            self.player_visual_yaw
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_width = width;
        self.surface_height = height;
        self.renderer.resize(width, height);
    }
}
