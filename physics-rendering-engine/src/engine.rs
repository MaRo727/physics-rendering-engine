use std::sync::Arc;

use anyhow::Result;
use glam::{Mat4, Vec3, Vec4};
use winit::window::Window;

use crate::building::{self, BuildingGrid};
use crate::game::camera::ThirdPersonCamera;
use crate::game::entity::{Entity, EntityKind};
use crate::game::player_model::{PlayerModel, BODY_PART_COUNT};
use crate::game::world::World;
use crate::input::InputState;
use crate::interaction::{Interaction, PickaxeHit, HammerHit};
use crate::mining::MiningSystem;
use crate::physics::body::{PhysicsBody, WeightClass};
use crate::physics::world::PhysicsWorld;
use crate::player::{GhostCamera, extract_frustum_planes, is_sphere_in_frustum};
use crate::renderer::{Renderer, pack_instance_id, MESH_CUBE, MESH_CAPSULE, MESH_WATER, MESH_TERRAIN_BASE};
use crate::scene::{self, UNIT_BOUNDING_RADIUS};
use crate::structures::StructureGrid;
use crate::terrain::{TerrainGrid, TerrainChunkInfo, TERRAIN_HALF};

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
    terrain_rb: rapier3d::prelude::RigidBodyHandle,
    terrain_col: rapier3d::prelude::ColliderHandle,
    terrain_rebuild_timer: f32,
    mesh_building_id: u32,
    building: BuildingGrid,
    mining: MiningSystem,
    structures: StructureGrid,
    place_prev: bool,
    spawn_prev: bool,
    debug_stats_prev: bool,
    fast_prev: bool,
    fast_mode: bool,
    show_debug_ui: bool,
    debug_ui_prev: bool,
    water_time: f32,
    surface_width: u32,
    surface_height: u32,
    light_dir: Vec3,
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

        // Add heightfield collider for terrain.
        let (heights, nrows, ncols) = terrain.heightfield_data();
        let scale = Vec3::new(
            (TERRAIN_HALF * 2) as f32,
            1.0,
            (TERRAIN_HALF * 2) as f32,
        );
        let (terrain_rb, terrain_col) = physics.add_heightfield(heights, nrows, ncols, scale);

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

        // Generate structures (trees, ruins) on terrain.
        let structures = StructureGrid::generate(42, &terrain);

        // Add tree trunk colliders (compound collider per chunk).
        for (_, trunks) in structures.trunk_colliders() {
            if !trunks.is_empty() {
                physics.add_compound_static(&trunks);
            }
        }

        // Extra headroom: base entities + terrain + building + player model parts + trees + dynamic.
        let max_instances = (world.entities.len() + num_terrain_chunks + 3 + BODY_PART_COUNT + 32768 + 512) as u32;
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
            terrain_rb,
            terrain_col,
            terrain_rebuild_timer: 0.0,
            mesh_building_id,
            building: BuildingGrid::new(),
            mining,
            structures,
            place_prev: false,
            spawn_prev: false,
            debug_stats_prev: false,
            fast_prev: false,
            fast_mode: false,
            show_debug_ui: false,
            debug_ui_prev: false,
            water_time: 0.0,
            surface_width,
            surface_height,
            light_dir: Vec3::new(1.0, 3.0, 1.0).normalize(),
        };

        // Spawn a few mining nodes scattered on the terrain.
        engine.spawn_mining_nodes();

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

    pub fn update(&mut self, dt: f32, input: &InputState) {
        self.water_time += dt;
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
                Some(self.terrain_rb),
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

            // --- Camera-relative player movement ---
            self.apply_player_movement(input);

            // --- Water buoyancy ---
            self.apply_buoyancy(dt);

            self.physics.step(dt);

            // --- Update player model animation ---
            let vel = self.physics.body_linvel_xz(self.player_rb);
            let horiz_speed = Vec3::new(vel.x, 0.0, vel.z).length();
            self.player_model.update(dt, horiz_speed);
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

        // Update physics heightfield.
        let (heights, nrows, ncols) = self.terrain.heightfield_data();
        let scale = Vec3::new(
            (TERRAIN_HALF * 2) as f32,
            1.0,
            (TERRAIN_HALF * 2) as f32,
        );
        self.physics.update_heightfield(self.terrain_col, heights, nrows, ncols, scale);
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
        // Compute player yaw from movement direction or camera facing.
        let player_yaw = self.player_facing_yaw();
        let parts = self.player_model.compute_transforms(player_pos, player_yaw);
        for (i, (transform, _scale)) in parts.iter().enumerate() {
            transforms.push(*transform);
            instance_ids.push(pack_instance_id(MESH_CAPSULE, PLAYER_MODEL_OBJECT_ID + i as u32));
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
            crate::terrain::Biome::Forest => 0.0,
            crate::terrain::Biome::Desert => 1.0,
            crate::terrain::Biome::Mountains => 2.0,
            crate::terrain::Biome::Dungeon => 3.0,
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

        self.renderer.draw_frame(
            &transforms,
            &instance_ids,
            render_view,
            render_proj,
            Vec4::from((self.light_dir, 0.0)),
            Vec4::new(1.0, 0.95, 0.9, 1.0),
            player_vp,
            self.ghost.active,
            pry_progress,
            tool_type,
            debug_info,
            debug_info2,
        )
    }

    /// Compute the direction the player character should face.
    /// Uses movement direction if moving, otherwise faces camera forward.
    fn player_facing_yaw(&self) -> f32 {
        let vel = self.physics.body_linvel_xz(self.player_rb);
        let horiz = Vec3::new(vel.x, 0.0, vel.z);
        if horiz.length_squared() > 0.5 {
            // Face movement direction.
            (-horiz.x).atan2(-horiz.z)
        } else {
            // Face camera forward direction.
            let fwd = self.camera.forward_flat();
            (-fwd.x).atan2(-fwd.z)
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_width = width;
        self.surface_height = height;
        self.renderer.resize(width, height);
    }
}
