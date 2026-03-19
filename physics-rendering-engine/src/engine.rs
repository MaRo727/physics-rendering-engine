use std::sync::Arc;

use anyhow::Result;
use glam::{Mat4, Vec3, Vec4};
use winit::window::Window;

use crate::building::{self, BuildingGrid};
use crate::game::entity::{Entity, EntityKind};
use crate::game::world::World;
use crate::input::InputState;
use crate::interaction::Interaction;
use crate::physics::body::{PhysicsBody, WeightClass};
use crate::physics::world::PhysicsWorld;
use crate::player::{Player, GhostCamera, extract_frustum_planes, is_sphere_in_frustum};
use crate::renderer::{Renderer, pack_instance_id, MESH_CUBE, MESH_TERRAIN_BASE};
use crate::scene::{self, UNIT_BOUNDING_RADIUS};
use crate::terrain::{TerrainGrid, TerrainChunkInfo};

const PLACE_RANGE: f32 = 8.0;
const BUILDING_OBJECT_ID: u32 = 0xFFF0;

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
    player: Player,
    renderer: Renderer,
    interaction: Interaction,
    ghost: GhostCamera,
    terrain: TerrainGrid,
    terrain_chunks: Vec<TerrainChunkInfo>,
    terrain_object_id: u32,
    mesh_building_id: u32,
    building: BuildingGrid,
    place_prev: bool,
    spawn_prev: bool,
    debug_stats_prev: bool,
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
        let (chunk_meshes, terrain_chunks, full_mesh) =
            terrain.generate_chunks(MESH_TERRAIN_BASE);
        let num_terrain_chunks = chunk_meshes.len();

        // Add trimesh collider for terrain (exact match with visual mesh).
        let (phys_verts, phys_tris) = TerrainGrid::physics_trimesh(&full_mesh);
        physics.add_trimesh(phys_verts, phys_tris);

        // Build game world.
        let world = World::new(entities, player_id, next_id);

        // Spawn player on terrain surface.
        let spawn_h = terrain.get_height(0, 4) + 1.0 + 0.9;
        let player_body = world.player().body.clone_handles();
        let player = Player::new(player_body, player_id);
        physics.set_body_position(player.body.rigid_body, Vec3::new(0.0, spawn_h, 4.0));

        let terrain_object_id = 0xFFF1;

        // Headroom for dynamically-spawned cubes pried from the building grid.
        let max_instances = (world.entities.len() + num_terrain_chunks + 3 + 256) as u32;
        let renderer = Renderer::new(window, max_instances, chunk_meshes)?;
        let mesh_building_id = renderer.mesh_building_id();

        Ok(Self {
            config,
            physics,
            world,
            player,
            renderer,
            interaction: Interaction::default(),
            ghost: GhostCamera::default(),
            terrain,
            terrain_chunks,
            terrain_object_id,
            mesh_building_id,
            building: BuildingGrid::new(),
            place_prev: false,
            spawn_prev: false,
            debug_stats_prev: false,
            surface_width,
            surface_height,
            light_dir: Vec3::new(1.0, 3.0, 1.0).normalize(),
        })
    }

    pub fn update(&mut self, dt: f32, input: &InputState) {
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
            let (view, proj) = self.player.camera_matrices(&self.physics, aspect);
            self.ghost.activate(&self.player, &self.physics, view, proj);
            self.interaction.drop_held(&mut self.physics);
        }

        // --- Tool cycling (Tab) ---
        self.interaction.cycle_tool(input.cycle_tool);

        if self.ghost.active {
            self.ghost.update(dt, input);
            self.physics.set_body_linvel(self.player.body.rigid_body, Vec3::ZERO);
            self.physics.step(dt);
        } else {
            self.player.look(input);
            let eye = self.player.eye(&self.physics);
            let look_dir = self.player.look_dir();

            // Determine if crosshair is aimed at a building cell (for pry logic).
            let building_cell_aimed_at = self.physics
                .cast_ray_detailed(eye, look_dir, PLACE_RANGE, self.player.body.collider)
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
                eye,
                look_dir,
                input.interact,
                input.throw,
                self.player.body.collider,
                dt,
                building_cell_aimed_at,
            );

            // --- Handle item pickup ---
            if let Some(drop_id) = interaction_result.picked_up_item {
                if self.world.pickup_item(drop_id) {
                    // Remove the entity from physics and world
                    if let Some(idx) = self.world.entities.iter().position(|e| e.id == drop_id) {
                        let entity = self.world.entities.swap_remove(idx);
                        self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                    }
                }
            }

            // --- Handle pried building cube: remove from grid, spawn as held dynamic object ---
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

            // --- Handle axe split: split cube into two halves ---
            if let Some(target_body) = interaction_result.axe_hit {
                self.split_cube(target_body, eye, look_dir);
            }

            // --- RMB: place held cube into building grid ---
            let place_pressed = input.place && !self.place_prev;
            self.place_prev = input.place;

            if place_pressed {
                if let Some(held_handle) = self.interaction.held_body.take() {
                    let pos = self.physics.body_position(held_handle);
                    let (cx, cy, cz) = building::snap_to_grid(pos);

                    if self.building.place(&mut self.physics, cx, cy, cz) {
                        // Remove the dynamic object from the world.
                        if let Some(idx) = self.world.entities.iter().position(|o| o.body.rigid_body == held_handle) {
                            let entity = self.world.entities.swap_remove(idx);
                            self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                        }
                    } else {
                        // Cell already occupied — keep holding.
                        self.interaction.held_body = Some(held_handle);
                    }
                }
            }

            // --- F: spawn a new block in front of the player ---
            let spawn_pressed = input.spawn && !self.spawn_prev;
            self.spawn_prev = input.spawn;

            if spawn_pressed && self.interaction.held_body.is_none() {
                let spawn_pos = eye + look_dir * 3.0;
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

            self.player.apply_movement(&mut self.physics, input);
            self.physics.step(dt);
        }

        // --- Game tick: regen, etc. ---
        self.world.game_tick(dt);
    }

    /// Split a cube object into two halves along the axis most aligned with the hit.
    fn split_cube(&mut self, target_body: rapier3d::prelude::RigidBodyHandle, _eye: Vec3, look_dir: Vec3) {
        // Find the entity and verify it's a cube.
        let obj_idx = match self.world.entities.iter().position(|o| o.body.rigid_body == target_body) {
            Some(idx) => idx,
            None => return,
        };

        if self.world.entities[obj_idx].mesh_type != MESH_CUBE {
            // Not a cube — just apply a knockback impulse instead.
            let wc = self.world.entities[obj_idx].body.weight_class;
            let force = look_dir * 8.0 * wc.punch_knockback();
            self.physics.apply_impulse(target_body, force);
            return;
        }

        let entity = self.world.entities.swap_remove(obj_idx);
        let pos = self.physics.body_position(entity.body.rigid_body);
        let transform = self.physics.body_transform(entity.body.rigid_body);
        let scale = entity.render_scale;

        // Determine split axis from look direction (split perpendicular to the
        // most-aligned local axis so the cut feels natural).
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

        // Remove original object from physics.
        self.physics.remove_body(entity.body.rigid_body, entity.body.collider);

        // Compute half scales and offsets.
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
            self.player.camera_matrices(&self.physics, aspect)
        };

        let (render_view, render_proj) = if self.ghost.active {
            self.ghost.camera_matrices(aspect)
        } else {
            (cull_view, cull_proj)
        };

        let frustum = extract_frustum_planes(cull_proj * cull_view);

        let mut transforms = Vec::new();
        let mut instance_ids = Vec::new();

        for entity in &self.world.entities {
            // Skip the player entity (first-person, not rendered).
            if entity.kind == EntityKind::Player { continue; }

            let pos = self.physics.body_position(entity.body.rigid_body);
            if !is_sphere_in_frustum(&frustum, pos, entity.bounding_radius) {
                continue;
            }
            let t = self.physics.body_transform(entity.body.rigid_body)
                * Mat4::from_scale(entity.render_scale);
            transforms.push(t);
            instance_ids.push(pack_instance_id(entity.mesh_type, entity.id));
        }

        // Terrain chunks — frustum-culled, each at identity (vertices in world space).
        for chunk in &self.terrain_chunks {
            if is_sphere_in_frustum(&frustum, chunk.center, chunk.radius) {
                transforms.push(Mat4::IDENTITY);
                instance_ids.push(pack_instance_id(chunk.mesh_type, self.terrain_object_id));
            }
        }

        // Building mesh — single instance at identity (vertices already in world space).
        if !self.building.is_empty() && self.renderer.has_building_blas() {
            transforms.push(Mat4::IDENTITY);
            instance_ids.push(pack_instance_id(self.mesh_building_id, BUILDING_OBJECT_ID));
        }

        let player_vp = cull_proj * cull_view;

        let pry_progress = self.interaction.pry_progress();
        let tool_type = match self.interaction.equipped_tool {
            crate::interaction::ToolType::Hands => 0.0,
            crate::interaction::ToolType::Axe => 1.0,
        };

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
        )
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_width = width;
        self.surface_height = height;
        self.renderer.resize(width, height);
    }
}
