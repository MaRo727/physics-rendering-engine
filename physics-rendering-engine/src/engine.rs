use std::sync::Arc;

use anyhow::Result;
use glam::{Mat4, Quat, Vec3, Vec4};
use winit::window::Window;

use crate::building::{self, BuildingGrid};
use crate::input::InputState;
use crate::interaction::Interaction;
use crate::physics::body::{PhysicsBody, WeightClass};
use crate::physics::world::PhysicsWorld;
use crate::player::{Player, GhostCamera, extract_frustum_planes, is_sphere_in_frustum};
use crate::renderer::{Renderer, pack_instance_id, MESH_CUBE, MESH_BUILDING};
use crate::scene::{self, WorldObject, UNIT_BOUNDING_RADIUS};

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
    objects: Vec<WorldObject>,
    player: Player,
    renderer: Renderer,
    interaction: Interaction,
    ghost: GhostCamera,
    building: BuildingGrid,
    place_prev: bool,
    spawn_prev: bool,
    next_object_id: u32,
    surface_width: u32,
    surface_height: u32,
    light_dir: Vec3,
}

impl Engine {
    pub fn new(config: EngineConfig, window: &Arc<Window>) -> Result<Self> {
        let surface_width = config.window_width;
        let surface_height = config.window_height;

        let mut physics = PhysicsWorld::new(config.gravity);
        let (objects, player_body, player_id, next_object_id) = scene::build_scene(&mut physics);
        let player = Player::new(player_body, player_id);

        // Headroom for dynamically-spawned cubes pried from the building grid.
        let max_instances = (objects.len() + 3 + 256) as u32;
        let renderer = Renderer::new(window, max_instances)?;

        Ok(Self {
            config,
            physics,
            objects,
            player,
            renderer,
            interaction: Interaction::default(),
            ghost: GhostCamera::default(),
            building: BuildingGrid::new(),
            place_prev: false,
            spawn_prev: false,
            next_object_id,
            surface_width,
            surface_height,
            light_dir: Vec3::new(1.0, 3.0, 1.0).normalize(),
        })
    }

    pub fn update(&mut self, dt: f32, input: &InputState) {
        // --- Ghost mode toggle ---
        let just_entered_ghost = self.ghost.handle_toggle(input);

        if just_entered_ghost {
            let aspect = self.surface_width as f32 / self.surface_height.max(1) as f32;
            let (view, proj) = self.player.camera_matrices(&self.physics, aspect);
            self.ghost.activate(&self.player, &self.physics, view, proj);
            self.interaction.drop_held(&mut self.physics);
        }

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

            let pried_cell = self.interaction.update(
                &mut self.physics,
                &self.objects,
                eye,
                look_dir,
                input.interact,
                input.throw,
                self.player.body.collider,
                dt,
                building_cell_aimed_at,
            );

            // --- Handle pried building cube: remove from grid, spawn as held dynamic object ---
            if let Some((cx, cy, cz)) = pried_cell {
                self.building.remove(&mut self.physics, cx, cy, cz);

                let center = building::cell_center(cx, cy, cz);
                let obj_id = self.next_object_id;
                self.next_object_id += 1;

                let body = PhysicsBody::new_dynamic_box(
                    &mut self.physics,
                    center,
                    Vec3::splat(0.5),
                    WeightClass::Medium,
                );
                self.physics.set_gravity_enabled(body.rigid_body, false);
                self.interaction.held_body = Some(body.rigid_body);

                self.objects.push(WorldObject {
                    body,
                    mesh_type: MESH_CUBE,
                    render_scale: Vec3::ONE,
                    object_id: obj_id,
                    bounding_radius: UNIT_BOUNDING_RADIUS,
                });
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
                        if let Some(idx) = self.objects.iter().position(|o| o.body.rigid_body == held_handle) {
                            let obj = self.objects.swap_remove(idx);
                            self.physics.remove_body(obj.body.rigid_body, obj.body.collider);
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
                let obj_id = self.next_object_id;
                self.next_object_id += 1;

                let body = PhysicsBody::new_dynamic_box(
                    &mut self.physics,
                    spawn_pos,
                    Vec3::splat(0.5),
                    WeightClass::Medium,
                );
                self.physics.set_gravity_enabled(body.rigid_body, false);
                self.interaction.held_body = Some(body.rigid_body);

                self.objects.push(WorldObject {
                    body,
                    mesh_type: MESH_CUBE,
                    render_scale: Vec3::ONE,
                    object_id: obj_id,
                    bounding_radius: UNIT_BOUNDING_RADIUS,
                });
            }

            self.player.apply_movement(&mut self.physics, input);
            self.physics.step(dt);
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

        for obj in &self.objects {
            let pos = self.physics.body_position(obj.body.rigid_body);
            if !is_sphere_in_frustum(&frustum, pos, obj.bounding_radius) {
                continue;
            }
            let t = self.physics.body_transform(obj.body.rigid_body)
                * Mat4::from_scale(obj.render_scale);
            transforms.push(t);
            instance_ids.push(pack_instance_id(obj.mesh_type, obj.object_id));
        }

        // Static floor — always rendered.
        let floor_transform = Mat4::from_scale_rotation_translation(
            Vec3::new(90.0, 1.0, 90.0),
            Quat::IDENTITY,
            Vec3::new(0.0, -0.5, 0.0),
        );
        transforms.push(floor_transform);
        instance_ids.push(pack_instance_id(MESH_CUBE, 1));

        // Building mesh — single instance at identity (vertices already in world space).
        if !self.building.is_empty() && self.renderer.has_building_blas() {
            transforms.push(Mat4::IDENTITY);
            instance_ids.push(pack_instance_id(MESH_BUILDING, BUILDING_OBJECT_ID));
        }

        let player_vp = cull_proj * cull_view;

        let pry_progress = self.interaction.pry_progress();

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
        )
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_width = width;
        self.surface_height = height;
        self.renderer.resize(width, height);
    }
}
