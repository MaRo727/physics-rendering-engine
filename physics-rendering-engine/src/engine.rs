use std::sync::Arc;

use anyhow::Result;
use glam::{Mat4, Quat, Vec3, Vec4};
use winit::window::Window;

use rapier3d::prelude::RigidBodyHandle;

use crate::physics::body::{PhysicsBody, WeightClass};
use crate::physics::world::PhysicsWorld;
use crate::renderer::{Renderer, pack_instance_id, MESH_CUBE, MESH_BALL, MESH_PYRAMID, MESH_TRIANGLE, MESH_SLOPE};

const PLAYER_SPEED: f32 = 5.0;
const JUMP_VELOCITY: f32 = 6.0;
const MOUSE_SENSITIVITY: f32 = 0.002;
const PICKUP_RANGE: f32 = 5.0;
const HOLD_DISTANCE: f32 = 3.0;
const HOLD_STIFFNESS: f32 = 20.0;
const PUNCH_RANGE: f32 = 3.0;
const BARE_PUNCH_FORCE: f32 = 8.0;

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    pub interact: bool,    // E — pick up / drop
    pub throw: bool,       // Left mouse button — throw held object
    pub mouse_dx: f32,
    pub mouse_dy: f32,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            jump: false,
            interact: false,
            throw: false,
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Engine types
// ---------------------------------------------------------------------------

pub struct EngineConfig {
    pub window_width: u32,
    pub window_height: u32,
    pub window_title: String,
    pub gravity: Vec3,
}

/// A world object with physics, a mesh type, a render scale, and an object id.
struct WorldObject {
    body: PhysicsBody,
    mesh_type: u32,
    render_scale: Vec3,
    object_id: u32,
}

pub struct Engine {
    pub config: EngineConfig,
    physics: PhysicsWorld,
    objects: Vec<WorldObject>,
    player: PhysicsBody,
    player_object_id: u32,
    renderer: Renderer,
    yaw: f32,
    pitch: f32,
    surface_width: u32,
    surface_height: u32,
    light_dir: Vec3,
    held_body: Option<RigidBodyHandle>,
    interact_prev: bool,
    punch_prev: bool,
}

impl Engine {
    pub fn new(config: EngineConfig, window: &Arc<Window>) -> Result<Self> {
        let surface_width = config.window_width;
        let surface_height = config.window_height;

        let mut physics = PhysicsWorld::new(config.gravity);

        let mut objects: Vec<WorldObject> = Vec::new();
        let mut next_id: u32 = 0;
        let mut alloc_id = || { let id = next_id; next_id += 1; id };

        // --- Cube (medium, 1x1x1) ---
        let cube_id = alloc_id();
        objects.push(WorldObject {
            body: PhysicsBody::new_dynamic_box(
                &mut physics,
                Vec3::new(0.0, 4.0, 0.0),
                Vec3::new(0.5, 0.5, 0.5),
                WeightClass::Medium,
            ),
            mesh_type: MESH_CUBE,
            render_scale: Vec3::ONE,
            object_id: cube_id,
        });

        // --- Floor (static, wide slab) ---
        let _floor_id = alloc_id();
        PhysicsBody::new_static_box(
            &mut physics,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(45.0, 0.5, 45.0),
        );
        // Floor has a static render object (no physics body tracked).
        // We'll handle it specially below.

        // --- Big cube (heavy, 3x3x3) ---
        let cube2_id = alloc_id();
        objects.push(WorldObject {
            body: PhysicsBody::new_dynamic_box(
                &mut physics,
                Vec3::new(3.0, 8.0, 0.0),
                Vec3::new(1.5, 1.5, 1.5),
                WeightClass::Heavy,
            ),
            mesh_type: MESH_CUBE,
            render_scale: Vec3::splat(3.0),
            object_id: cube2_id,
        });

        // --- Stick (light, thin box) ---
        let stick_id = alloc_id();
        objects.push(WorldObject {
            body: PhysicsBody::new_dynamic_box(
                &mut physics,
                Vec3::new(-2.0, 1.0, 3.0),
                Vec3::new(0.06, 0.06, 0.5),
                WeightClass::Light,
            ),
            mesh_type: MESH_CUBE,
            render_scale: Vec3::new(0.12, 0.12, 1.0),
            object_id: stick_id,
        });

        // --- Ball (medium, radius 0.5) ---
        let ball_id = alloc_id();
        objects.push(WorldObject {
            body: PhysicsBody::new_dynamic_ball(
                &mut physics,
                Vec3::new(-3.0, 5.0, -2.0),
                0.5,
                WeightClass::Medium,
            ),
            mesh_type: MESH_BALL,
            render_scale: Vec3::ONE,
            object_id: ball_id,
        });

        // --- Pyramid (medium) ---
        let pyramid_id = alloc_id();
        let pyramid_half = 0.5_f32;
        let pyramid_points = vec![
            Vec3::new(0.0, pyramid_half, 0.0),
            Vec3::new(-pyramid_half, -pyramid_half, pyramid_half),
            Vec3::new(pyramid_half, -pyramid_half, pyramid_half),
            Vec3::new(pyramid_half, -pyramid_half, -pyramid_half),
            Vec3::new(-pyramid_half, -pyramid_half, -pyramid_half),
        ];
        objects.push(WorldObject {
            body: PhysicsBody::new_dynamic_convex(
                &mut physics,
                Vec3::new(2.0, 6.0, -3.0),
                &pyramid_points,
                WeightClass::Medium,
            ),
            mesh_type: MESH_PYRAMID,
            render_scale: Vec3::ONE,
            object_id: pyramid_id,
        });

        // --- Triangle prism (light) ---
        let tri_id = alloc_id();
        let tri_points = vec![
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.0, 0.5, 0.5),
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.0, 0.5, -0.5),
        ];
        objects.push(WorldObject {
            body: PhysicsBody::new_dynamic_convex(
                &mut physics,
                Vec3::new(-4.0, 3.0, 1.0),
                &tri_points,
                WeightClass::Light,
            ),
            mesh_type: MESH_TRIANGLE,
            render_scale: Vec3::ONE,
            object_id: tri_id,
        });

        // --- Slope / ramp (heavy, static-like but dynamic so it can be punched) ---
        let slope_id = alloc_id();
        let slope_points = vec![
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
        ];
        objects.push(WorldObject {
            body: PhysicsBody::new_dynamic_convex(
                &mut physics,
                Vec3::new(5.0, 1.0, 2.0),
                &slope_points,
                WeightClass::Heavy,
            ),
            mesh_type: MESH_SLOPE,
            render_scale: Vec3::splat(2.0),
            object_id: slope_id,
        });

        // --- Player ---
        let player_id = alloc_id();
        let player = PhysicsBody::new_player_box(
            &mut physics,
            Vec3::new(0.0, 0.9, 4.0),
            Vec3::new(0.4, 0.9, 0.4),
        );

        // Total render instances: objects + floor + player
        let max_instances = (objects.len() + 2) as u32;
        let renderer = Renderer::new(window, max_instances)?;

        let yaw = -std::f32::consts::FRAC_PI_2;

        Ok(Self {
            config,
            physics,
            objects,
            player,
            player_object_id: player_id,
            renderer,
            yaw,
            pitch: 0.0,
            surface_width,
            surface_height,
            light_dir: Vec3::new(1.0, 3.0, 1.0).normalize(),
            held_body: None,
            interact_prev: false,
            punch_prev: false,
        })
    }

    pub fn update(&mut self, dt: f32, input: &InputState) {
        self.yaw   -= input.mouse_dx * MOUSE_SENSITIVITY;
        self.pitch  = (self.pitch - input.mouse_dy * MOUSE_SENSITIVITY)
            .clamp(-89_f32.to_radians(), 89_f32.to_radians());

        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let look_dir = Vec3::new(-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch);

        let player_pos = self.physics.body_position(self.player.rigid_body);
        let eye = player_pos + Vec3::new(0.0, 0.7, 0.0);

        // --- Interact (E) — edge-triggered pickup / drop ---
        let interact_pressed = input.interact && !self.interact_prev;
        self.interact_prev = input.interact;

        if interact_pressed {
            if let Some(held) = self.held_body.take() {
                self.physics.set_gravity_enabled(held, true);
            } else {
                let hit = self.physics.cast_ray(
                    eye,
                    look_dir,
                    PICKUP_RANGE,
                    self.player.collider,
                );
                if let Some(handle) = hit {
                    if self.physics.is_dynamic(handle) {
                        self.held_body = Some(handle);
                        self.physics.set_gravity_enabled(handle, false);
                    }
                }
            }
        }

        // --- LMB: throw held object, or punch ---
        let lmb_pressed = input.throw && !self.punch_prev;
        self.punch_prev = input.throw;

        if lmb_pressed {
            if let Some(held) = self.held_body.take() {
                let throw_speed = self.weight_class_of(held).throw_speed();
                self.physics.set_gravity_enabled(held, true);
                self.physics.set_body_linvel(held, look_dir * throw_speed);
            } else {
                let hit = self.physics.cast_ray(
                    eye,
                    look_dir,
                    PUNCH_RANGE,
                    self.player.collider,
                );
                if let Some(target_body) = hit {
                    if self.physics.is_dynamic(target_body) {
                        let wc = self.weight_class_of(target_body);
                        let force = look_dir * BARE_PUNCH_FORCE * wc.punch_knockback();
                        self.physics.apply_impulse(target_body, force);
                    }
                }
            }
        }

        // --- Hold: steer held object toward target point ---
        if let Some(held) = self.held_body {
            let target = eye + look_dir * HOLD_DISTANCE;
            let obj_pos = self.physics.body_position(held);
            let delta = target - obj_pos;
            self.physics.set_body_linvel(held, delta * HOLD_STIFFNESS);
        }

        // Flat (XZ-plane) movement directions derived from yaw only.
        let forward = Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos());
        let right   = Vec3::new( self.yaw.cos(), 0.0, -self.yaw.sin());

        let mut move_vel = Vec3::ZERO;
        if input.forward  { move_vel += forward; }
        if input.backward { move_vel -= forward; }
        if input.right    { move_vel += right; }
        if input.left     { move_vel -= right; }
        if move_vel.length_squared() > 0.0 {
            move_vel = move_vel.normalize() * PLAYER_SPEED;
        }

        let mut vy = self.physics.body_linvel_y(self.player.rigid_body);
        if input.jump && self.physics.is_on_ground(self.player.collider) {
            vy = JUMP_VELOCITY;
        }
        self.physics.set_body_linvel(
            self.player.rigid_body,
            Vec3::new(move_vel.x, vy, move_vel.z),
        );

        self.physics.step(dt);
    }

    pub fn render(&mut self) -> Result<()> {
        let mut transforms = Vec::new();
        let mut instance_ids = Vec::new();

        // Dynamic objects.
        for obj in &self.objects {
            let t = self.physics.body_transform(obj.body.rigid_body)
                * Mat4::from_scale(obj.render_scale);
            transforms.push(t);
            instance_ids.push(pack_instance_id(obj.mesh_type, obj.object_id));
        }

        // Static floor.
        let floor_transform = Mat4::from_scale_rotation_translation(
            Vec3::new(90.0, 1.0, 90.0),
            Quat::IDENTITY,
            Vec3::new(0.0, -0.5, 0.0),
        );
        transforms.push(floor_transform);
        instance_ids.push(pack_instance_id(MESH_CUBE, 1)); // object_id 1 = floor

        // Player (skip rendering — camera is inside).
        // Not added to the render list.

        let aspect = self.surface_width as f32 / self.surface_height.max(1) as f32;
        let (view, proj) = self.camera_matrices(aspect);

        self.renderer.draw_frame(
            &transforms,
            &instance_ids,
            view,
            proj,
            Vec4::from((self.light_dir, 0.0)),
            Vec4::new(1.0, 0.95, 0.9, 1.0),
        )
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_width = width;
        self.surface_height = height;
        self.renderer.resize(width, height);
    }

    fn weight_class_of(&self, handle: RigidBodyHandle) -> WeightClass {
        for obj in &self.objects {
            if obj.body.rigid_body == handle {
                return obj.body.weight_class;
            }
        }
        WeightClass::Medium
    }

    fn camera_matrices(&self, aspect: f32) -> (Mat4, Mat4) {
        let pos = self.physics.body_position(self.player.rigid_body);
        let eye = pos + Vec3::new(0.0, 0.7, 0.0);

        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let dir = Vec3::new(
            -sin_yaw * cos_pitch,
            sin_pitch,
            -cos_yaw * cos_pitch,
        );

        let view = Mat4::look_at_rh(eye, eye + dir, Vec3::Y);
        let proj = perspective_vk(std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0);
        (view, proj)
    }
}

/// Right-handed perspective projection for Vulkan (depth [0,1], Y flipped).
fn perspective_vk(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, -f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, far / (near - far), -1.0),
        Vec4::new(0.0, 0.0, far * near / (near - far), 0.0),
    )
}
