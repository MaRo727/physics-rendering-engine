use std::sync::Arc;

use anyhow::Result;
use glam::{Mat4, Quat, Vec3, Vec4};
use winit::window::Window;

use crate::physics::body::PhysicsBody;
use crate::physics::world::PhysicsWorld;
use crate::renderer::Renderer;

const PLAYER_SPEED: f32 = 5.0;
const JUMP_VELOCITY: f32 = 6.0;
const MOUSE_SENSITIVITY: f32 = 0.002;

// Render-object indices.
const CUBE_IDX: usize = 0;
const PLAYER_IDX: usize = 2;
const CUBE2_IDX: usize = 3;

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub jump: bool,
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
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Engine types
// ---------------------------------------------------------------------------

pub struct MeshId(pub u32);

pub struct RenderObject {
    pub mesh_id: MeshId,
    pub transform: Mat4,
}

pub struct EngineConfig {
    pub window_width: u32,
    pub window_height: u32,
    pub window_title: String,
    pub gravity: Vec3,
}

pub struct Engine {
    pub config: EngineConfig,
    physics: PhysicsWorld,
    cube: PhysicsBody,
    cube2: PhysicsBody,
    player: PhysicsBody,
    renderer: Renderer,
    render_objects: Vec<RenderObject>,
    yaw: f32,   // camera horizontal rotation (radians)
    pitch: f32, // camera vertical rotation (radians)
    surface_width: u32,
    surface_height: u32,
    light_dir: Vec3,
}

impl Engine {
    pub fn new(config: EngineConfig, window: &Arc<Window>) -> Result<Self> {
        let surface_width = config.window_width;
        let surface_height = config.window_height;

        let renderer = Renderer::new(window)?;

        let mut physics = PhysicsWorld::new(config.gravity);

        // Falling cube — 1×1×1 box, starts 4 units above the floor.
        let cube = PhysicsBody::new_dynamic_box(
            &mut physics,
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
        );

        // Static floor — thin wide slab at y = -0.5 (top surface at y = 0).
        PhysicsBody::new_static_box(
            &mut physics,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(45.0, 0.5, 45.0),
        );

        // Second falling cube — 3×3×3, offset so it doesn't land on the first.
        let cube2 = PhysicsBody::new_dynamic_box(
            &mut physics,
            Vec3::new(3.0, 8.0, 0.0),
            Vec3::new(1.5, 1.5, 1.5),
        );

        // Player — tall box on the floor, 4 units back from centre.
        // Half-extents (0.4, 0.9, 0.4) → full size 0.8 × 1.8 × 0.8, centre at y = 0.9.
        let player = PhysicsBody::new_player_box(
            &mut physics,
            Vec3::new(0.0, 0.9, 4.0),
            Vec3::new(0.4, 0.9, 0.4),
        );

        let floor_transform = Mat4::from_scale_rotation_translation(
            Vec3::new(90.0, 1.0, 90.0),
            Quat::IDENTITY,
            Vec3::new(0.0, -0.5, 0.0),
        );

        let render_objects = vec![
            RenderObject { mesh_id: MeshId(0), transform: Mat4::IDENTITY }, // CUBE_IDX
            RenderObject { mesh_id: MeshId(0), transform: floor_transform }, // FLOOR_IDX
            RenderObject { mesh_id: MeshId(0), transform: Mat4::IDENTITY }, // PLAYER_IDX
            RenderObject { mesh_id: MeshId(0), transform: Mat4::IDENTITY }, // CUBE2_IDX
        ];

        // Start facing toward the falling cube (-Z direction).
        let yaw = -std::f32::consts::FRAC_PI_2;

        Ok(Self {
            config,
            physics,
            cube,
            cube2,
            player,
            renderer,
            render_objects,
            yaw,
            pitch: 0.0,
            surface_width,
            surface_height,
            light_dir: Vec3::new(1.0, 3.0, 1.0).normalize(),
        })
    }

    pub fn update(&mut self, dt: f32, input: &InputState) {
        // Update camera orientation from mouse delta.
        self.yaw   -= input.mouse_dx * MOUSE_SENSITIVITY;
        self.pitch  = (self.pitch - input.mouse_dy * MOUSE_SENSITIVITY)
            .clamp(-89_f32.to_radians(), 89_f32.to_radians());

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

        // Set player XZ velocity; preserve Y so gravity still applies.
        let mut vy = self.physics.body_linvel_y(self.player.rigid_body);
        if input.jump && self.physics.is_on_ground(self.player.collider) {
            vy = JUMP_VELOCITY;
        }
        self.physics.set_body_linvel(
            self.player.rigid_body,
            Vec3::new(move_vel.x, vy, move_vel.z),
        );

        self.physics.step(dt);

        // Extract render transforms.
        self.render_objects[CUBE_IDX].transform =
            self.physics.body_transform(self.cube.rigid_body);

        self.render_objects[CUBE2_IDX].transform =
            self.physics.body_transform(self.cube2.rigid_body) * Mat4::from_scale(Vec3::splat(3.0));

        let body_t = self.physics.body_transform(self.player.rigid_body);
        self.render_objects[PLAYER_IDX].transform =
            body_t * Mat4::from_scale(Vec3::new(0.8, 1.8, 0.8));
        // FLOOR_IDX is static — transform never changes.
    }

    pub fn render(&mut self) -> Result<()> {
        // Skip the player mesh — the camera is inside it, and its inner faces
        // would occlude the entire scene.
        let (transforms, instance_ids): (Vec<Mat4>, Vec<u32>) = self.render_objects.iter()
            .enumerate()
            .filter(|(i, _)| *i != PLAYER_IDX)
            .map(|(i, o)| (o.transform, i as u32))
            .unzip();

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

    fn camera_matrices(&self, aspect: f32) -> (Mat4, Mat4) {
        // Eye at the player's head — 0.7 m above body centre.
        let pos = self.physics.body_position(self.player.rigid_body);
        let eye = pos + Vec3::new(0.0, 0.7, 0.0);

        // Look direction from yaw + pitch.
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
