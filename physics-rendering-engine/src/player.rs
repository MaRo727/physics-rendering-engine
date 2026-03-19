use glam::{Mat4, Vec3, Vec4};

use crate::input::InputState;
use crate::physics::body::PhysicsBody;
use crate::physics::world::PhysicsWorld;

const PLAYER_SPEED: f32 = 5.0;
const GHOST_SPEED: f32 = 12.0;
const JUMP_VELOCITY: f32 = 6.0;
const MOUSE_SENSITIVITY: f32 = 0.002;

pub struct Player {
    pub body: PhysicsBody,
    pub object_id: u32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Player {
    pub fn new(body: PhysicsBody, object_id: u32) -> Self {
        Self {
            body,
            object_id,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
        }
    }

    /// Update player camera look direction from mouse input.
    pub fn look(&mut self, input: &InputState) {
        self.yaw   -= input.mouse_dx * MOUSE_SENSITIVITY;
        self.pitch  = (self.pitch - input.mouse_dy * MOUSE_SENSITIVITY)
            .clamp(-89_f32.to_radians(), 89_f32.to_radians());
    }

    /// Eye position (camera position).
    pub fn eye(&self, physics: &PhysicsWorld) -> Vec3 {
        let pos = physics.body_position(self.body.rigid_body);
        pos + Vec3::new(0.0, 0.7, 0.0)
    }

    /// Forward look direction (for interaction, camera, etc.).
    pub fn look_dir(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Vec3::new(-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch)
    }

    /// Apply movement input to the player rigid body.
    pub fn apply_movement(&self, physics: &mut PhysicsWorld, input: &InputState) {
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

        let mut vy = physics.body_linvel_y(self.body.rigid_body);
        if input.jump && physics.is_on_ground(self.body.collider) {
            vy = JUMP_VELOCITY;
        }
        physics.set_body_linvel(
            self.body.rigid_body,
            Vec3::new(move_vel.x, vy, move_vel.z),
        );
    }

    /// Compute view and projection matrices for the player camera.
    pub fn camera_matrices(&self, physics: &PhysicsWorld, aspect: f32) -> (Mat4, Mat4) {
        let eye = self.eye(physics);
        let dir = self.look_dir();
        let view = Mat4::look_at_rh(eye, eye + dir, Vec3::Y);
        let proj = perspective_vk(std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0);
        (view, proj)
    }
}

// ---------------------------------------------------------------------------
// Ghost / debug camera
// ---------------------------------------------------------------------------

pub struct GhostCamera {
    pub active: bool,
    toggle_prev: bool,
    pub eye: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub frozen_view: Mat4,
    pub frozen_proj: Mat4,
}

impl Default for GhostCamera {
    fn default() -> Self {
        Self {
            active: false,
            toggle_prev: false,
            eye: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            frozen_view: Mat4::IDENTITY,
            frozen_proj: Mat4::IDENTITY,
        }
    }
}

impl GhostCamera {
    /// Returns true if ghost mode was just toggled on this frame.
    pub fn handle_toggle(&mut self, input: &InputState) -> bool {
        let toggled = input.toggle_ghost && !self.toggle_prev;
        self.toggle_prev = input.toggle_ghost;

        if toggled {
            self.active = !self.active;
        }
        toggled && self.active
    }

    /// Activate ghost mode from the current player state.
    pub fn activate(&mut self, player: &Player, physics: &PhysicsWorld, view: Mat4, proj: Mat4) {
        self.frozen_view = view;
        self.frozen_proj = proj;
        self.eye = player.eye(physics);
        self.yaw = player.yaw;
        self.pitch = player.pitch;
    }

    /// Activate ghost mode from a third-person camera.
    pub fn activate_from_camera(&mut self, camera: &crate::game::camera::ThirdPersonCamera, view: Mat4, proj: Mat4) {
        self.frozen_view = view;
        self.frozen_proj = proj;
        self.eye = camera.eye;
        self.yaw = camera.yaw;
        self.pitch = camera.pitch;
    }

    /// Update ghost camera movement.
    pub fn update(&mut self, dt: f32, input: &InputState) {
        self.yaw -= input.mouse_dx * MOUSE_SENSITIVITY;
        self.pitch = (self.pitch - input.mouse_dy * MOUSE_SENSITIVITY)
            .clamp(-89_f32.to_radians(), 89_f32.to_radians());

        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let fwd = Vec3::new(-sy * cp, sp, -cy * cp);
        let right = Vec3::new(cy, 0.0, -sy);

        let mut vel = Vec3::ZERO;
        if input.forward  { vel += fwd; }
        if input.backward { vel -= fwd; }
        if input.right    { vel += right; }
        if input.left     { vel -= right; }
        if input.jump     { vel += Vec3::Y; }
        if input.descend  { vel -= Vec3::Y; }
        if vel.length_squared() > 0.0 {
            vel = vel.normalize() * GHOST_SPEED;
        }
        self.eye += vel * dt;
    }

    /// Compute view and projection matrices for the ghost camera.
    pub fn camera_matrices(&self, aspect: f32) -> (Mat4, Mat4) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let dir = Vec3::new(-sy * cp, sp, -cy * cp);
        let view = Mat4::look_at_rh(self.eye, self.eye + dir, Vec3::Y);
        let proj = perspective_vk(std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0);
        (view, proj)
    }
}

// ---------------------------------------------------------------------------
// Frustum culling
// ---------------------------------------------------------------------------

/// Extract 6 frustum planes from a view-projection matrix (Vulkan [0,1] depth).
/// Each plane is (nx, ny, nz, d) — inside when dot(n, point) + d >= 0.
pub fn extract_frustum_planes(vp: Mat4) -> [Vec4; 6] {
    let r0 = vp.row(0);
    let r1 = vp.row(1);
    let r2 = vp.row(2);
    let r3 = vp.row(3);

    let mut planes = [
        r3 + r0,    // left
        r3 - r0,    // right
        r3 + r1,    // bottom
        r3 - r1,    // top
        r2,         // near  (Vulkan [0,1] depth: z_clip >= 0)
        r3 - r2,    // far   (w_clip - z_clip >= 0)
    ];

    for p in &mut planes {
        let len = Vec3::new(p.x, p.y, p.z).length();
        if len > 0.0 {
            *p /= len;
        }
    }
    planes
}

/// Test whether a bounding sphere is at least partially inside the frustum.
pub fn is_sphere_in_frustum(planes: &[Vec4; 6], center: Vec3, radius: f32) -> bool {
    for plane in planes {
        let dist = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
        if dist < -radius {
            return false;
        }
    }
    true
}

/// Right-handed perspective projection for Vulkan (depth [0,1], Y flipped).
pub fn perspective_vk(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    Mat4::from_cols(
        Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, -f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, far / (near - far), -1.0),
        Vec4::new(0.0, 0.0, far * near / (near - far), 0.0),
    )
}
