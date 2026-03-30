/// Camera systems: first-person (active) and third-person (preserved for future use).

use glam::{Mat4, Vec3, Vec4};

use crate::physics::world::PhysicsWorld;
use crate::input::InputState;
use super::player::perspective_vk;

const MOUSE_SENSITIVITY: f32 = 0.002;
const EYE_HEIGHT: f32 = 1.6; // eye height above physics body base

// ---------------------------------------------------------------------------
// First-person camera
// ---------------------------------------------------------------------------

pub struct FirstPersonCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub eye: Vec3,
}

impl FirstPersonCamera {
    pub fn new() -> Self {
        Self {
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            eye: Vec3::ZERO,
        }
    }

    /// Update camera angles from mouse input.
    pub fn look(&mut self, input: &InputState) {
        self.yaw -= input.mouse_dx * MOUSE_SENSITIVITY;
        self.pitch = (self.pitch + input.mouse_dy * MOUSE_SENSITIVITY)
            .clamp(-89_f32.to_radians(), 89_f32.to_radians());
    }

    /// Update eye position from player physics body position.
    pub fn update(&mut self, player_pos: Vec3) {
        self.eye = player_pos + Vec3::new(0.0, EYE_HEIGHT, 0.0);
    }

    /// Look direction vector from yaw and pitch.
    pub fn look_dir(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Vec3::new(sin_yaw * cos_pitch, -sin_pitch, cos_yaw * cos_pitch)
    }

    /// Forward direction on XZ plane (for player movement).
    pub fn forward_flat(&self) -> Vec3 {
        let dir = self.look_dir();
        Vec3::new(dir.x, 0.0, dir.z).normalize_or_zero()
    }

    /// Right direction (perpendicular to forward, on XZ plane).
    pub fn right_flat(&self) -> Vec3 {
        let fwd = self.forward_flat();
        Vec3::new(-fwd.z, 0.0, fwd.x)
    }

    /// Compute view and projection matrices.
    pub fn camera_matrices(&self, aspect: f32) -> (Mat4, Mat4) {
        self.camera_matrices_fov(std::f32::consts::FRAC_PI_4, aspect)
    }

    /// Compute view and projection with custom FOV (radians).
    pub fn camera_matrices_fov(&self, fov_y: f32, aspect: f32) -> (Mat4, Mat4) {
        let target = self.eye + self.look_dir();
        let view = Mat4::look_at_rh(self.eye, target, Vec3::Y);
        let proj = perspective_vk(fov_y, aspect, 0.1, 5000.0);
        (view, proj)
    }

    /// Update camera angles from mouse input with sensitivity multiplier.
    pub fn look_with_sensitivity(&mut self, input: &InputState, sensitivity: f32) {
        let sens = MOUSE_SENSITIVITY * sensitivity;
        self.yaw -= input.mouse_dx * sens;
        self.pitch = (self.pitch + input.mouse_dy * sens)
            .clamp(-89_f32.to_radians(), 89_f32.to_radians());
    }
}

// ---------------------------------------------------------------------------
// Third-person orbit camera (preserved for future use)
// ---------------------------------------------------------------------------
const DEFAULT_DISTANCE: f32 = 6.0;
const MIN_DISTANCE: f32 = 1.5;
const MAX_DISTANCE: f32 = 12.0;
const HEIGHT_OFFSET: f32 = 1.5; // camera target is above player center
const COLLISION_MARGIN: f32 = 0.3; // pull camera forward from wall

pub struct ThirdPersonCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub eye: Vec3,
    pub target: Vec3,
}

impl ThirdPersonCamera {
    pub fn new() -> Self {
        Self {
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.3, // slightly above horizontal
            distance: DEFAULT_DISTANCE,
            eye: Vec3::ZERO,
            target: Vec3::ZERO,
        }
    }

    /// Update camera angles from mouse input.
    pub fn look(&mut self, input: &InputState) {
        self.yaw -= input.mouse_dx * MOUSE_SENSITIVITY;
        self.pitch = (self.pitch + input.mouse_dy * MOUSE_SENSITIVITY)
            .clamp(-60_f32.to_radians(), 80_f32.to_radians());
    }

    /// Update camera position based on player position + wall collision.
    pub fn update(
        &mut self,
        player_pos: Vec3,
        physics: &PhysicsWorld,
        player_collider: rapier3d::prelude::ColliderHandle,
    ) {
        self.target = player_pos + Vec3::new(0.0, HEIGHT_OFFSET, 0.0);

        // Compute desired camera position from spherical coordinates.
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let offset = Vec3::new(
            sin_yaw * cos_pitch,
            sin_pitch,
            cos_yaw * cos_pitch,
        ) * self.distance;

        let desired = self.target + offset;

        // Raycast from target toward desired position to detect walls.
        let dir = offset;
        let dir_len = dir.length();
        if dir_len > 0.001 {
            let dir_norm = dir / dir_len;
            let actual_dist = physics
                .cast_ray_distance(self.target, dir_norm, dir_len, player_collider)
                .map(|d| (d - COLLISION_MARGIN).max(MIN_DISTANCE))
                .unwrap_or(dir_len);

            self.eye = self.target + dir_norm * actual_dist;
        } else {
            self.eye = desired;
        }
    }

    /// Forward direction from camera to target (for player movement).
    pub fn forward_flat(&self) -> Vec3 {
        let dir = self.target - self.eye;
        Vec3::new(dir.x, 0.0, dir.z).normalize_or_zero()
    }

    /// Right direction (perpendicular to forward, on XZ plane).
    pub fn right_flat(&self) -> Vec3 {
        let fwd = self.forward_flat();
        Vec3::new(-fwd.z, 0.0, fwd.x)
    }

    /// Look direction from camera eye toward target.
    pub fn look_dir(&self) -> Vec3 {
        (self.target - self.eye).normalize_or_zero()
    }

    /// Compute view and projection matrices.
    pub fn camera_matrices(&self, aspect: f32) -> (Mat4, Mat4) {
        let view = Mat4::look_at_rh(self.eye, self.target, Vec3::Y);
        let proj = perspective_vk(std::f32::consts::FRAC_PI_4, aspect, 0.1, 5000.0);
        (view, proj)
    }
}

/// Extract 6 frustum planes from a view-projection matrix (Vulkan [0,1] depth).
pub fn extract_frustum_planes(vp: Mat4) -> [Vec4; 6] {
    let r0 = vp.row(0);
    let r1 = vp.row(1);
    let r2 = vp.row(2);
    let r3 = vp.row(3);

    let mut planes = [
        r3 + r0,
        r3 - r0,
        r3 + r1,
        r3 - r1,
        r2,
        r3 - r2,
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
