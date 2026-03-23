/// Player model: capsule body with two sphere fists that animate on punch.

use glam::{Mat4, Vec3};

use crate::renderer::{MESH_CAPSULE, MESH_FIST};

pub const BODY_PART_COUNT: usize = 3;
/// Number of parts rendered in first-person view (fists only, no body).
pub const FP_PART_COUNT: usize = 2;

/// Player scale: a tall pill roughly matching the physics collider.
const PLAYER_SCALE: Vec3 = Vec3::new(0.8, 1.8, 0.8);
const FIST_SCALE: Vec3 = Vec3::new(0.25, 0.25, 0.25);

// Fist rest positions relative to body center (model faces -Z after yaw rotation).
// Positive Z = forward in local space (yaw flips it to face camera forward).
const LEFT_FIST_REST: Vec3 = Vec3::new(-0.55, 0.3, 0.3);
const RIGHT_FIST_REST: Vec3 = Vec3::new(0.55, 0.3, 0.3);

// How far forward the right fist extends during the active punch phase.
const PUNCH_REACH: f32 = 0.8;

pub struct PlayerModel {
    attack_progress: f32,
}

impl PlayerModel {
    pub fn new() -> Self {
        Self {
            attack_progress: 0.0,
        }
    }

    pub fn update(&mut self, _dt: f32, _move_speed: f32) {}

    pub fn set_attack_progress(&mut self, progress: f32) {
        self.attack_progress = progress;
    }

    /// Returns mesh type for each body part.
    pub fn mesh_types() -> [u32; BODY_PART_COUNT] {
        [MESH_CAPSULE, MESH_FIST, MESH_FIST]
    }

    /// Returns a transform + scale for each body part.
    /// `root_pos` is the physics body center, `yaw` is facing direction.
    pub fn compute_transforms(
        &self,
        root_pos: Vec3,
        yaw: f32,
    ) -> [(Mat4, Vec3); BODY_PART_COUNT] {
        let body_transform = Mat4::from_translation(root_pos)
            * Mat4::from_rotation_y(yaw)
            * Mat4::from_scale(PLAYER_SCALE);

        // Left fist: punches forward with a counter-clockwise arc.
        // Right fist: stays at rest position.
        let (left_offset_z, left_offset_x) = self.punch_offsets();

        let left_pos = LEFT_FIST_REST + Vec3::new(-left_offset_x, 0.0, left_offset_z);
        let right_pos = RIGHT_FIST_REST;

        let left_transform = Mat4::from_translation(root_pos)
            * Mat4::from_rotation_y(yaw)
            * Mat4::from_translation(left_pos)
            * Mat4::from_scale(FIST_SCALE);

        let right_transform = Mat4::from_translation(root_pos)
            * Mat4::from_rotation_y(yaw)
            * Mat4::from_translation(right_pos)
            * Mat4::from_scale(FIST_SCALE);

        [
            (body_transform, PLAYER_SCALE),
            (left_transform, FIST_SCALE),
            (right_transform, FIST_SCALE),
        ]
    }

    /// Mesh types for first-person view (fists only).
    pub fn fp_mesh_types() -> [u32; FP_PART_COUNT] {
        [MESH_FIST, MESH_FIST]
    }

    /// Compute fist transforms for first-person view.
    /// Fists are positioned relative to the camera eye, oriented by camera yaw/pitch.
    pub fn compute_fp_transforms(
        &self,
        camera_eye: Vec3,
        yaw: f32,
        pitch: f32,
    ) -> [(Mat4, Vec3); FP_PART_COUNT] {
        // Camera basis vectors.
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let (sin_pitch, cos_pitch) = pitch.sin_cos();
        let forward = Vec3::new(sin_yaw * cos_pitch, -sin_pitch, cos_yaw * cos_pitch);
        let right = Vec3::new(-cos_yaw, 0.0, sin_yaw);
        let up = right.cross(forward);

        // Base offset: lower center of screen, slightly forward.
        let base = camera_eye + forward * 0.5 - up * 0.45;

        let (left_offset_z, left_offset_x) = self.punch_offsets();

        let left_pos = base - right * 0.35 + forward * left_offset_z - right * left_offset_x;
        let right_pos = base + right * 0.35;

        let fp_scale = Vec3::splat(0.20);

        let left_transform = Mat4::from_translation(left_pos)
            * Mat4::from_rotation_y(yaw)
            * Mat4::from_scale(fp_scale);

        let right_transform = Mat4::from_translation(right_pos)
            * Mat4::from_rotation_y(yaw)
            * Mat4::from_scale(fp_scale);

        [
            (left_transform, fp_scale),
            (right_transform, fp_scale),
        ]
    }

    /// Compute (forward_offset, lateral_offset) for the right fist punch.
    /// Counter-clockwise arc: fist sweeps inward (negative X) as it extends forward.
    /// Windup (0.0-0.22): pull back slightly.
    /// Active (0.22-0.55): thrust forward with inward arc.
    /// Recovery (0.55-1.0): return to rest.
    fn punch_offsets(&self) -> (f32, f32) {
        let p = self.attack_progress;
        if p <= 0.0 {
            (0.0, 0.0)
        } else if p < 0.22 {
            // Windup: pull back, slight outward.
            let t = p / 0.22;
            (-0.15 * t, 0.08 * t)
        } else if p < 0.55 {
            // Active: thrust forward, sweep inward (counter-clockwise arc).
            let t = (p - 0.22) / 0.33;
            let forward = -0.15 + (PUNCH_REACH + 0.15) * t;
            let lateral = 0.08 - 0.4 * t; // sweeps from +0.08 to -0.32 (inward)
            (forward, lateral)
        } else {
            // Recovery: return to rest.
            let t = (p - 0.55) / 0.45;
            let forward = PUNCH_REACH * (1.0 - t);
            let lateral = -0.32 * (1.0 - t); // return from inward to center
            (forward, lateral)
        }
    }
}
