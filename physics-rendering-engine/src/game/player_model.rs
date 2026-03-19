/// Simple capsule (pill) player model.

use glam::{Mat4, Vec3};

pub const BODY_PART_COUNT: usize = 1;

/// Player scale: a tall pill roughly matching the physics collider.
const PLAYER_SCALE: Vec3 = Vec3::new(0.8, 1.8, 0.8);

pub struct PlayerModel;

impl PlayerModel {
    pub fn new() -> Self {
        Self
    }

    pub fn update(&mut self, _dt: f32, _move_speed: f32) {}

    /// Returns a single transform + scale for the player capsule.
    /// `root_pos` is the physics body center, `yaw` is facing direction.
    pub fn compute_transforms(
        &self,
        root_pos: Vec3,
        yaw: f32,
    ) -> [(Mat4, Vec3); BODY_PART_COUNT] {
        let transform = Mat4::from_translation(root_pos)
            * Mat4::from_rotation_y(yaw)
            * Mat4::from_scale(PLAYER_SCALE);
        [(transform, PLAYER_SCALE)]
    }
}
