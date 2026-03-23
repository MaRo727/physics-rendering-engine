use glam::Vec3;
use rapier3d::prelude::*;

// Re-export rapier types so code outside physics/ doesn't import rapier3d directly
pub use rapier3d::prelude::{ColliderHandle, RigidBodyHandle, Isometry, SharedShape};

use super::world::PhysicsWorld;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightClass {
    Light,  // stick, small debris
    Medium, // small cube
    Heavy,  // large cube
}

impl WeightClass {
    pub fn mass(self) -> f32 {
        match self {
            WeightClass::Light  => 1.0,
            WeightClass::Medium => 5.0,
            WeightClass::Heavy  => 20.0,
        }
    }

    pub fn punch_knockback(self) -> f32 {
        match self {
            WeightClass::Light  => 12.0,
            WeightClass::Medium => 6.0,
            WeightClass::Heavy  => 2.0,
        }
    }

    pub fn throw_speed(self) -> f32 {
        match self {
            WeightClass::Light  => 25.0,
            WeightClass::Medium => 18.0,
            WeightClass::Heavy  => 10.0,
        }
    }
}

pub struct PhysicsBody {
    pub rigid_body: RigidBodyHandle,
    pub collider: ColliderHandle,
    pub weight_class: WeightClass,
}

impl PhysicsBody {
    /// Create a lightweight copy with just handles and weight class (no new physics objects).
    pub fn clone_handles(&self) -> Self {
        Self {
            rigid_body: self.rigid_body,
            collider: self.collider,
            weight_class: self.weight_class,
        }
    }

    /// Dynamic rigid body with a box collider — falls under gravity.
    pub fn new_dynamic_box(
        world: &mut PhysicsWorld,
        position: Vec3,
        half_extents: Vec3,
        weight_class: WeightClass,
    ) -> Self {
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![position.x, position.y, position.z])
            .additional_mass(weight_class.mass())
            .build();
        let rigid_body = world.rigid_body_set.insert(rigid_body);

        let collider = ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z)
            .restitution(0.4)
            .build();
        let collider =
            world
                .collider_set
                .insert_with_parent(collider, rigid_body, &mut world.rigid_body_set);

        Self { rigid_body, collider, weight_class }
    }

    /// Dynamic rigid body with a capsule collider, all rotations locked — for the player.
    /// Capsule prevents edge-catching on walls and terrain that causes box colliders to stick.
    pub fn new_player_box(world: &mut PhysicsWorld, position: Vec3, half_extents: Vec3) -> Self {
        let radius = half_extents.x; // 0.4
        let half_height = (half_extents.y - radius).max(0.0); // cylinder half-height
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![position.x, position.y, position.z])
            .locked_axes(LockedAxes::ROTATION_LOCKED)
            .build();
        let rigid_body = world.rigid_body_set.insert(rigid_body);

        let collider = ColliderBuilder::capsule_y(half_height, radius)
            .friction(0.0)
            .build();
        let collider =
            world
                .collider_set
                .insert_with_parent(collider, rigid_body, &mut world.rigid_body_set);

        Self { rigid_body, collider, weight_class: WeightClass::Medium }
    }

    /// Dynamic rigid body with a ball (sphere) collider — falls under gravity.
    pub fn new_dynamic_ball(
        world: &mut PhysicsWorld,
        position: Vec3,
        radius: f32,
        weight_class: WeightClass,
    ) -> Self {
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![position.x, position.y, position.z])
            .additional_mass(weight_class.mass())
            .build();
        let rigid_body = world.rigid_body_set.insert(rigid_body);

        let collider = ColliderBuilder::ball(radius)
            .restitution(0.6)
            .build();
        let collider =
            world
                .collider_set
                .insert_with_parent(collider, rigid_body, &mut world.rigid_body_set);

        Self { rigid_body, collider, weight_class }
    }

    /// Dynamic rigid body with a ball collider and locked rotations — for enemies like slimes.
    pub fn new_enemy_ball(
        world: &mut PhysicsWorld,
        position: Vec3,
        radius: f32,
        weight_class: WeightClass,
    ) -> Self {
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![position.x, position.y, position.z])
            .additional_mass(weight_class.mass())
            .locked_axes(LockedAxes::ROTATION_LOCKED)
            .build();
        let rigid_body = world.rigid_body_set.insert(rigid_body);

        let collider = ColliderBuilder::ball(radius)
            .restitution(0.3)
            .build();
        let collider =
            world
                .collider_set
                .insert_with_parent(collider, rigid_body, &mut world.rigid_body_set);

        Self { rigid_body, collider, weight_class }
    }

    /// Dynamic rigid body with a convex hull collider — falls under gravity.
    pub fn new_dynamic_convex(
        world: &mut PhysicsWorld,
        position: Vec3,
        points: &[Vec3],
        weight_class: WeightClass,
    ) -> Self {
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![position.x, position.y, position.z])
            .additional_mass(weight_class.mass())
            .build();
        let rigid_body = world.rigid_body_set.insert(rigid_body);

        let rapier_points: Vec<_> = points
            .iter()
            .map(|p| point![p.x, p.y, p.z])
            .collect();
        let collider = ColliderBuilder::convex_hull(&rapier_points)
            .unwrap_or_else(|| ColliderBuilder::cuboid(0.5, 0.5, 0.5))
            .restitution(0.4)
            .build();
        let collider =
            world
                .collider_set
                .insert_with_parent(collider, rigid_body, &mut world.rigid_body_set);

        Self { rigid_body, collider, weight_class }
    }

    /// Static rigid body with a box collider — never moves.
    pub fn new_static_box(world: &mut PhysicsWorld, position: Vec3, half_extents: Vec3) -> Self {
        let rigid_body = RigidBodyBuilder::fixed()
            .translation(vector![position.x, position.y, position.z])
            .build();
        let rigid_body = world.rigid_body_set.insert(rigid_body);

        let collider = ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z)
            .build();
        let collider =
            world
                .collider_set
                .insert_with_parent(collider, rigid_body, &mut world.rigid_body_set);

        Self { rigid_body, collider, weight_class: WeightClass::Heavy }
    }
}
