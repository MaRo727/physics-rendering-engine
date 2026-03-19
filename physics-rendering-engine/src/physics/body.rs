use glam::Vec3;
use rapier3d::prelude::*;

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

    /// Dynamic rigid body with a box collider, all rotations locked — for the player.
    pub fn new_player_box(world: &mut PhysicsWorld, position: Vec3, half_extents: Vec3) -> Self {
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![position.x, position.y, position.z])
            .locked_axes(LockedAxes::ROTATION_LOCKED)
            .build();
        let rigid_body = world.rigid_body_set.insert(rigid_body);

        let collider = ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z)
            .build();
        let collider =
            world
                .collider_set
                .insert_with_parent(collider, rigid_body, &mut world.rigid_body_set);

        Self { rigid_body, collider, weight_class: WeightClass::Medium }
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
