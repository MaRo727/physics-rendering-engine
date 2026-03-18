use glam::Vec3;
use rapier3d::prelude::*;

use super::world::PhysicsWorld;

pub struct PhysicsBody {
    pub rigid_body: RigidBodyHandle,
    pub collider: ColliderHandle,
}

impl PhysicsBody {
    /// Dynamic rigid body with a box collider — falls under gravity.
    pub fn new_dynamic_box(world: &mut PhysicsWorld, position: Vec3, half_extents: Vec3) -> Self {
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(vector![position.x, position.y, position.z])
            .build();
        let rigid_body = world.rigid_body_set.insert(rigid_body);

        let collider = ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z)
            .restitution(0.4)
            .build();
        let collider =
            world
                .collider_set
                .insert_with_parent(collider, rigid_body, &mut world.rigid_body_set);

        Self { rigid_body, collider }
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

        Self { rigid_body, collider }
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

        Self { rigid_body, collider }
    }
}
