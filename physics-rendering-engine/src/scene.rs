use glam::Vec3;

use crate::physics::body::{PhysicsBody, WeightClass};
use crate::physics::world::PhysicsWorld;
use crate::renderer::{MESH_CUBE, MESH_BALL, MESH_PYRAMID, MESH_TRIANGLE, MESH_SLOPE};

/// Half-diagonal of a unit cube — conservative bounding sphere for any unit mesh.
pub const UNIT_BOUNDING_RADIUS: f32 = 0.87; // sqrt(3)/2

/// A world object with physics, a mesh type, a render scale, and an object id.
pub struct WorldObject {
    pub body: PhysicsBody,
    pub mesh_type: u32,
    pub render_scale: Vec3,
    pub object_id: u32,
    pub bounding_radius: f32,
}

/// Build the default scene. Returns (objects, player, player_object_id, next_object_id).
pub fn build_scene(physics: &mut PhysicsWorld) -> (Vec<WorldObject>, PhysicsBody, u32, u32) {
    let mut objects: Vec<WorldObject> = Vec::new();
    let mut next_id: u32 = 0;
    let mut alloc_id = || { let id = next_id; next_id += 1; id };

    // --- Cube (medium, 1x1x1) ---
    let cube_id = alloc_id();
    let scale = Vec3::ONE;
    objects.push(WorldObject {
        body: PhysicsBody::new_dynamic_box(
            physics,
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            WeightClass::Medium,
        ),
        mesh_type: MESH_CUBE,
        render_scale: scale,
        object_id: cube_id,
        bounding_radius: scale.max_element() * UNIT_BOUNDING_RADIUS,
    });

    // Floor ID reserved (terrain replaces the floor).
    let _floor_id = alloc_id();

    // --- Big cube (heavy, 3x3x3) ---
    let cube2_id = alloc_id();
    let scale = Vec3::splat(3.0);
    objects.push(WorldObject {
        body: PhysicsBody::new_dynamic_box(
            physics,
            Vec3::new(3.0, 8.0, 0.0),
            Vec3::new(1.5, 1.5, 1.5),
            WeightClass::Heavy,
        ),
        mesh_type: MESH_CUBE,
        render_scale: scale,
        object_id: cube2_id,
        bounding_radius: scale.max_element() * UNIT_BOUNDING_RADIUS,
    });

    // --- Stick (light, thin box) ---
    let stick_id = alloc_id();
    let scale = Vec3::new(0.12, 0.12, 1.0);
    objects.push(WorldObject {
        body: PhysicsBody::new_dynamic_box(
            physics,
            Vec3::new(-2.0, 1.0, 3.0),
            Vec3::new(0.06, 0.06, 0.5),
            WeightClass::Light,
        ),
        mesh_type: MESH_CUBE,
        render_scale: scale,
        object_id: stick_id,
        bounding_radius: scale.max_element() * UNIT_BOUNDING_RADIUS,
    });

    // --- Ball (medium, radius 0.5) ---
    let ball_id = alloc_id();
    let scale = Vec3::ONE;
    objects.push(WorldObject {
        body: PhysicsBody::new_dynamic_ball(
            physics,
            Vec3::new(-3.0, 5.0, -2.0),
            0.5,
            WeightClass::Medium,
        ),
        mesh_type: MESH_BALL,
        render_scale: scale,
        object_id: ball_id,
        bounding_radius: scale.max_element() * UNIT_BOUNDING_RADIUS,
    });

    // --- Pyramid (medium) ---
    let pyramid_id = alloc_id();
    let scale = Vec3::ONE;
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
            physics,
            Vec3::new(2.0, 6.0, -3.0),
            &pyramid_points,
            WeightClass::Medium,
        ),
        mesh_type: MESH_PYRAMID,
        render_scale: scale,
        object_id: pyramid_id,
        bounding_radius: scale.max_element() * UNIT_BOUNDING_RADIUS,
    });

    // --- Triangle prism (light) ---
    let tri_id = alloc_id();
    let scale = Vec3::ONE;
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
            physics,
            Vec3::new(-4.0, 3.0, 1.0),
            &tri_points,
            WeightClass::Light,
        ),
        mesh_type: MESH_TRIANGLE,
        render_scale: scale,
        object_id: tri_id,
        bounding_radius: scale.max_element() * UNIT_BOUNDING_RADIUS,
    });

    // --- Slope / ramp (heavy) ---
    let slope_id = alloc_id();
    let scale = Vec3::splat(2.0);
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
            physics,
            Vec3::new(5.0, 1.0, 2.0),
            &slope_points,
            WeightClass::Heavy,
        ),
        mesh_type: MESH_SLOPE,
        render_scale: scale,
        object_id: slope_id,
        bounding_radius: scale.max_element() * UNIT_BOUNDING_RADIUS,
    });

    // --- Player ---
    let player_id = alloc_id();
    let player = PhysicsBody::new_player_box(
        physics,
        Vec3::new(0.0, 0.9, 4.0),
        Vec3::new(0.4, 0.9, 0.4),
    );

    (objects, player, player_id, next_id)
}
