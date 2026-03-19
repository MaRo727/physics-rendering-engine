use glam::Vec3;

use crate::physics::body::{PhysicsBody, WeightClass};
use crate::physics::world::PhysicsWorld;
use crate::renderer::{MESH_CUBE, MESH_BALL, MESH_PYRAMID, MESH_TRIANGLE, MESH_SLOPE};
use crate::game::entity::{Entity, EntityId};

/// Half-diagonal of a unit cube — conservative bounding sphere for any unit mesh.
pub const UNIT_BOUNDING_RADIUS: f32 = 0.87; // sqrt(3)/2

/// Build the default scene. Returns (entities, player_entity_id, next_entity_id).
pub fn build_scene(physics: &mut PhysicsWorld) -> (Vec<Entity>, EntityId, EntityId) {
    let mut entities: Vec<Entity> = Vec::new();
    let mut next_id: EntityId = 0;
    let mut alloc_id = || { let id = next_id; next_id += 1; id };

    // --- Cube (medium, 1x1x1) ---
    let cube_id = alloc_id();
    let scale = Vec3::ONE;
    entities.push(Entity::prop(
        cube_id,
        PhysicsBody::new_dynamic_box(physics, Vec3::new(0.0, 4.0, 0.0), Vec3::new(0.5, 0.5, 0.5), WeightClass::Medium),
        MESH_CUBE,
        scale,
        scale.max_element() * UNIT_BOUNDING_RADIUS,
    ));

    // Floor ID reserved (terrain replaces the floor).
    let _floor_id = alloc_id();

    // --- Big cube (heavy, 3x3x3) ---
    let cube2_id = alloc_id();
    let scale = Vec3::splat(3.0);
    entities.push(Entity::prop(
        cube2_id,
        PhysicsBody::new_dynamic_box(physics, Vec3::new(3.0, 8.0, 0.0), Vec3::new(1.5, 1.5, 1.5), WeightClass::Heavy),
        MESH_CUBE,
        scale,
        scale.max_element() * UNIT_BOUNDING_RADIUS,
    ));

    // --- Stick (light, thin box) ---
    let stick_id = alloc_id();
    let scale = Vec3::new(0.12, 0.12, 1.0);
    entities.push(Entity::prop(
        stick_id,
        PhysicsBody::new_dynamic_box(physics, Vec3::new(-2.0, 1.0, 3.0), Vec3::new(0.06, 0.06, 0.5), WeightClass::Light),
        MESH_CUBE,
        scale,
        scale.max_element() * UNIT_BOUNDING_RADIUS,
    ));

    // --- Ball (medium, radius 0.5) ---
    let ball_id = alloc_id();
    let scale = Vec3::ONE;
    entities.push(Entity::prop(
        ball_id,
        PhysicsBody::new_dynamic_ball(physics, Vec3::new(-3.0, 5.0, -2.0), 0.5, WeightClass::Medium),
        MESH_BALL,
        scale,
        scale.max_element() * UNIT_BOUNDING_RADIUS,
    ));

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
    entities.push(Entity::prop(
        pyramid_id,
        PhysicsBody::new_dynamic_convex(physics, Vec3::new(2.0, 6.0, -3.0), &pyramid_points, WeightClass::Medium),
        MESH_PYRAMID,
        scale,
        scale.max_element() * UNIT_BOUNDING_RADIUS,
    ));

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
    entities.push(Entity::prop(
        tri_id,
        PhysicsBody::new_dynamic_convex(physics, Vec3::new(-4.0, 3.0, 1.0), &tri_points, WeightClass::Light),
        MESH_TRIANGLE,
        scale,
        scale.max_element() * UNIT_BOUNDING_RADIUS,
    ));

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
    entities.push(Entity::prop(
        slope_id,
        PhysicsBody::new_dynamic_convex(physics, Vec3::new(5.0, 1.0, 2.0), &slope_points, WeightClass::Heavy),
        MESH_SLOPE,
        scale,
        scale.max_element() * UNIT_BOUNDING_RADIUS,
    ));

    // --- Player ---
    let player_id = alloc_id();
    let player_body = PhysicsBody::new_player_box(physics, Vec3::new(0.0, 0.9, 4.0), Vec3::new(0.4, 0.9, 0.4));
    let player_entity = Entity::player(player_id, player_body);
    entities.push(player_entity);

    (entities, player_id, next_id)
}
