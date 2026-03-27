use glam::Vec3;

use crate::physics::body::PhysicsBody;
use crate::physics::world::PhysicsWorld;
use crate::game::entity::{Entity, EntityId};

/// Half-diagonal of a unit cube — conservative bounding sphere for any unit mesh.
pub const UNIT_BOUNDING_RADIUS: f32 = 0.87; // sqrt(3)/2

/// Build the default scene. Returns (entities, player_entity_id, next_entity_id).
pub fn build_scene(physics: &mut PhysicsWorld) -> (Vec<Entity>, EntityId, EntityId) {
    let mut entities: Vec<Entity> = Vec::new();
    let mut next_id: EntityId = 0;
    let mut alloc_id = || { let id = next_id; next_id += 1; id };

    // --- Player ---
    let player_id = alloc_id();
    let player_body = PhysicsBody::new_player_box(physics, Vec3::new(0.0, 0.9, 4.0), Vec3::new(0.4, 0.9, 0.4));
    let player_entity = Entity::player(player_id, player_body);
    entities.push(player_entity);

    (entities, player_id, next_id)
}
