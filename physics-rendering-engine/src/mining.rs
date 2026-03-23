use glam::Vec3;

use crate::game::entity::{Entity, EntityId};
use crate::physics::body::{PhysicsBody, RigidBodyHandle};
use crate::physics::world::PhysicsWorld;
use crate::renderer::MESH_ROCK;
use crate::terrain::TerrainGrid;

const CHUNK_HALF: f32 = 0.4;
const CHUNK_SIZE: f32 = CHUNK_HALF * 2.0;
const CHUNK_MAX_HEALTH: u32 = 3;
const STABILITY_RAY_DIST: f32 = 0.5;
const STABILITY_CHECK_RADIUS: f32 = 10.0;

// ---------------------------------------------------------------------------
// Mining chunk — one breakable piece of a mining node
// ---------------------------------------------------------------------------

pub struct MiningChunk {
    pub entity_id: EntityId,
    pub rb: RigidBodyHandle,
    pub health: u32,
    pub node_id: u32,
    /// Grid position within the node (local).
    pub local: (i32, i32, i32),
}

// ---------------------------------------------------------------------------
// Mining system — manages all mining nodes and chunks
// ---------------------------------------------------------------------------

pub struct MiningSystem {
    chunks: Vec<MiningChunk>,
    next_node_id: u32,
}

impl MiningSystem {
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            next_node_id: 0,
        }
    }

    /// Spawn a mining node at `base_pos` (world space) as a grid of chunks.
    /// `size` = (width, height, depth) in chunks.
    /// Returns the entities that were created (caller adds them to the world).
    pub fn spawn_node(
        &mut self,
        physics: &mut PhysicsWorld,
        terrain: &TerrainGrid,
        base_pos: Vec3,
        size: (i32, i32, i32),
        alloc_id: &mut impl FnMut() -> EntityId,
    ) -> Vec<Entity> {
        let node_id = self.next_node_id;
        self.next_node_id += 1;

        let mut entities = Vec::new();
        let half_ext = Vec3::splat(CHUNK_HALF);

        for y in 0..size.1 {
            for z in 0..size.2 {
                for x in 0..size.0 {
                    let offset = Vec3::new(
                        x as f32 * CHUNK_SIZE,
                        y as f32 * CHUNK_SIZE,
                        z as f32 * CHUNK_SIZE,
                    );
                    // Place bottom of the node on the terrain surface.
                    let ground_y = terrain.height_at_world(
                        base_pos.x + offset.x,
                        base_pos.z + offset.z,
                    );
                    let pos = Vec3::new(
                        base_pos.x + offset.x,
                        ground_y + CHUNK_HALF + y as f32 * CHUNK_SIZE,
                        base_pos.z + offset.z,
                    );

                    let entity_id = alloc_id();
                    let body = PhysicsBody::new_static_box(physics, pos, half_ext);

                    self.chunks.push(MiningChunk {
                        entity_id,
                        rb: body.rigid_body,
                        health: CHUNK_MAX_HEALTH,
                        node_id,
                        local: (x, y, z),
                    });

                    let bounding_radius = CHUNK_HALF * 1.73; // sqrt(3)
                    entities.push(Entity::prop(
                        entity_id,
                        body,
                        MESH_ROCK,
                        Vec3::splat(CHUNK_SIZE),
                        bounding_radius,
                    ));
                }
            }
        }

        entities
    }

    /// Apply damage to the chunk whose rigid body matches `rb`.
    /// Returns the entity_id of the destroyed chunk (if health reached 0).
    pub fn damage_chunk(&mut self, rb: RigidBodyHandle, damage: u32) -> Option<EntityId> {
        let idx = self.chunks.iter().position(|c| c.rb == rb)?;
        let chunk = &mut self.chunks[idx];
        chunk.health = chunk.health.saturating_sub(damage);
        if chunk.health == 0 {
            let id = chunk.entity_id;
            self.chunks.swap_remove(idx);
            Some(id)
        } else {
            None
        }
    }

    /// Run stability checks on chunks within `radius` of `impact_pos`.
    /// Returns entity_ids of chunks that should collapse (and removes them).
    pub fn check_stability(
        &mut self,
        physics: &PhysicsWorld,
        impact_pos: Vec3,
    ) -> Vec<EntityId> {
        let mut collapsed = Vec::new();

        // First pass: determine which chunks are in range and whether they are grounded.
        let mut grounded = vec![false; self.chunks.len()];
        for (i, chunk) in self.chunks.iter().enumerate() {
            let pos = physics.body_position(chunk.rb);
            if (pos - impact_pos).length_squared() > STABILITY_CHECK_RADIUS * STABILITY_CHECK_RADIUS
            {
                grounded[i] = true; // out of range, assume stable
                continue;
            }
            // Raycast downward to check ground contact.
            let ray_origin = pos - Vec3::new(0.0, CHUNK_HALF - 0.05, 0.0);
            // Use cast_ray_distance with a dummy exclude (we don't have a collider handle,
            // so just check if there's something below). We can't exclude self easily
            // from a static body, but the ray starts just below the chunk so it should
            // hit terrain or another chunk.
            if let Some(dist) = physics.cast_ray_distance(
                ray_origin,
                Vec3::NEG_Y,
                STABILITY_RAY_DIST,
                rapier3d::prelude::ColliderHandle::invalid(),
            ) {
                if dist < STABILITY_RAY_DIST {
                    grounded[i] = true;
                }
            }
        }

        // Second pass: propagate stability through neighbors (same node, adjacent grid positions).
        let max_iterations = 10;
        for _ in 0..max_iterations {
            let mut changed = false;
            for i in 0..self.chunks.len() {
                if grounded[i] {
                    continue;
                }
                // Check if any neighbor in the same node is grounded.
                let node_id = self.chunks[i].node_id;
                let (lx, ly, lz) = self.chunks[i].local;
                for j in 0..self.chunks.len() {
                    if i == j || !grounded[j] || self.chunks[j].node_id != node_id {
                        continue;
                    }
                    let (nx, ny, nz) = self.chunks[j].local;
                    let dx = (lx - nx).abs();
                    let dy = (ly - ny).abs();
                    let dz = (lz - nz).abs();
                    // Adjacent if exactly one axis differs by 1.
                    if dx + dy + dz == 1 {
                        grounded[i] = true;
                        changed = true;
                        break;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Collect unstable chunks (reverse order to avoid index issues with swap_remove).
        let mut to_remove: Vec<usize> = Vec::new();
        for i in 0..self.chunks.len() {
            if !grounded[i] {
                to_remove.push(i);
            }
        }
        // Sort descending for safe swap_remove.
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_remove {
            collapsed.push(self.chunks[idx].entity_id);
            self.chunks.swap_remove(idx);
        }

        collapsed
    }

    /// Check if a rigid body belongs to a mining chunk.
    pub fn is_mining_chunk(&self, rb: RigidBodyHandle) -> bool {
        self.chunks.iter().any(|c| c.rb == rb)
    }
}
