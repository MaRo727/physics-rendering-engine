use glam::{Mat4, Vec3, Vec4};

use crate::renderer::{pack_instance_id, MESH_TREE_OAK, MESH_TREE_PINE, MESH_TREE_DEAD,
    MESH_TREE_OAK_LOD, MESH_TREE_PINE_LOD, MESH_TREE_DEAD_LOD};
use crate::terrain::{Biome, TerrainGrid, TERRAIN_HALF, CHUNKS_PER_SIDE};

const TREE_OBJECT_ID: u32 = 0xFFE0;
/// Full world diagonal: sqrt(3600² + 3600²) ≈ 5091. Render all trees in view.
const TREE_RENDER_DISTANCE: f32 = 5200.0;
/// Distance beyond which LOD meshes are used.
const LOD_DISTANCE: f32 = 400.0;
const LOD_DISTANCE_SQ: f32 = LOD_DISTANCE * LOD_DISTANCE;
/// Bounding sphere radius for frustum culling (covers tallest tree ~8 units * max scale 1.4).
const TREE_BOUNDING_RADIUS: f32 = 12.0;

// Spacing between placement grid points (world units).
const PLACEMENT_STEP: f32 = 25.0;

// Chunk world size for bucketing.
const CHUNK_WORLD_SIZE: f32 = (TERRAIN_HALF * 2) as f32 / CHUNKS_PER_SIDE as f32;

pub struct TreeInstance {
    pub position: Vec3,
    pub mesh_type: u32,
    pub scale: Vec3,
    pub rotation_y: f32,
}

pub struct StructureGrid {
    trees: Vec<TreeInstance>,
    /// Trees bucketed by terrain chunk index (CHUNKS_PER_SIDE * CHUNKS_PER_SIDE buckets).
    chunk_buckets: Vec<Vec<usize>>,
}

/// Simple deterministic hash for seeded placement.
fn placement_hash(seed: u32, gx: i32, gz: i32) -> u32 {
    let mut h = seed;
    h = h.wrapping_mul(1664525).wrapping_add(gx as u32);
    h = h.wrapping_mul(1664525).wrapping_add(gz as u32);
    h ^= h >> 16;
    h = h.wrapping_mul(2654435769);
    h ^= h >> 13;
    h
}

/// Convert hash to f32 in [0, 1).
fn hash_f32(h: u32) -> f32 {
    (h & 0xFFFFFF) as f32 / 16777216.0
}

impl StructureGrid {
    pub fn generate(seed: u32, terrain: &TerrainGrid) -> Self {
        let num_buckets = (CHUNKS_PER_SIDE * CHUNKS_PER_SIDE) as usize;
        let mut trees = Vec::new();
        let mut chunk_buckets = vec![Vec::new(); num_buckets];

        let half = TERRAIN_HALF as f32;
        let steps = ((half * 2.0) / PLACEMENT_STEP) as i32;

        for gz in 0..steps {
            for gx in 0..steps {
                let base_x = -half + gx as f32 * PLACEMENT_STEP;
                let base_z = -half + gz as f32 * PLACEMENT_STEP;

                let h = placement_hash(seed, gx, gz);

                // Jitter position within the cell.
                let jx = base_x + hash_f32(h) * PLACEMENT_STEP;
                let jz = base_z + hash_f32(h.wrapping_mul(2654435761)) * PLACEMENT_STEP;

                let biome = terrain.biome_at_world(jx, jz);
                let height = terrain.height_at_world(jx, jz);

                // Biome-specific placement rules.
                let (density_threshold, mesh_type) = match biome {
                    Biome::Forest => {
                        if height <= 6.0 { continue; } // below water
                        // Mix oak and pine.
                        let mesh = if hash_f32(h.wrapping_add(1)) > 0.4 {
                            MESH_TREE_OAK
                        } else {
                            MESH_TREE_PINE
                        };
                        (0.55, mesh) // ~55% of grid cells get a tree
                    }
                    Biome::Desert => {
                        if height <= 4.0 { continue; }
                        (0.05, MESH_TREE_DEAD) // very sparse
                    }
                    Biome::Mountains => {
                        if height <= 6.0 || height > 30.0 { continue; } // below water or above treeline
                        (0.3, MESH_TREE_PINE) // moderate density, pine only
                    }
                    Biome::Dungeon => continue, // no trees
                };

                // Density check.
                if hash_f32(h.wrapping_add(2)) > density_threshold {
                    continue;
                }

                // Random scale and rotation.
                let scale_factor = 0.8 + hash_f32(h.wrapping_add(3)) * 0.6; // 0.8–1.4
                let rotation_y = hash_f32(h.wrapping_add(4)) * std::f32::consts::TAU;

                let tree_idx = trees.len();
                trees.push(TreeInstance {
                    position: Vec3::new(jx, height, jz),
                    mesh_type,
                    scale: Vec3::splat(scale_factor),
                    rotation_y,
                });

                // Bucket by chunk.
                let cx = ((jx + half) / CHUNK_WORLD_SIZE) as usize;
                let cz = ((jz + half) / CHUNK_WORLD_SIZE) as usize;
                let cx = cx.min(CHUNKS_PER_SIDE as usize - 1);
                let cz = cz.min(CHUNKS_PER_SIDE as usize - 1);
                let bucket = cx * CHUNKS_PER_SIDE as usize + cz;
                chunk_buckets[bucket].push(tree_idx);
            }
        }

        log::info!("Placed {} trees across {} chunks", trees.len(), num_buckets);

        Self { trees, chunk_buckets }
    }

    /// Collect transforms and instance IDs for trees near `player_pos` that pass frustum culling.
    pub fn render_nearby(
        &self,
        player_pos: Vec3,
        frustum: &[Vec4; 6],
        transforms: &mut Vec<Mat4>,
        instance_ids: &mut Vec<u32>,
    ) {
        let half = TERRAIN_HALF as f32;
        let dist_sq = TREE_RENDER_DISTANCE * TREE_RENDER_DISTANCE;

        // Determine which chunk buckets are within range.
        let min_cx = ((player_pos.x - TREE_RENDER_DISTANCE + half) / CHUNK_WORLD_SIZE)
            .floor().max(0.0) as usize;
        let max_cx = ((player_pos.x + TREE_RENDER_DISTANCE + half) / CHUNK_WORLD_SIZE)
            .ceil().min(CHUNKS_PER_SIDE as f32) as usize;
        let min_cz = ((player_pos.z - TREE_RENDER_DISTANCE + half) / CHUNK_WORLD_SIZE)
            .floor().max(0.0) as usize;
        let max_cz = ((player_pos.z + TREE_RENDER_DISTANCE + half) / CHUNK_WORLD_SIZE)
            .ceil().min(CHUNKS_PER_SIDE as f32) as usize;

        for cx in min_cx..max_cx {
            for cz in min_cz..max_cz {
                let bucket = cx * CHUNKS_PER_SIDE as usize + cz;
                for &tree_idx in &self.chunk_buckets[bucket] {
                    let tree = &self.trees[tree_idx];
                    let dx = tree.position.x - player_pos.x;
                    let dz = tree.position.z - player_pos.z;
                    if dx * dx + dz * dz > dist_sq {
                        continue;
                    }

                    let d_sq = dx * dx + dz * dz;

                    // Frustum cull using a bounding sphere at the tree's mid-height.
                    let center = tree.position + Vec3::new(0.0, 4.0 * tree.scale.y, 0.0);
                    let radius = TREE_BOUNDING_RADIUS * tree.scale.x;
                    if !sphere_in_frustum(frustum, center, radius) {
                        continue;
                    }

                    // Pick full or LOD mesh based on distance.
                    let mesh = if d_sq > LOD_DISTANCE_SQ {
                        lod_mesh(tree.mesh_type)
                    } else {
                        tree.mesh_type
                    };

                    let transform = Mat4::from_translation(tree.position)
                        * Mat4::from_rotation_y(tree.rotation_y)
                        * Mat4::from_scale(tree.scale);
                    transforms.push(transform);
                    instance_ids.push(pack_instance_id(mesh, TREE_OBJECT_ID));
                }
            }
        }
    }

    /// Collect all tree trunk positions and half-extents for physics colliders.
    /// Returns groups by chunk bucket: Vec<(chunk_idx, Vec<(position, half_extents)>)>.
    pub fn trunk_colliders(&self) -> Vec<(usize, Vec<(Vec3, Vec3)>)> {
        let mut result = Vec::new();
        for (bucket_idx, bucket) in self.chunk_buckets.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let mut trunks = Vec::new();
            for &tree_idx in bucket {
                let tree = &self.trees[tree_idx];
                let s = tree.scale.x; // uniform scale
                // Trunk collider: thin box at base of tree
                let half_w = 0.3 * s;
                let half_h = match tree.mesh_type {
                    MESH_TREE_OAK => 1.5 * s,
                    MESH_TREE_PINE => 2.5 * s,
                    MESH_TREE_DEAD => 2.0 * s,
                    _ => 1.5 * s,
                };
                let trunk_center = tree.position + Vec3::new(0.0, half_h, 0.0);
                trunks.push((trunk_center, Vec3::new(half_w, half_h, half_w)));
            }
            result.push((bucket_idx, trunks));
        }
        result
    }
}

/// Map a full-detail tree mesh type to its LOD variant.
fn lod_mesh(mesh_type: u32) -> u32 {
    match mesh_type {
        MESH_TREE_OAK => MESH_TREE_OAK_LOD,
        MESH_TREE_PINE => MESH_TREE_PINE_LOD,
        MESH_TREE_DEAD => MESH_TREE_DEAD_LOD,
        other => other,
    }
}

/// Test whether a bounding sphere is inside (or intersects) all 6 frustum planes.
fn sphere_in_frustum(planes: &[Vec4; 6], center: Vec3, radius: f32) -> bool {
    for plane in planes {
        let dist = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
        if dist < -radius {
            return false;
        }
    }
    true
}
