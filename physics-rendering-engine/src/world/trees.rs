use glam::{Mat4, Vec3, Vec4};

use crate::renderer::{pack_instance_id, MESH_TREE_OAK, MESH_TREE_PINE, MESH_TREE_DEAD,
    MESH_TREE_OAK_LOD, MESH_TREE_PINE_LOD, MESH_TREE_DEAD_LOD,
    MESH_CACTUS, MESH_CACTUS_SMALL, MESH_CACTUS_LOD, MESH_CACTUS_SMALL_LOD,
    MESH_LEAF_PARTICLE, MESH_BARK_CHIP,
};
use crate::renderer::mesh::Vertex;
use crate::renderer::shapes;
use crate::world::terrain::{Biome, TerrainGrid, TERRAIN_HALF, CHUNKS_PER_SIDE};

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
    #[allow(dead_code)]
    pub rotation_y: f32,
    /// Precomputed wind-phase trig values for the four sway frequencies.
    /// `phase = position.x * 0.13 + position.z * 0.17`.
    /// Stores `[(sin(phase), cos(phase)), (sin(1.3*phase), cos(1.3*phase)),
    ///          (sin(0.7*phase), cos(0.7*phase)), (sin(1.1*phase), cos(1.1*phase))]`.
    /// Eliminates 8 trig calls per visible tree each frame.
    phase_sc: [(f32, f32); 4],
    /// Pre-computed static transform: translation * rotation_y (without scale).
    /// Scale is applied separately so sway/shake are inserted between rotation and scale.
    base_transform_no_scale: Mat4,
}

/// A leaf particle spawned when a tree is punched.
pub struct LeafParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub lifetime: f32,
    pub mesh_type: u32,
    pub scale: f32,
    pub rotation_y: f32,
}

/// Active tree shake: (tree index, remaining time).
struct TreeShake {
    tree_idx: usize,
    timer: f32,
}

const SHAKE_DURATION: f32 = 0.5;
const SHAKE_AMPLITUDE: f32 = 0.06; // radians
const SHAKE_FREQUENCY: f32 = 25.0; // Hz

const LEAF_COUNT: usize = 10;
const LEAF_LIFETIME: f32 = 2.0;

pub struct StructureGrid {
    trees: Vec<TreeInstance>,
    /// Trees bucketed by terrain chunk index (CHUNKS_PER_SIDE * CHUNKS_PER_SIDE buckets).
    chunk_buckets: Vec<Vec<usize>>,
    /// Maps chunk index → absolute mesh type ID for merged LOD trees. None = no trees in chunk.
    chunk_mesh_ids: Vec<Option<u32>>,
    /// Currently shaking trees.
    shakes: Vec<TreeShake>,
    /// Active leaf particles from tree punches.
    pub leaf_particles: Vec<LeafParticle>,
}

/// Simple deterministic hash for seeded placement.
pub(crate) fn placement_hash(seed: u32, gx: i32, gz: i32) -> u32 {
    let mut h = seed;
    h = h.wrapping_mul(1664525).wrapping_add(gx as u32);
    h = h.wrapping_mul(1664525).wrapping_add(gz as u32);
    h ^= h >> 16;
    h = h.wrapping_mul(2654435769);
    h ^= h >> 13;
    h
}

/// Convert hash to f32 in [0, 1).
pub(crate) fn hash_f32(h: u32) -> f32 {
    (h & 0xFFFFFF) as f32 / 16777216.0
}

/// Test whether a bounding sphere is inside (or intersects) all 6 frustum planes.
pub(crate) fn sphere_in_frustum(planes: &[Vec4; 6], center: Vec3, radius: f32) -> bool {
    for plane in planes {
        let dist = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
        if dist < -radius {
            return false;
        }
    }
    true
}

impl StructureGrid {
    pub fn generate(seed: u32, terrain: &TerrainGrid) -> Self {
        let num_buckets = (CHUNKS_PER_SIDE * CHUNKS_PER_SIDE) as usize;
        let mut trees = Vec::new();
        let mut chunk_buckets = vec![Vec::new(); num_buckets];

        let half = TERRAIN_HALF as f32;
        let steps = ((half * 2.0) / PLACEMENT_STEP) as i32;

        // Multiple placement passes: pass 0 is base grid, passes 1-2 are
        // offset sub-grids that only place trees in forest biome (3x density).
        for pass in 0u32..3 {
            let pass_seed = seed.wrapping_add(pass * 7919);
            // Offset sub-grids by 1/3 and 2/3 of step to avoid overlap.
            let offset_x = pass as f32 * PLACEMENT_STEP / 3.0;
            let offset_z = pass as f32 * PLACEMENT_STEP * 0.41; // non-aligned offset

            for gz in 0..steps {
                for gx in 0..steps {
                    let base_x = -half + gx as f32 * PLACEMENT_STEP + offset_x;
                    let base_z = -half + gz as f32 * PLACEMENT_STEP + offset_z;

                    let h = placement_hash(pass_seed, gx, gz);

                    // Jitter position within the cell.
                    let jx = base_x + hash_f32(h) * PLACEMENT_STEP;
                    let jz = base_z + hash_f32(h.wrapping_mul(2654435761)) * PLACEMENT_STEP;

                    let biome = terrain.biome_at_world(jx, jz);
                    let height = terrain.height_at_world(jx, jz);

                    // Biome-specific placement rules.
                    let (density_threshold, mesh_type) = match biome {
                        Biome::Plains => {
                            if pass > 0 { continue; } // only base pass for plains
                            if height <= 6.0 { continue; }
                            (0.12, MESH_TREE_OAK)
                        }
                        Biome::Forest => {
                            if height <= 6.0 { continue; }
                            let mesh = if hash_f32(h.wrapping_add(1)) > 0.35 {
                                MESH_TREE_OAK
                            } else {
                                MESH_TREE_PINE
                            };
                            (0.75, mesh) // all 3 passes place in forest
                        }
                        Biome::Desert => {
                            if pass > 0 { continue; }
                            if height <= 4.0 { continue; }
                            let r = hash_f32(h.wrapping_add(1));
                            let mesh = if r < 0.50 {
                                MESH_CACTUS
                            } else if r < 0.80 {
                                MESH_CACTUS_SMALL
                            } else {
                                MESH_TREE_DEAD
                            };
                            (0.12, mesh)
                        }
                        Biome::Mountains => {
                            if pass > 0 { continue; }
                            if height <= 6.0 || height > 30.0 { continue; }
                            (0.3, MESH_TREE_PINE)
                        }
                        Biome::Dungeon => continue,
                        Biome::Crystal => continue,
                    };

                    // Density check.
                    if hash_f32(h.wrapping_add(2)) > density_threshold {
                        continue;
                    }

                    // Random scale and rotation.
                    let scale_factor = 0.8 + hash_f32(h.wrapping_add(3)) * 0.6;
                    let rotation_y = hash_f32(h.wrapping_add(4)) * std::f32::consts::TAU;

                    let pos = Vec3::new(jx, height, jz);
                    let base_transform_no_scale = Mat4::from_translation(pos)
                        * Mat4::from_rotation_y(rotation_y);
                    let tree_idx = trees.len();
                    let phase = jx * 0.13 + jz * 0.17;
                    let sc0 = phase.sin_cos();
                    let sc1 = (phase * 1.3).sin_cos();
                    let sc2 = (phase * 0.7).sin_cos();
                    let sc3 = (phase * 1.1).sin_cos();
                    trees.push(TreeInstance {
                        position: pos,
                        mesh_type,
                        scale: Vec3::splat(scale_factor),
                        rotation_y,
                        phase_sc: [sc0, sc1, sc2, sc3],
                        base_transform_no_scale,
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
        }

        log::info!("Placed {} trees across {} chunks", trees.len(), num_buckets);

        Self {
            trees,
            chunk_buckets,
            chunk_mesh_ids: vec![None; num_buckets],
            shakes: Vec::new(),
            leaf_particles: Vec::new(),
        }
    }

    /// Build a single pre-merged LOD mesh per chunk (all tree LOD geometry baked with
    /// world-space transforms). Returns mesh data and a chunk→local-index mapping.
    pub fn build_chunk_meshes(&self) -> (Vec<(Vec<Vertex>, Vec<u32>)>, Vec<Option<usize>>) {
        // Generate LOD mesh data once for each tree type.
        let oak_lod = shapes::tree_oak_lod();
        let pine_lod = shapes::tree_pine_lod();
        let dead_lod = shapes::tree_dead_lod();
        let cactus_lod = shapes::cactus_lod();
        let cactus_small_lod = shapes::cactus_small_lod();

        let mut meshes = Vec::new();
        let mut chunk_map: Vec<Option<usize>> = vec![None; self.chunk_buckets.len()];

        for (chunk_idx, bucket) in self.chunk_buckets.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }

            let mut merged_verts = Vec::new();
            let mut merged_indices = Vec::new();

            for &tree_idx in bucket {
                let tree = &self.trees[tree_idx];

                let (src_verts, src_indices) = match tree.mesh_type {
                    MESH_TREE_OAK => &oak_lod,
                    MESH_TREE_PINE => &pine_lod,
                    MESH_TREE_DEAD => &dead_lod,
                    MESH_CACTUS => &cactus_lod,
                    MESH_CACTUS_SMALL => &cactus_small_lod,
                    _ => continue,
                };

                // Bake the tree's static transform into the vertices.
                let transform = tree.base_transform_no_scale * Mat4::from_scale(tree.scale);

                let base_idx = merged_verts.len() as u32;
                for v in src_verts {
                    let pos = transform.transform_point3(v.position);
                    // Uniform scale → transform_vector3 + normalize gives correct normals.
                    let norm = transform.transform_vector3(v.normal).normalize();
                    merged_verts.push(Vertex { position: pos, normal: norm, color: v.color });
                }
                for &idx in src_indices {
                    merged_indices.push(base_idx + idx);
                }
            }

            if !merged_verts.is_empty() {
                chunk_map[chunk_idx] = Some(meshes.len());
                meshes.push((merged_verts, merged_indices));
            }
        }

        log::info!("Built {} merged tree-chunk meshes", meshes.len());
        (meshes, chunk_map)
    }

    /// Assign absolute mesh type IDs for the merged chunk meshes.
    pub fn set_chunk_mesh_ids(&mut self, chunk_map: Vec<Option<usize>>, first_mesh_id: u32) {
        self.chunk_mesh_ids = chunk_map.iter().map(|opt| {
            opt.map(|local_idx| first_mesh_id + local_idx as u32)
        }).collect();
    }

    /// Collect transforms and instance IDs for trees near `player_pos` that pass frustum culling.
    /// `wind_strength` (0..1), `wind_dir` is normalized (x, z), `time` is weather animation time.
    pub fn render_nearby(
        &self,
        player_pos: Vec3,
        frustum: &[Vec4; 6],
        wind_strength: f32,
        wind_dir: (f32, f32),
        time: f32,
        draw_distance: f32,
        transforms: &mut Vec<Mat4>,
        instance_ids: &mut Vec<u32>,
    ) {
        let half = TERRAIN_HALF as f32;
        let dist_sq = draw_distance * draw_distance;

        // Determine which chunk buckets are within range.
        let min_cx = ((player_pos.x - draw_distance + half) / CHUNK_WORLD_SIZE)
            .floor().max(0.0) as usize;
        let max_cx = ((player_pos.x + draw_distance + half) / CHUNK_WORLD_SIZE)
            .ceil().min(CHUNKS_PER_SIDE as f32) as usize;
        let min_cz = ((player_pos.z - draw_distance + half) / CHUNK_WORLD_SIZE)
            .floor().max(0.0) as usize;
        let max_cz = ((player_pos.z + draw_distance + half) / CHUNK_WORLD_SIZE)
            .ceil().min(CHUNKS_PER_SIDE as f32) as usize;

        // Precompute shared wind trig values used across all trees.
        // These four frequencies cover the multi-frequency sway; trees add per-tree
        // variation via a cheap linear phase offset rather than extra sin() calls.
        let sway_amount = wind_strength * 0.04; // max ~2.3 degrees in full wind
        let wind_bias = wind_strength * 0.015;
        let base_sin_1_2 = (time * 1.2).sin();
        let base_cos_1_2 = (time * 1.2).cos();
        let base_sin_2_7 = (time * 2.7).sin();
        let base_cos_2_7 = (time * 2.7).cos();
        let base_sin_0_9 = (time * 0.9).sin();
        let base_cos_0_9 = (time * 0.9).cos();
        let base_sin_2_1 = (time * 2.1).sin();
        let base_cos_2_1 = (time * 2.1).cos();

        // Bounding sphere covers half-diagonal of the chunk + tallest tree.
        let chunk_cull_radius: f32 = CHUNK_WORLD_SIZE * 0.75 + TREE_BOUNDING_RADIUS;

        for cx in min_cx..max_cx {
            for cz in min_cz..max_cz {
                // Chunk-level frustum cull: skip entire bucket if chunk sphere is outside frustum.
                let chunk_center_x = (cx as f32 + 0.5) * CHUNK_WORLD_SIZE - half;
                let chunk_center_z = (cz as f32 + 0.5) * CHUNK_WORLD_SIZE - half;
                let chunk_center = Vec3::new(chunk_center_x, 15.0, chunk_center_z);
                if !sphere_in_frustum(frustum, chunk_center, chunk_cull_radius) {
                    continue;
                }
                let bucket = cx * CHUNKS_PER_SIDE as usize + cz;

                // Check if entire chunk is beyond LOD_DISTANCE — use merged mesh
                // (single TLAS instance per chunk instead of one per tree).
                let chunk_min_x = cx as f32 * CHUNK_WORLD_SIZE - half;
                let chunk_max_x = (cx as f32 + 1.0) * CHUNK_WORLD_SIZE - half;
                let chunk_min_z = cz as f32 * CHUNK_WORLD_SIZE - half;
                let chunk_max_z = (cz as f32 + 1.0) * CHUNK_WORLD_SIZE - half;
                let nearest_x = player_pos.x.clamp(chunk_min_x, chunk_max_x);
                let nearest_z = player_pos.z.clamp(chunk_min_z, chunk_max_z);
                let near_dx = player_pos.x - nearest_x;
                let near_dz = player_pos.z - nearest_z;
                let chunk_near_dist_sq = near_dx * near_dx + near_dz * near_dz;

                if chunk_near_dist_sq > LOD_DISTANCE_SQ {
                    if let Some(mesh_id) = self.chunk_mesh_ids[bucket] {
                        transforms.push(Mat4::IDENTITY);
                        instance_ids.push(pack_instance_id(mesh_id, TREE_OBJECT_ID));
                    }
                    continue;
                }

                // Near chunk — render individual trees with sway animation.
                for &tree_idx in &self.chunk_buckets[bucket] {
                    let tree = &self.trees[tree_idx];
                    let dx = tree.position.x - player_pos.x;
                    let dz = tree.position.z - player_pos.z;
                    let d_sq = dx * dx + dz * dz;
                    if d_sq > dist_sq {
                        continue;
                    }

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

                    let shake = self.shake_angle(tree_idx);

                    // Wind sway: use angle-addition identity sin(A+B) = sinA cosB + cosA sinB
                    // to derive per-tree sway from precomputed base trig values + precomputed per-tree phase.
                    let [(sp, cp), (sp13, cp13), (sp07, cp07), (sp11, cp11)] = tree.phase_sc;

                    // sin(time*f + phase*k) = base_sin_f * cos(phase*k) + base_cos_f * sin(phase*k)
                    let sway_x = (base_sin_1_2 * cp + base_cos_1_2 * sp) * sway_amount
                        + (base_sin_2_7 * cp13 + base_cos_2_7 * sp13) * sway_amount * 0.3;
                    let sway_z = (base_sin_0_9 * cp07 + base_cos_0_9 * sp07) * sway_amount * 0.6
                        + (base_sin_2_1 * cp11 + base_cos_2_1 * sp11) * sway_amount * 0.2;

                    let total_sway_x = sway_x + wind_dir.0 * wind_bias;
                    let total_sway_z = sway_z + wind_dir.1 * wind_bias;

                    // Start from pre-computed base (translation * rotation_y).
                    // Apply sway and shake dynamically, then scale last.
                    let has_sway = total_sway_x.abs() > 0.0001 || total_sway_z.abs() > 0.0001;
                    let transform = if has_sway || shake != 0.0 {
                        let mut t = tree.base_transform_no_scale;
                        if has_sway {
                            let a = total_sway_z; // rotation about X
                            let b = total_sway_x; // rotation about Z
                            // Small-angle combined Rx(a)*Rz(b) matrix.
                            let sway_mat = Mat4::from_cols(
                                Vec4::new(1.0, b, 0.0, 0.0),
                                Vec4::new(-b, 1.0, a, 0.0),
                                Vec4::new(a * b, -a, 1.0, 0.0),
                                Vec4::new(0.0, 0.0, 0.0, 1.0),
                            );
                            t = t * sway_mat;
                        }
                        if shake != 0.0 {
                            t = t * Mat4::from_rotation_z(shake);
                        }
                        t * Mat4::from_scale(tree.scale)
                    } else {
                        tree.base_transform_no_scale * Mat4::from_scale(tree.scale)
                    };
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

    /// Find the nearest tree to a world position and trigger a shake + leaf particles.
    /// `seed` is used for randomizing leaf directions.
    pub fn punch_tree_at(&mut self, hit_pos: Vec3, seed: u32) {
        // Find nearest tree within 3 units of hit position using the spatial index.
        const PUNCH_RADIUS: f32 = 3.0;
        let half = TERRAIN_HALF as f32;
        let min_cx = ((hit_pos.x - PUNCH_RADIUS + half) / CHUNK_WORLD_SIZE)
            .floor().max(0.0) as usize;
        let max_cx = ((hit_pos.x + PUNCH_RADIUS + half) / CHUNK_WORLD_SIZE)
            .ceil().min(CHUNKS_PER_SIDE as f32) as usize;
        let min_cz = ((hit_pos.z - PUNCH_RADIUS + half) / CHUNK_WORLD_SIZE)
            .floor().max(0.0) as usize;
        let max_cz = ((hit_pos.z + PUNCH_RADIUS + half) / CHUNK_WORLD_SIZE)
            .ceil().min(CHUNKS_PER_SIDE as f32) as usize;

        let mut best: Option<(usize, f32)> = None;
        for cx in min_cx..max_cx {
            for cz in min_cz..max_cz {
                let bucket = cx * CHUNKS_PER_SIDE as usize + cz;
                for &tree_idx in &self.chunk_buckets[bucket] {
                    let tree = &self.trees[tree_idx];
                    let dx = tree.position.x - hit_pos.x;
                    let dz = tree.position.z - hit_pos.z;
                    let dist_sq = dx * dx + dz * dz;
                    if dist_sq < PUNCH_RADIUS * PUNCH_RADIUS {
                        if best.map_or(true, |(_, d)| dist_sq < d) {
                            best = Some((tree_idx, dist_sq));
                        }
                    }
                }
            }
        }

        let tree_idx = match best {
            Some((idx, _)) => idx,
            None => return,
        };

        // Don't stack shakes on the same tree.
        if self.shakes.iter().any(|s| s.tree_idx == tree_idx) {
            return;
        }

        self.shakes.push(TreeShake { tree_idx, timer: SHAKE_DURATION });

        // Spawn particles at the canopy (trees) or body (cacti).
        let tree = &self.trees[tree_idx];
        let (particle_height, particle_radius) = match tree.mesh_type {
            MESH_TREE_OAK => (5.0 * tree.scale.y, 3.0 * tree.scale.x),
            MESH_TREE_PINE => (6.0 * tree.scale.y, 3.0 * tree.scale.x),
            MESH_TREE_DEAD => (3.0 * tree.scale.y, 2.0 * tree.scale.x),
            MESH_CACTUS => (2.0 * tree.scale.y, 0.8 * tree.scale.x),
            MESH_CACTUS_SMALL => (1.2 * tree.scale.y, 0.5 * tree.scale.x),
            _ => (5.0 * tree.scale.y, 3.0 * tree.scale.x),
        };
        let canopy_center = tree.position + Vec3::new(0.0, particle_height, 0.0);
        let canopy_radius = particle_radius;

        let mut h = seed;
        for _ in 0..LEAF_COUNT {
            h = h.wrapping_mul(1664525).wrapping_add(1013904223);
            let angle = hash_f32(h) * std::f32::consts::TAU;
            h = h.wrapping_mul(1664525).wrapping_add(1013904223);
            let r = hash_f32(h) * canopy_radius;
            h = h.wrapping_mul(1664525).wrapping_add(1013904223);
            let vy = -0.5 - hash_f32(h) * 1.5; // falling speed
            h = h.wrapping_mul(1664525).wrapping_add(1013904223);
            let vx = (hash_f32(h) - 0.5) * 2.0;
            h = h.wrapping_mul(1664525).wrapping_add(1013904223);
            let vz = (hash_f32(h) - 0.5) * 2.0;
            h = h.wrapping_mul(1664525).wrapping_add(1013904223);
            let rot = hash_f32(h) * std::f32::consts::TAU;

            let pos = canopy_center + Vec3::new(angle.cos() * r, (hash_f32(h) - 0.5) * 1.0, angle.sin() * r);

            let mesh = match tree.mesh_type {
                MESH_TREE_DEAD | MESH_CACTUS | MESH_CACTUS_SMALL => MESH_BARK_CHIP,
                _ => MESH_LEAF_PARTICLE,
            };

            self.leaf_particles.push(LeafParticle {
                position: pos,
                velocity: Vec3::new(vx, vy, vz),
                lifetime: LEAF_LIFETIME,
                mesh_type: mesh,
                scale: 0.15 + hash_f32(h.wrapping_add(1)) * 0.1,
                rotation_y: rot,
            });
        }
    }

    /// Update shake timers and leaf particles. Call once per frame.
    /// `wind_strength` (0..1), `wind_dir` is (x, z) normalized direction.
    pub fn update_effects(&mut self, dt: f32, wind_strength: f32, wind_dir: (f32, f32)) {
        // Update shakes.
        self.shakes.retain_mut(|s| {
            s.timer -= dt;
            s.timer > 0.0
        });

        // Wind force applied to leaf particles.
        let wind_force_x = wind_dir.0 * wind_strength * 8.0;
        let wind_force_z = wind_dir.1 * wind_strength * 8.0;

        // Update leaf particles with gravity, wind drift, and air drag.
        self.leaf_particles.retain_mut(|p| {
            p.lifetime -= dt;
            if p.lifetime <= 0.0 {
                return false;
            }
            p.velocity.y -= 2.0 * dt; // gravity
            // Wind pushes leaves.
            p.velocity.x += wind_force_x * dt;
            p.velocity.z += wind_force_z * dt;
            p.velocity.x *= 1.0 - 0.5 * dt; // air drag
            p.velocity.z *= 1.0 - 0.5 * dt;
            p.position += p.velocity * dt;
            p.rotation_y += (3.0 + wind_strength * 5.0) * dt; // spin faster in wind
            true
        });
    }

    /// Spawn wind-blown leaves from trees near the player.
    /// Call periodically (not every frame) when wind is strong enough.
    pub fn emit_wind_leaves(
        &mut self,
        player_pos: Vec3,
        wind_strength: f32,
        wind_dir: (f32, f32),
        seed: &mut u32,
    ) {
        // Only emit when wind is moderate or above.
        if wind_strength < 0.2 {
            return;
        }
        // Cap leaf particles to avoid flooding.
        if self.leaf_particles.len() > 400 {
            return;
        }

        let half = TERRAIN_HALF as f32;
        let emit_range = 80.0; // only from nearby trees
        let emit_range_sq = emit_range * emit_range;

        // Number of leaves scales with wind strength.
        let max_leaves = (wind_strength * 6.0) as usize;

        let min_cx = ((player_pos.x - emit_range + half) / CHUNK_WORLD_SIZE)
            .floor().max(0.0) as usize;
        let max_cx = ((player_pos.x + emit_range + half) / CHUNK_WORLD_SIZE)
            .ceil().min(CHUNKS_PER_SIDE as f32) as usize;
        let min_cz = ((player_pos.z - emit_range + half) / CHUNK_WORLD_SIZE)
            .floor().max(0.0) as usize;
        let max_cz = ((player_pos.z + emit_range + half) / CHUNK_WORLD_SIZE)
            .ceil().min(CHUNKS_PER_SIDE as f32) as usize;

        let mut h = *seed;
        let mut emitted = 0usize;
        for cx in min_cx..max_cx {
            for cz in min_cz..max_cz {
                let bucket = cx * CHUNKS_PER_SIDE as usize + cz;
                for &tree_idx in &self.chunk_buckets[bucket] {
                    if emitted >= max_leaves {
                        *seed = h;
                        return;
                    }
                    let tree = &self.trees[tree_idx];
                    // Skip dead trees (no foliage to blow off).
                    if tree.mesh_type == MESH_TREE_DEAD {
                        continue;
                    }
                    let dx = tree.position.x - player_pos.x;
                    let dz = tree.position.z - player_pos.z;
                    if dx * dx + dz * dz > emit_range_sq {
                        continue;
                    }
                    // Random chance per tree per call — higher wind = more likely.
                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let roll = hash_f32(h);
                    if roll > wind_strength * 0.04 {
                        continue;
                    }

                    let canopy_height = match tree.mesh_type {
                        MESH_TREE_OAK => 5.0 * tree.scale.y,
                        MESH_TREE_PINE => 6.0 * tree.scale.y,
                        _ => 5.0 * tree.scale.y,
                    };
                    let canopy_center = tree.position + Vec3::new(0.0, canopy_height, 0.0);
                    let canopy_radius = 3.0 * tree.scale.x;

                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let angle = hash_f32(h) * std::f32::consts::TAU;
                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let r = hash_f32(h) * canopy_radius;
                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let pos = canopy_center + Vec3::new(angle.cos() * r, (hash_f32(h) - 0.5) * 1.0, angle.sin() * r);

                    // Velocity: mostly wind-driven with some random spread.
                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let vx = wind_dir.0 * wind_strength * 4.0 + (hash_f32(h) - 0.5) * 2.0;
                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let vy = -0.5 - hash_f32(h) * 1.0;
                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let vz = wind_dir.1 * wind_strength * 4.0 + (hash_f32(h) - 0.5) * 2.0;
                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let rot = hash_f32(h) * std::f32::consts::TAU;
                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let life_extra = hash_f32(h);
                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let _mesh_roll = hash_f32(h);
                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let scale_extra = hash_f32(h);

                    self.leaf_particles.push(LeafParticle {
                        position: pos,
                        velocity: Vec3::new(vx, vy, vz),
                        lifetime: LEAF_LIFETIME + life_extra * 1.0,
                        mesh_type: MESH_LEAF_PARTICLE,
                        scale: 0.12 + scale_extra * 0.08,
                        rotation_y: rot,
                    });
                    emitted += 1;
                }
            }
        }
        *seed = h;
    }

    /// Get the current shake angle for a tree index (0.0 if not shaking).
    fn shake_angle(&self, tree_idx: usize) -> f32 {
        for s in &self.shakes {
            if s.tree_idx == tree_idx {
                let t = SHAKE_DURATION - s.timer;
                let decay = (s.timer / SHAKE_DURATION).powi(2);
                return (t * SHAKE_FREQUENCY).sin() * SHAKE_AMPLITUDE * decay;
            }
        }
        0.0
    }
}

/// Map a full-detail tree mesh type to its LOD variant.
fn lod_mesh(mesh_type: u32) -> u32 {
    match mesh_type {
        MESH_TREE_OAK => MESH_TREE_OAK_LOD,
        MESH_TREE_PINE => MESH_TREE_PINE_LOD,
        MESH_TREE_DEAD => MESH_TREE_DEAD_LOD,
        MESH_CACTUS => MESH_CACTUS_LOD,
        MESH_CACTUS_SMALL => MESH_CACTUS_SMALL_LOD,
        other => other,
    }
}
