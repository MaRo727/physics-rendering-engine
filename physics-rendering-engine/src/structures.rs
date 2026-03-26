use glam::{Mat4, Vec3, Vec4};

use crate::renderer::{pack_instance_id, MESH_TREE_OAK, MESH_TREE_PINE, MESH_TREE_DEAD,
    MESH_TREE_OAK_LOD, MESH_TREE_PINE_LOD, MESH_TREE_DEAD_LOD,
    MESH_GRASS_A, MESH_GRASS_B, MESH_GRASS_C,
    MESH_FLOWER_RED, MESH_FLOWER_YELLOW, MESH_FLOWER_BLUE, MESH_FLOWER_WHITE, MESH_FLOWER_PURPLE};
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
    #[allow(dead_code)]
    pub rotation_y: f32,
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
    /// Currently shaking trees.
    shakes: Vec<TreeShake>,
    /// Active leaf particles from tree punches.
    pub leaf_particles: Vec<LeafParticle>,
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
                            (0.05, MESH_TREE_DEAD)
                        }
                        Biome::Mountains => {
                            if pass > 0 { continue; }
                            if height <= 6.0 || height > 30.0 { continue; }
                            (0.3, MESH_TREE_PINE)
                        }
                        Biome::Dungeon => continue,
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
                    trees.push(TreeInstance {
                        position: pos,
                        mesh_type,
                        scale: Vec3::splat(scale_factor),
                        rotation_y,
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

        Self { trees, chunk_buckets, shakes: Vec::new(), leaf_particles: Vec::new() }
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

        for cx in min_cx..max_cx {
            for cz in min_cz..max_cz {
                let bucket = cx * CHUNKS_PER_SIDE as usize + cz;
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
                    // to derive per-tree sway from precomputed base trig values + per-tree phase.
                    let phase = tree.position.x * 0.13 + tree.position.z * 0.17;
                    let (sp, cp) = (phase.sin(), phase.cos());
                    let (sp13, cp13) = {
                        let p13 = phase * 1.3;
                        (p13.sin(), p13.cos())
                    };
                    let (sp07, cp07) = {
                        let p07 = phase * 0.7;
                        (p07.sin(), p07.cos())
                    };
                    let (sp11, cp11) = {
                        let p11 = phase * 1.1;
                        (p11.sin(), p11.cos())
                    };

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

        // Spawn leaf particles at the tree canopy.
        let tree = &self.trees[tree_idx];
        let canopy_height = match tree.mesh_type {
            MESH_TREE_OAK => 5.0 * tree.scale.y,
            MESH_TREE_PINE => 6.0 * tree.scale.y,
            MESH_TREE_DEAD => 3.0 * tree.scale.y,
            _ => 5.0 * tree.scale.y,
        };
        let canopy_center = tree.position + Vec3::new(0.0, canopy_height, 0.0);
        let canopy_radius = 3.0 * tree.scale.x;

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

            // Use grass mesh types for leaf look
            let mesh = if tree.mesh_type == MESH_TREE_DEAD {
                MESH_GRASS_C // brownish for dead trees
            } else {
                MESH_GRASS_A
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
                    let mesh_roll = hash_f32(h);
                    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
                    let scale_extra = hash_f32(h);

                    self.leaf_particles.push(LeafParticle {
                        position: pos,
                        velocity: Vec3::new(vx, vy, vz),
                        lifetime: LEAF_LIFETIME + life_extra * 1.0,
                        mesh_type: if mesh_roll > 0.5 { MESH_GRASS_A } else { MESH_GRASS_B },
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

// ---------------------------------------------------------------------------
// Grass patches — rare, large fields of 4000-16000 strands
// ---------------------------------------------------------------------------

const GRASS_OBJECT_ID: u32 = 0xFFC0;
/// Only render grass within this distance from the player.
const GRASS_RENDER_DISTANCE: f32 = 250.0;
/// Grid spacing for patch *centers* — wide spacing for rare fields.
const PATCH_CENTER_STEP: f32 = 250.0;
/// Maximum number of strand attempts per patch.
const MAX_STRANDS_PER_PATCH: u32 = 16000;
/// Minimum strands for the smallest patches.
const MIN_STRANDS_PER_PATCH: u32 = 4000;

struct GrassStrand {
    mesh_type: u32,
    /// Pre-computed static transform: translation * rotation_y (without scale).
    /// Sway is applied between rotation and scale at render time.
    base_transform_no_scale: Mat4,
    /// Pre-computed full static transform: translation * rotation_y * scale.
    /// Used when there is no sway (wind off).
    base_transform_full: Mat4,
    scale: Vec3,
}

/// Metadata for a grass patch (used for coarse frustum culling of the whole cluster).
struct PatchInfo {
    center: Vec3,
    radius: f32,
    strand_range: (usize, usize), // index range into strands vec
}

pub struct GrassGrid {
    strands: Vec<GrassStrand>,
    patches: Vec<PatchInfo>,
    /// Patches bucketed by terrain chunk index for spatial queries.
    chunk_buckets: Vec<Vec<usize>>,
}

impl GrassGrid {
    pub fn generate(seed: u32, terrain: &TerrainGrid) -> Self {
        let num_buckets = (CHUNKS_PER_SIDE * CHUNKS_PER_SIDE) as usize;
        let mut strands: Vec<GrassStrand> = Vec::new();
        let mut patches: Vec<PatchInfo> = Vec::new();
        let mut chunk_buckets: Vec<Vec<usize>> = vec![Vec::new(); num_buckets];

        let half = TERRAIN_HALF as f32;
        let steps = ((half * 2.0) / PATCH_CENTER_STEP) as i32;
        let grass_seed = seed.wrapping_add(9999);

        for gz in 0..steps {
            for gx in 0..steps {
                let base_x = -half + gx as f32 * PATCH_CENTER_STEP;
                let base_z = -half + gz as f32 * PATCH_CENTER_STEP;

                let h0 = placement_hash(grass_seed, gx, gz);

                // Jitter patch center within its cell.
                let cx = base_x + hash_f32(h0) * PATCH_CENTER_STEP;
                let cz = base_z + hash_f32(h0.wrapping_mul(2654435761)) * PATCH_CENTER_STEP;

                let biome = terrain.biome_at_world(cx, cz);
                let center_height = terrain.height_at_world(cx, cz);

                // Only spawn patches in grassy biomes.
                let patch_probability = match biome {
                    Biome::Plains => {
                        if center_height <= 6.0 || center_height > 20.0 { continue; }
                        0.05 // more open meadows
                    }
                    Biome::Forest => {
                        if center_height <= 6.0 || center_height > 22.0 { continue; }
                        0.025 // less grass under dense canopy
                    }
                    Biome::Mountains => {
                        if center_height <= 6.0 || center_height > 15.0 { continue; }
                        0.02
                    }
                    Biome::Desert | Biome::Dungeon => continue,
                };

                if hash_f32(h0.wrapping_add(1)) > patch_probability {
                    continue;
                }

                // Determine patch shape parameters.
                let h1 = placement_hash(grass_seed.wrapping_add(100), gx, gz);

                // Patch radius: 40–150 units (large meadows).
                let base_radius = 40.0 + hash_f32(h1) * 110.0;

                // Ellipse stretch: aspect ratio 1.0–2.5 with random orientation.
                let aspect = 1.0 + hash_f32(h1.wrapping_add(1)) * 1.5;
                let ellipse_angle = hash_f32(h1.wrapping_add(2)) * std::f32::consts::TAU;
                let (sin_e, cos_e) = ellipse_angle.sin_cos();

                // Strand count scales with patch area.
                let area_factor = base_radius * base_radius * aspect;
                let max_area = 150.0 * 150.0 * 2.5;
                let strand_t = (area_factor / max_area).min(1.0);
                let strand_count = MIN_STRANDS_PER_PATCH
                    + ((MAX_STRANDS_PER_PATCH - MIN_STRANDS_PER_PATCH) as f32 * strand_t) as u32;

                // Irregularity: use hash-based bumps to make edges organic.
                let irregularity = 0.15 + hash_f32(h1.wrapping_add(3)) * 0.2; // 15-35% edge wobble

                let strand_start = strands.len();

                for si in 0..strand_count {
                    // Use two independent hashes for uncorrelated r and theta.
                    let sh0 = placement_hash(h1.wrapping_add(1000), si as i32, gz);
                    let sh1 = placement_hash(h1.wrapping_add(2000), gx, si as i32);

                    // Random point in unit disk (sqrt for uniform area distribution).
                    let r = hash_f32(sh0).sqrt();
                    let theta = hash_f32(sh1) * std::f32::consts::TAU;
                    let (sin_t, cos_t) = theta.sin_cos();

                    // Apply ellipse stretch.
                    let local_x = r * cos_t * base_radius;
                    let local_z = r * sin_t * base_radius / aspect;

                    // Rotate by ellipse orientation.
                    let rx = local_x * cos_e - local_z * sin_e;
                    let rz = local_x * sin_e + local_z * cos_e;

                    // Apply irregular edge wobble: modulate the effective radius by angle.
                    let sh2 = placement_hash(h1.wrapping_add(3000), si as i32, gx);
                    let wobble_angle = theta * 3.0 + hash_f32(sh2) * 2.0;
                    let wobble = 1.0 + irregularity * wobble_angle.sin();
                    let wx = rx * wobble;
                    let wz = rz * wobble;

                    let strand_x = cx + wx;
                    let strand_z = cz + wz;

                    // Clamp to world bounds.
                    if strand_x < -half || strand_x > half || strand_z < -half || strand_z > half {
                        continue;
                    }

                    // Check biome/height at the strand's actual position.
                    let strand_biome = terrain.biome_at_world(strand_x, strand_z);
                    let strand_height = terrain.height_at_world(strand_x, strand_z);
                    match strand_biome {
                        Biome::Plains => {
                            if strand_height <= 6.0 || strand_height > 20.0 { continue; }
                        }
                        Biome::Forest => {
                            if strand_height <= 6.0 || strand_height > 22.0 { continue; }
                        }
                        Biome::Mountains => {
                            if strand_height <= 6.0 || strand_height > 15.0 { continue; }
                        }
                        _ => continue,
                    }

                    // Pick mesh, scale, rotation — each from an independent hash.
                    let sh3 = placement_hash(h1.wrapping_add(4000), si as i32, gz.wrapping_add(gx));
                    let sh4 = placement_hash(h1.wrapping_add(5000), gz, si as i32);
                    let sh5 = placement_hash(h1.wrapping_add(6000), si as i32, gx.wrapping_mul(17));

                    let shape_roll = hash_f32(sh3);
                    let mesh_type = if shape_roll < 0.32 {
                        MESH_GRASS_A
                    } else if shape_roll < 0.60 {
                        MESH_GRASS_B
                    } else if shape_roll < 0.80 {
                        MESH_GRASS_C
                    } else if shape_roll < 0.84 {
                        MESH_FLOWER_RED
                    } else if shape_roll < 0.88 {
                        MESH_FLOWER_YELLOW
                    } else if shape_roll < 0.92 {
                        MESH_FLOWER_BLUE
                    } else if shape_roll < 0.96 {
                        MESH_FLOWER_WHITE
                    } else {
                        MESH_FLOWER_PURPLE
                    };

                    // Strands near the edge are shorter, center ones taller.
                    let edge_factor = 1.0 - r * 0.4; // 0.6 at edge, 1.0 at center
                    let scale_factor = (0.5 + hash_f32(sh4) * 0.7) * edge_factor;
                    let rotation_y = hash_f32(sh5) * std::f32::consts::TAU;

                    let strand_pos = Vec3::new(strand_x, strand_height, strand_z);
                    let strand_scale = Vec3::splat(scale_factor);
                    let base_transform_no_scale = Mat4::from_translation(strand_pos)
                        * Mat4::from_rotation_y(rotation_y);
                    let base_transform_full = base_transform_no_scale
                        * Mat4::from_scale(strand_scale);
                    strands.push(GrassStrand {
                        mesh_type,
                        base_transform_no_scale,
                        base_transform_full,
                        scale: strand_scale,
                    });
                }

                let strand_end = strands.len();
                if strand_end == strand_start {
                    continue; // no valid strands placed
                }

                let patch_idx = patches.len();
                let cull_radius = base_radius * aspect.max(1.0) * 1.4; // generous for wobble
                patches.push(PatchInfo {
                    center: Vec3::new(cx, center_height, cz),
                    radius: cull_radius,
                    strand_range: (strand_start, strand_end),
                });

                // Bucket the patch by its center chunk.
                let bcx = ((cx + half) / CHUNK_WORLD_SIZE).clamp(0.0, CHUNKS_PER_SIDE as f32 - 1.0) as usize;
                let bcz = ((cz + half) / CHUNK_WORLD_SIZE).clamp(0.0, CHUNKS_PER_SIDE as f32 - 1.0) as usize;
                let bucket = bcx * CHUNKS_PER_SIDE as usize + bcz;
                chunk_buckets[bucket].push(patch_idx);
            }
        }

        log::info!(
            "Placed {} grass strands in {} patches across {} chunks",
            strands.len(), patches.len(), num_buckets,
        );

        Self { strands, patches, chunk_buckets }
    }

    /// Render grass strands near the player with wind-based sway animation.
    /// `wind_strength` (0..1), `wind_dir` is normalized (x, z), `time` is weather animation time.
    pub fn render_nearby(
        &self,
        player_pos: Vec3,
        frustum: &[Vec4; 6],
        wind_strength: f32,
        wind_dir: (f32, f32),
        time: f32,
        transforms: &mut Vec<Mat4>,
        instance_ids: &mut Vec<u32>,
    ) {
        let half = TERRAIN_HALF as f32;
        let render_dist = GRASS_RENDER_DISTANCE;

        let min_cx = ((player_pos.x - render_dist + half) / CHUNK_WORLD_SIZE)
            .floor().max(0.0) as usize;
        let max_cx = ((player_pos.x + render_dist + half) / CHUNK_WORLD_SIZE)
            .ceil().min(CHUNKS_PER_SIDE as f32) as usize;
        let min_cz = ((player_pos.z - render_dist + half) / CHUNK_WORLD_SIZE)
            .floor().max(0.0) as usize;
        let max_cz = ((player_pos.z + render_dist + half) / CHUNK_WORLD_SIZE)
            .ceil().min(CHUNKS_PER_SIDE as f32) as usize;

        // Wind sway is applied as a small tilt (rotation about a horizontal axis perpendicular
        // to wind direction). Each strand gets a unique phase offset from its position hash.
        let sway_enabled = wind_strength > 0.01;

        // Precompute base trig values for the three sway frequencies. Per-patch phase
        // offsets are applied via angle-addition identities (no per-strand sin/cos).
        let sway_scale = wind_strength * 0.15;
        let lean = if sway_enabled { wind_strength * 0.06 } else { 0.0 };
        let base_sin_2_5 = (time * 2.5).sin();
        let base_cos_2_5 = (time * 2.5).cos();
        let base_sin_5_3 = (time * 5.3).sin();
        let base_cos_5_3 = (time * 5.3).cos();
        let base_sin_9_1 = (time * 9.1).sin();
        let base_cos_9_1 = (time * 9.1).cos();

        for cx in min_cx..max_cx {
            for cz in min_cz..max_cz {
                let bucket = cx * CHUNKS_PER_SIDE as usize + cz;
                for &patch_idx in &self.chunk_buckets[bucket] {
                    let patch = &self.patches[patch_idx];

                    // Coarse distance check on the whole patch.
                    let dx = patch.center.x - player_pos.x;
                    let dz = patch.center.z - player_pos.z;
                    let patch_dist_sq = dx * dx + dz * dz;
                    let max_d = render_dist + patch.radius;
                    if patch_dist_sq > max_d * max_d {
                        continue;
                    }

                    // Coarse frustum cull on the whole patch bounding sphere.
                    if !sphere_in_frustum(frustum, patch.center, patch.radius) {
                        continue;
                    }

                    // Precompute per-patch wind sway using the patch center position.
                    // Individual strands add cheap linear variation on top of this.
                    let patch_sway_mat = if sway_enabled {
                        let phase = patch.center.x * 0.37 + patch.center.z * 0.53;
                        let (sp, cp) = (phase.sin(), phase.cos());
                        let (sp17, cp17) = { let p = phase * 1.7; (p.sin(), p.cos()) };
                        let (sp23, cp23) = { let p = phase * 2.3; (p.sin(), p.cos()) };
                        // sin(base + phase*k) via angle-addition identity
                        let sway = sway_scale
                            * ((base_sin_2_5 * cp + base_cos_2_5 * sp)
                               + 0.4 * (base_sin_5_3 * cp17 + base_cos_5_3 * sp17)
                               + 0.15 * (base_sin_9_1 * cp23 + base_cos_9_1 * sp23));
                        let tilt = sway + lean;
                        let a = tilt * wind_dir.1;  // rotation about X
                        let b = -tilt * wind_dir.0; // rotation about Z
                        // Small-angle combined Rx(a)*Rz(b) matrix.
                        Some((a, b, Mat4::from_cols(
                            Vec4::new(1.0, b, 0.0, 0.0),
                            Vec4::new(-b, 1.0, a, 0.0),
                            Vec4::new(a * b, -a, 1.0, 0.0),
                            Vec4::new(0.0, 0.0, 0.0, 1.0),
                        )))
                    } else {
                        None
                    };

                    // Render individual strands within this patch.
                    // Per-strand distance check is removed: patch-level distance + frustum
                    // culling above already limits visibility. The patch bounding sphere
                    // (with generous 1.4x margin) ensures no strands outside render range
                    // are drawn.
                    let (start, end) = patch.strand_range;
                    if let Some((_a, _b, ref sway_mat)) = patch_sway_mat {
                        // Sway active: insert sway between rotation and scale.
                        for strand in &self.strands[start..end] {
                            let transform = strand.base_transform_no_scale
                                * *sway_mat
                                * Mat4::from_scale(strand.scale);
                            transforms.push(transform);
                            instance_ids.push(pack_instance_id(strand.mesh_type, GRASS_OBJECT_ID));
                        }
                    } else {
                        // No sway: use fully pre-computed transform directly.
                        for strand in &self.strands[start..end] {
                            transforms.push(strand.base_transform_full);
                            instance_ids.push(pack_instance_id(strand.mesh_type, GRASS_OBJECT_ID));
                        }
                    }
                }
            }
        }
    }
}
