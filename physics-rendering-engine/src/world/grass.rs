use glam::{Mat4, Vec3, Vec4};

use crate::renderer::{pack_instance_id,
    MESH_GRASS_A, MESH_GRASS_B, MESH_GRASS_C,
    MESH_FLOWER_RED, MESH_FLOWER_YELLOW, MESH_FLOWER_BLUE, MESH_FLOWER_WHITE, MESH_FLOWER_PURPLE,
};
use crate::world::terrain::{Biome, TerrainGrid, TERRAIN_HALF, CHUNKS_PER_SIDE};
use crate::world::trees::{placement_hash, hash_f32, sphere_in_frustum};

// Chunk world size for bucketing.
const CHUNK_WORLD_SIZE: f32 = (TERRAIN_HALF * 2) as f32 / CHUNKS_PER_SIDE as f32;

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
    /// Precomputed wind-phase trig for the three sway frequencies.
    /// `phase = center.x * 0.37 + center.z * 0.53`.
    /// Stores `[(sin(phase), cos(phase)), (sin(1.7*phase), cos(1.7*phase)),
    ///          (sin(2.3*phase), cos(2.3*phase))]`.
    /// Eliminates 6 trig calls per visible patch each frame.
    phase_sc: [(f32, f32); 3],
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
                    Biome::Desert | Biome::Dungeon | Biome::Crystal => continue,
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
                let phase = cx * 0.37 + cz * 0.53;
                let sc0 = phase.sin_cos();
                let sc1 = (phase * 1.7).sin_cos();
                let sc2 = (phase * 2.3).sin_cos();
                patches.push(PatchInfo {
                    center: Vec3::new(cx, center_height, cz),
                    radius: cull_radius,
                    strand_range: (strand_start, strand_end),
                    phase_sc: [sc0, sc1, sc2],
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

                    // Precompute per-patch wind sway using precomputed phase trig.
                    // Individual strands add cheap linear variation on top of this.
                    let patch_sway_mat = if sway_enabled {
                        let [(sp, cp), (sp17, cp17), (sp23, cp23)] = patch.phase_sc;
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
