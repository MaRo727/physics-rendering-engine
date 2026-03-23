use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::renderer::mesh::Vertex;

pub const TERRAIN_HALF: i32 = 1800;
pub const CELL_SIZE: i32 = 3;
pub const CHUNKS_PER_SIDE: i32 = 12;
const GRID_HALF: i32 = TERRAIN_HALF / CELL_SIZE; // 600
const GRID_SIZE: usize = (GRID_HALF * 2 + 1) as usize; // 1201
const CELLS_PER_CHUNK: i32 = (GRID_HALF * 2) / CHUNKS_PER_SIDE; // 100

// ---------------------------------------------------------------------------
// Biomes
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Biome {
    Plains,
    Forest,
    Desert,
    Mountains,
    Dungeon,
}

struct BiomeParams {
    min_height: f32,
    max_height: f32,
    frequency: f64,
    power: f64,
}

impl Biome {
    fn params(self) -> BiomeParams {
        match self {
            Biome::Plains => BiomeParams {
                min_height: -8.0,
                max_height: 22.0,
                frequency: 0.005,
                power: 1.3,
            },
            Biome::Forest => BiomeParams {
                min_height: -10.0,
                max_height: 30.0,
                frequency: 0.006,
                power: 1.5,
            },
            Biome::Desert => BiomeParams {
                min_height: -5.0,
                max_height: 18.0,
                frequency: 0.003,
                power: 1.2,
            },
            Biome::Mountains => BiomeParams {
                min_height: -15.0,
                max_height: 80.0,
                frequency: 0.01,
                power: 2.5,
            },
            Biome::Dungeon => BiomeParams {
                min_height: -20.0,
                max_height: 15.0,
                frequency: 0.012,
                power: 1.0,
            },
        }
    }

    fn color(self, y: f32) -> Vec3 {
        match self {
            Biome::Plains => {
                if y <= 2.0 {
                    Vec3::new(0.4, 0.25, 0.1) // underwater bed
                } else if y <= 6.0 {
                    Vec3::new(0.76, 0.70, 0.50) // shore sand
                } else if y <= 14.0 {
                    Vec3::new(0.35, 0.60, 0.20) // bright open grass
                } else {
                    Vec3::new(0.40, 0.52, 0.25) // dry highland grass
                }
            }
            Biome::Forest => {
                if y <= 2.0 {
                    Vec3::new(0.4, 0.25, 0.1) // underwater bed
                } else if y <= 6.0 {
                    Vec3::new(0.76, 0.70, 0.50) // shore sand
                } else if y <= 15.0 {
                    Vec3::new(0.25, 0.55, 0.15) // lush grass
                } else if y <= 22.0 {
                    Vec3::new(0.2, 0.45, 0.18) // deep forest green
                } else {
                    Vec3::new(0.35, 0.42, 0.28) // highland
                }
            }
            Biome::Desert => {
                if y <= 2.0 {
                    Vec3::new(0.5, 0.35, 0.15) // dry bed
                } else if y <= 6.0 {
                    Vec3::new(0.85, 0.75, 0.50) // light sand
                } else if y <= 12.0 {
                    Vec3::new(0.78, 0.65, 0.38) // dune sand
                } else {
                    Vec3::new(0.72, 0.55, 0.30) // hardpan
                }
            }
            Biome::Mountains => {
                if y <= 2.0 {
                    Vec3::new(0.35, 0.25, 0.15) // dark bed
                } else if y <= 10.0 {
                    Vec3::new(0.3, 0.5, 0.2) // low grass
                } else if y <= 30.0 {
                    Vec3::new(0.45, 0.42, 0.35) // rock
                } else if y <= 55.0 {
                    Vec3::new(0.6, 0.58, 0.55) // bare stone
                } else {
                    Vec3::new(0.9, 0.9, 0.92) // snow
                }
            }
            Biome::Dungeon => {
                if y <= 0.0 {
                    Vec3::new(0.15, 0.12, 0.18) // dark pit
                } else if y <= 5.0 {
                    Vec3::new(0.25, 0.22, 0.28) // dark stone
                } else if y <= 10.0 {
                    Vec3::new(0.35, 0.30, 0.35) // mossy rock
                } else {
                    Vec3::new(0.4, 0.38, 0.42) // weathered stone
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Terrain
// ---------------------------------------------------------------------------

pub struct TerrainChunkInfo {
    pub mesh_type: u32,
    pub center: Vec3,
    pub radius: f32,
}

pub struct TerrainGrid {
    heights: Vec<f32>,
    original_heights: Vec<f32>,
    biomes: Vec<Biome>,
    dirty_chunks: Vec<bool>,
}

/// Determine biome from temperature/moisture noise values.
fn pick_biome(temperature: f64, moisture: f64) -> Biome {
    if temperature > 0.3 && moisture < -0.1 {
        Biome::Desert
    } else if temperature < -0.2 && moisture > 0.0 {
        Biome::Mountains
    } else if temperature < -0.1 && moisture < -0.2 {
        Biome::Dungeon
    } else if moisture > 0.15 {
        Biome::Forest // dense woodland in wetter areas
    } else {
        Biome::Plains // open grassland in drier temperate areas
    }
}

/// Sample terrain height for a given world position and biome.
fn sample_height(fbm: &Fbm<Perlin>, wx: f32, wz: f32, biome: Biome) -> f32 {
    let p = biome.params();
    // Use the FBM at the biome's frequency (scale world coords).
    let scale = p.frequency / 0.008; // relative to base FBM frequency
    let val = fbm.get([wx as f64 * scale, wz as f64 * scale]);
    let sign = val.signum();
    let shaped = sign * val.abs().powf(p.power);
    let range = (p.max_height - p.min_height) as f64;
    let h = p.min_height as f64 + (shaped + 1.0) * 0.5 * range;
    h.clamp(p.min_height as f64, p.max_height as f64) as f32
}

/// Blend two biome heights at a transition zone.
fn blend_biome_height(
    fbm: &Fbm<Perlin>,
    wx: f32,
    wz: f32,
    biome_a: Biome,
    biome_b: Biome,
    t: f32,
) -> f32 {
    let ha = sample_height(fbm, wx, wz, biome_a);
    let hb = sample_height(fbm, wx, wz, biome_b);
    ha * (1.0 - t) + hb * t
}

/// Color a terrain vertex, blending toward dirt brown if it has been dug.
fn terrain_color(y: f32, original_y: f32, biome: Biome) -> Vec3 {
    let base = biome.color(y);
    let dig_depth = original_y - y;
    if dig_depth > 0.1 {
        let t = (dig_depth / 5.0).min(1.0);
        let dirt = Vec3::new(0.35, 0.22, 0.1);
        base * (1.0 - t) + dirt * t
    } else {
        base
    }
}

impl TerrainGrid {
    pub fn generate(seed: u32) -> Self {
        let fbm = Fbm::<Perlin>::new(seed)
            .set_octaves(5)
            .set_frequency(0.008)
            .set_persistence(0.5);

        // Separate noise layers for biome selection (temperature & moisture).
        let temp_noise = Fbm::<Perlin>::new(seed.wrapping_add(100))
            .set_octaves(3)
            .set_frequency(0.0015)
            .set_persistence(0.5);
        let moist_noise = Fbm::<Perlin>::new(seed.wrapping_add(200))
            .set_octaves(3)
            .set_frequency(0.002)
            .set_persistence(0.5);

        let total = GRID_SIZE * GRID_SIZE;
        let mut heights = vec![0.0f32; total];
        let mut biomes = vec![Biome::Plains; total];

        for gz in 0..GRID_SIZE {
            for gx in 0..GRID_SIZE {
                let wx = (gx as i32 - GRID_HALF) as f32 * CELL_SIZE as f32;
                let wz = (gz as i32 - GRID_HALF) as f32 * CELL_SIZE as f32;

                let temperature = temp_noise.get([wx as f64, wz as f64]);
                let moisture = moist_noise.get([wx as f64, wz as f64]);
                let biome = pick_biome(temperature, moisture);

                let idx = gz * GRID_SIZE + gx;
                biomes[idx] = biome;

                // Check neighboring biome for smooth transitions.
                // Use a slightly offset sample to detect transitions.
                let temp2 = temp_noise.get([wx as f64 + 30.0, wz as f64 + 30.0]);
                let moist2 = moist_noise.get([wx as f64 + 30.0, wz as f64 + 30.0]);
                let neighbor_biome = pick_biome(temp2, moist2);

                if neighbor_biome != biome {
                    // Blend zone — compute a soft transition factor.
                    let edge_dist = (temperature.abs().min(0.15) + moisture.abs().min(0.15)) as f32;
                    let t = (1.0 - edge_dist / 0.3).clamp(0.0, 0.5);
                    heights[idx] = blend_biome_height(&fbm, wx, wz, biome, neighbor_biome, t);
                } else {
                    heights[idx] = sample_height(&fbm, wx, wz, biome);
                }
            }
        }

        let original_heights = heights.clone();
        let num_chunks = (CHUNKS_PER_SIDE * CHUNKS_PER_SIDE) as usize;
        let dirty_chunks = vec![false; num_chunks];

        Self {
            heights,
            original_heights,
            biomes,
            dirty_chunks,
        }
    }

    // -----------------------------------------------------------------------
    // Height access
    // -----------------------------------------------------------------------

    /// Height at a grid vertex. gx, gz are grid indices in [-GRID_HALF, GRID_HALF].
    fn height_at_grid(&self, gx: i32, gz: i32) -> f32 {
        let col = (gx + GRID_HALF) as usize;
        let row = (gz + GRID_HALF) as usize;
        self.heights[row * GRID_SIZE + col]
    }

    fn original_at_grid(&self, gx: i32, gz: i32) -> f32 {
        let col = (gx + GRID_HALF) as usize;
        let row = (gz + GRID_HALF) as usize;
        self.original_heights[row * GRID_SIZE + col]
    }

    fn biome_at_grid(&self, gx: i32, gz: i32) -> Biome {
        let col = (gx + GRID_HALF) as usize;
        let row = (gz + GRID_HALF) as usize;
        self.biomes[row * GRID_SIZE + col]
    }

    /// Bilinear-interpolated height at an arbitrary world-space (x, z).
    pub fn height_at_world(&self, x: f32, z: f32) -> f32 {
        let step = CELL_SIZE as f32;
        let gx_f = x / step + GRID_HALF as f32;
        let gz_f = z / step + GRID_HALF as f32;
        let gx0 = (gx_f.floor() as usize).min(GRID_SIZE - 2);
        let gz0 = (gz_f.floor() as usize).min(GRID_SIZE - 2);
        let gx1 = gx0 + 1;
        let gz1 = gz0 + 1;
        let fx = (gx_f - gx0 as f32).clamp(0.0, 1.0);
        let fz = (gz_f - gz0 as f32).clamp(0.0, 1.0);

        let h00 = self.heights[gz0 * GRID_SIZE + gx0];
        let h10 = self.heights[gz0 * GRID_SIZE + gx1];
        let h01 = self.heights[gz1 * GRID_SIZE + gx0];
        let h11 = self.heights[gz1 * GRID_SIZE + gx1];

        let h0 = h00 + (h10 - h00) * fx;
        let h1 = h01 + (h11 - h01) * fx;
        h0 + (h1 - h0) * fz
    }

    /// Legacy helper used for player spawn position.
    pub fn get_height(&self, x: i32, z: i32) -> f32 {
        self.height_at_world(x as f32, z as f32)
    }

    /// Get the biome at a world-space position.
    pub fn biome_at_world(&self, x: f32, z: f32) -> Biome {
        let step = CELL_SIZE as f32;
        let gx = ((x / step) + GRID_HALF as f32).round() as usize;
        let gz = ((z / step) + GRID_HALF as f32).round() as usize;
        let gx = gx.min(GRID_SIZE - 1);
        let gz = gz.min(GRID_SIZE - 1);
        self.biomes[gz * GRID_SIZE + gx]
    }

    // -----------------------------------------------------------------------
    // Deformation
    // -----------------------------------------------------------------------

    /// Check if a world position is in the mountain snow zone (biome=Mountains, height>30).
    pub fn is_snow_zone(&self, x: f32, z: f32) -> bool {
        self.biome_at_world(x, z) == Biome::Mountains && self.height_at_world(x, z) > 30.0
    }

    /// Lower terrain within `radius` of `point` by up to `amount` (with linear falloff).
    pub fn deform_ground(&mut self, point: Vec3, radius: f32, amount: f32) {
        let step = CELL_SIZE as f32;
        let min_gx = ((point.x - radius) / step).floor() as i32;
        let max_gx = ((point.x + radius) / step).ceil() as i32;
        let min_gz = ((point.z - radius) / step).floor() as i32;
        let max_gz = ((point.z + radius) / step).ceil() as i32;

        for gz in min_gz.max(-GRID_HALF)..=max_gz.min(GRID_HALF) {
            for gx in min_gx.max(-GRID_HALF)..=max_gx.min(GRID_HALF) {
                let vx = gx as f32 * step;
                let vz = gz as f32 * step;
                let dist = ((vx - point.x).powi(2) + (vz - point.z).powi(2)).sqrt();
                if dist < radius {
                    let falloff = 1.0 - dist / radius;
                    let idx = (gz + GRID_HALF) as usize * GRID_SIZE + (gx + GRID_HALF) as usize;
                    self.heights[idx] = (self.heights[idx] - amount * falloff)
                        .clamp(-20.0, 80.0);
                    self.mark_dirty_for_vertex(gx, gz);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Dirty chunk tracking
    // -----------------------------------------------------------------------

    fn mark_dirty_for_vertex(&mut self, gx: i32, gz: i32) {
        // A vertex is used by cells (gx-1..gx, gz-1..gz), which may span chunks.
        for dgx in -1..=0 {
            for dgz in -1..=0 {
                let cell_gx = gx + dgx;
                let cell_gz = gz + dgz;
                if cell_gx >= -GRID_HALF
                    && cell_gx < GRID_HALF
                    && cell_gz >= -GRID_HALF
                    && cell_gz < GRID_HALF
                {
                    let cx =
                        ((cell_gx + GRID_HALF) / CELLS_PER_CHUNK).min(CHUNKS_PER_SIDE - 1);
                    let cz =
                        ((cell_gz + GRID_HALF) / CELLS_PER_CHUNK).min(CHUNKS_PER_SIDE - 1);
                    self.dirty_chunks[(cx * CHUNKS_PER_SIDE + cz) as usize] = true;
                }
            }
        }
    }

    /// Returns indices of dirty chunks and clears the dirty flags.
    pub fn take_dirty_chunks(&mut self) -> Vec<usize> {
        let mut dirty = Vec::new();
        for (i, d) in self.dirty_chunks.iter_mut().enumerate() {
            if *d {
                dirty.push(i);
                *d = false;
            }
        }
        dirty
    }

    pub fn has_dirty_chunks(&self) -> bool {
        self.dirty_chunks.iter().any(|&d| d)
    }

    // -----------------------------------------------------------------------
    // Chunk mesh generation
    // -----------------------------------------------------------------------

    /// Generate all terrain chunks. Returns (chunk_meshes, chunk_infos, full_mesh).
    pub fn generate_chunks(
        &self,
        mesh_type_base: u32,
    ) -> (
        Vec<(Vec<Vertex>, Vec<u32>)>,
        Vec<TerrainChunkInfo>,
        (Vec<Vertex>, Vec<u32>),
    ) {
        let step = CELL_SIZE as f32;

        let mut chunk_meshes = Vec::new();
        let mut chunk_infos = Vec::new();
        let mut all_vertices = Vec::new();
        let mut all_indices = Vec::new();

        for chunk_x in 0..CHUNKS_PER_SIDE {
            for chunk_z in 0..CHUNKS_PER_SIDE {
                let (vertices, indices, min_y, max_y) =
                    self.build_chunk_mesh(chunk_x, chunk_z);

                // Append to full mesh (with offset indices).
                let vert_offset = all_vertices.len() as u32;
                all_vertices.extend_from_slice(&vertices);
                for &idx in &indices {
                    all_indices.push(idx + vert_offset);
                }

                // Compute bounding sphere.
                let cell_x_start = -GRID_HALF + chunk_x * CELLS_PER_CHUNK;
                let cell_z_start = -GRID_HALF + chunk_z * CELLS_PER_CHUNK;
                let cell_x_end = cell_x_start + CELLS_PER_CHUNK;
                let cell_z_end = cell_z_start + CELLS_PER_CHUNK;
                let world_x_min = cell_x_start as f32 * step;
                let world_x_max = cell_x_end as f32 * step;
                let world_z_min = cell_z_start as f32 * step;
                let world_z_max = cell_z_end as f32 * step;
                let center = Vec3::new(
                    (world_x_min + world_x_max) * 0.5,
                    (min_y + max_y) * 0.5,
                    (world_z_min + world_z_max) * 0.5,
                );
                let half_x = (world_x_max - world_x_min) * 0.5;
                let half_y = (max_y - min_y) * 0.5;
                let half_z = (world_z_max - world_z_min) * 0.5;
                let radius = (half_x * half_x + half_y * half_y + half_z * half_z).sqrt();

                let mesh_type = mesh_type_base + chunk_meshes.len() as u32;
                chunk_infos.push(TerrainChunkInfo {
                    mesh_type,
                    center,
                    radius,
                });
                chunk_meshes.push((vertices, indices));
            }
        }

        (chunk_meshes, chunk_infos, (all_vertices, all_indices))
    }

    /// Regenerate a single chunk by its linear index (chunk_x * CHUNKS_PER_SIDE + chunk_z).
    pub fn regenerate_chunk(&self, chunk_idx: usize) -> (Vec<Vertex>, Vec<u32>) {
        let chunk_x = chunk_idx as i32 / CHUNKS_PER_SIDE;
        let chunk_z = chunk_idx as i32 % CHUNKS_PER_SIDE;
        let (verts, indices, _min_y, _max_y) = self.build_chunk_mesh(chunk_x, chunk_z);
        (verts, indices)
    }

    fn build_chunk_mesh(
        &self,
        chunk_x: i32,
        chunk_z: i32,
    ) -> (Vec<Vertex>, Vec<u32>, f32, f32) {
        let step = CELL_SIZE as f32;
        let cell_x_start = -GRID_HALF + chunk_x * CELLS_PER_CHUNK;
        let cell_z_start = -GRID_HALF + chunk_z * CELLS_PER_CHUNK;
        let cell_x_end = cell_x_start + CELLS_PER_CHUNK;
        let cell_z_end = cell_z_start + CELLS_PER_CHUNK;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        for gx in cell_x_start..cell_x_end {
            for gz in cell_z_start..cell_z_end {
                let x = gx as f32 * step;
                let z = gz as f32 * step;

                let h00 = self.height_at_grid(gx, gz);
                let h10 = self.height_at_grid(gx + 1, gz);
                let h01 = self.height_at_grid(gx, gz + 1);
                let h11 = self.height_at_grid(gx + 1, gz + 1);

                min_y = min_y.min(h00).min(h10).min(h01).min(h11);
                max_y = max_y.max(h00).max(h10).max(h01).max(h11);

                let v0 = Vec3::new(x, h00, z);
                let v1 = Vec3::new(x + step, h10, z);
                let v2 = Vec3::new(x + step, h11, z + step);
                let v3 = Vec3::new(x, h01, z + step);

                let normal = (v3 - v0).cross(v1 - v0).normalize();
                let avg_h = (h00 + h10 + h01 + h11) * 0.25;

                // Blend toward dirt if vertex has been dug from original height.
                let orig_avg = (self.original_at_grid(gx, gz)
                    + self.original_at_grid(gx + 1, gz)
                    + self.original_at_grid(gx, gz + 1)
                    + self.original_at_grid(gx + 1, gz + 1))
                    * 0.25;
                let biome = self.biome_at_grid(gx, gz);
                let color = terrain_color(avg_h, orig_avg, biome);

                let base = vertices.len() as u32;
                vertices.push(Vertex {
                    position: v0,
                    normal,
                    color,
                });
                vertices.push(Vertex {
                    position: v1,
                    normal,
                    color,
                });
                vertices.push(Vertex {
                    position: v2,
                    normal,
                    color,
                });
                vertices.push(Vertex {
                    position: v3,
                    normal,
                    color,
                });
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }

        (vertices, indices, min_y, max_y)
    }

    // -----------------------------------------------------------------------
    // Physics heightfield data
    // -----------------------------------------------------------------------

    /// Return the raw height data and grid dimensions for building a Rapier HeightField.
    /// Layout: heights\[row * ncols + col\] where row=Z index, col=X index.
    pub fn heightfield_data(&self) -> (&[f32], usize, usize) {
        (&self.heights, GRID_SIZE, GRID_SIZE)
    }

    /// Return per-chunk heightfield data: (heights_vec, nrows, ncols, center_x, center_z, scale).
    /// Heights for one chunk: (CELLS_PER_CHUNK+1) × (CELLS_PER_CHUNK+1) grid points.
    pub fn chunk_heightfield_data(&self, chunk_idx: usize) -> (Vec<f32>, usize, usize, f32, f32) {
        let chunk_x = chunk_idx as i32 / CHUNKS_PER_SIDE;
        let chunk_z = chunk_idx as i32 % CHUNKS_PER_SIDE;
        let gx_start = chunk_x * CELLS_PER_CHUNK; // relative to -GRID_HALF
        let gz_start = chunk_z * CELLS_PER_CHUNK;
        let npts = CELLS_PER_CHUNK + 1; // grid points per side (101)
        let npts_u = npts as usize;

        let mut heights = Vec::with_capacity(npts_u * npts_u);
        for dz in 0..npts {
            for dx in 0..npts {
                let gx = (gx_start + dx) as usize;
                let gz = (gz_start + dz) as usize;
                heights.push(self.heights[gz * GRID_SIZE + gx]);
            }
        }

        let step = CELL_SIZE as f32;
        let world_x_start = (-GRID_HALF + gx_start) as f32 * step;
        let world_z_start = (-GRID_HALF + gz_start) as f32 * step;
        let chunk_world_size = CELLS_PER_CHUNK as f32 * step;
        let center_x = world_x_start + chunk_world_size * 0.5;
        let center_z = world_z_start + chunk_world_size * 0.5;

        (heights, npts_u, npts_u, center_x, center_z)
    }

    /// Number of terrain chunks.
    pub fn chunk_count(&self) -> usize {
        (CHUNKS_PER_SIDE * CHUNKS_PER_SIDE) as usize
    }

    /// Legacy helper — convert a full mesh to trimesh format.
    #[allow(dead_code)]
    pub fn physics_trimesh(mesh: &(Vec<Vertex>, Vec<u32>)) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let (verts, indices) = mesh;
        let positions: Vec<Vec3> = verts.iter().map(|v| v.position).collect();
        let triangles: Vec<[u32; 3]> = indices
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        (positions, triangles)
    }
}
