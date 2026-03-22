use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::renderer::mesh::Vertex;

pub const TERRAIN_HALF: i32 = 450;
pub const CELL_SIZE: i32 = 3;
const MIN_HEIGHT: f64 = -15.0;
const MAX_HEIGHT: f64 = 40.0;
pub const CHUNKS_PER_SIDE: i32 = 6;
const GRID_HALF: i32 = TERRAIN_HALF / CELL_SIZE; // 150
const GRID_SIZE: usize = (GRID_HALF * 2 + 1) as usize; // 301
const CELLS_PER_CHUNK: i32 = (GRID_HALF * 2) / CHUNKS_PER_SIDE; // 50

pub struct TerrainChunkInfo {
    pub mesh_type: u32,
    pub center: Vec3,
    pub radius: f32,
}

pub struct TerrainGrid {
    #[allow(dead_code)]
    fbm: Fbm<Perlin>,
    heights: Vec<f32>,
    original_heights: Vec<f32>,
    dirty_chunks: Vec<bool>,
}

fn sample_noise(fbm: &Fbm<Perlin>, x: f32, z: f32) -> f32 {
    let val = fbm.get([x as f64, z as f64]);
    let sign = val.signum();
    let shaped = sign * val.abs().powf(2.0);
    let range = MAX_HEIGHT - MIN_HEIGHT;
    let h = MIN_HEIGHT + (shaped + 1.0) * 0.5 * range;
    h.clamp(MIN_HEIGHT, MAX_HEIGHT) as f32
}

fn height_color(y: f32) -> Vec3 {
    if y <= 2.0 {
        Vec3::new(0.4, 0.25, 0.1) // underwater bed (brown)
    } else if y <= 6.0 {
        Vec3::new(0.76, 0.70, 0.50) // shore (sand)
    } else if y <= 15.0 {
        Vec3::new(0.3, 0.55, 0.2) // grass (green)
    } else if y <= 25.0 {
        Vec3::new(0.35, 0.42, 0.28) // highland (dark green)
    } else {
        Vec3::new(0.65, 0.65, 0.62) // mountain (gray)
    }
}

/// Color a terrain vertex, blending toward dirt brown if it has been dug.
fn terrain_color(y: f32, original_y: f32) -> Vec3 {
    let base = height_color(y);
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

        let mut heights = vec![0.0f32; GRID_SIZE * GRID_SIZE];
        for gz in 0..GRID_SIZE {
            for gx in 0..GRID_SIZE {
                let wx = (gx as i32 - GRID_HALF) as f32 * CELL_SIZE as f32;
                let wz = (gz as i32 - GRID_HALF) as f32 * CELL_SIZE as f32;
                heights[gz * GRID_SIZE + gx] = sample_noise(&fbm, wx, wz);
            }
        }
        let original_heights = heights.clone();
        let num_chunks = (CHUNKS_PER_SIDE * CHUNKS_PER_SIDE) as usize;
        let dirty_chunks = vec![false; num_chunks];

        Self {
            fbm,
            heights,
            original_heights,
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

    // -----------------------------------------------------------------------
    // Deformation
    // -----------------------------------------------------------------------

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
                        .clamp(MIN_HEIGHT as f32, MAX_HEIGHT as f32);
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
                let color = terrain_color(avg_h, orig_avg);

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
