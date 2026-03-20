use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::renderer::mesh::Vertex;

const TERRAIN_HALF: i32 = 450;
const CELL_SIZE: i32 = 3;
const MIN_HEIGHT: f64 = -15.0;
const MAX_HEIGHT: f64 = 40.0;
const CHUNKS_PER_SIDE: i32 = 6;

pub struct TerrainChunkInfo {
    pub mesh_type: u32,
    pub center: Vec3,
    pub radius: f32,
}

pub struct TerrainGrid {
    fbm: Fbm<Perlin>,
}

impl TerrainGrid {
    pub fn generate(seed: u32) -> Self {
        let fbm = Fbm::<Perlin>::new(seed)
            .set_octaves(5)
            .set_frequency(0.008)
            .set_persistence(0.5);
        Self { fbm }
    }

    fn sample(&self, x: f32, z: f32) -> f32 {
        let val = self.fbm.get([x as f64, z as f64]);
        let sign = val.signum();
        let shaped = sign * val.abs().powf(2.0);
        let range = MAX_HEIGHT - MIN_HEIGHT;
        let h = MIN_HEIGHT + (shaped + 1.0) * 0.5 * range;
        h.clamp(MIN_HEIGHT, MAX_HEIGHT) as f32
    }

    pub fn get_height(&self, x: i32, z: i32) -> f32 {
        self.sample(x as f32, z as f32)
    }

    /// Generate terrain as chunked meshes for frustum culling.
    /// Returns (chunk_meshes for renderer, chunk_infos for engine, full_mesh for physics).
    pub fn generate_chunks(
        &self,
        mesh_type_base: u32,
    ) -> (Vec<(Vec<Vertex>, Vec<u32>)>, Vec<TerrainChunkInfo>, (Vec<Vertex>, Vec<u32>)) {
        let grid_half = TERRAIN_HALF / CELL_SIZE;
        let cells_per_chunk = (grid_half * 2) / CHUNKS_PER_SIDE;
        let step = CELL_SIZE as f32;

        let mut chunk_meshes = Vec::new();
        let mut chunk_infos = Vec::new();
        let mut all_vertices = Vec::new();
        let mut all_indices = Vec::new();

        for chunk_x in 0..CHUNKS_PER_SIDE {
            for chunk_z in 0..CHUNKS_PER_SIDE {
                let cell_x_start = -grid_half + chunk_x * cells_per_chunk;
                let cell_z_start = -grid_half + chunk_z * cells_per_chunk;
                let cell_x_end = cell_x_start + cells_per_chunk;
                let cell_z_end = cell_z_start + cells_per_chunk;

                let mut vertices = Vec::new();
                let mut indices = Vec::new();
                let mut min_y = f32::MAX;
                let mut max_y = f32::MIN;

                for gx in cell_x_start..cell_x_end {
                    for gz in cell_z_start..cell_z_end {
                        let x = gx as f32 * step;
                        let z = gz as f32 * step;

                        let h00 = self.sample(x, z);
                        let h10 = self.sample(x + step, z);
                        let h01 = self.sample(x, z + step);
                        let h11 = self.sample(x + step, z + step);

                        min_y = min_y.min(h00).min(h10).min(h01).min(h11);
                        max_y = max_y.max(h00).max(h10).max(h01).max(h11);

                        let v0 = Vec3::new(x, h00, z);
                        let v1 = Vec3::new(x + step, h10, z);
                        let v2 = Vec3::new(x + step, h11, z + step);
                        let v3 = Vec3::new(x, h01, z + step);

                        let normal = (v3 - v0).cross(v1 - v0).normalize();
                        let avg_h = (h00 + h10 + h01 + h11) * 0.25;
                        let color = height_color(avg_h);

                        let base = vertices.len() as u32;
                        vertices.push(Vertex { position: v0, normal, color });
                        vertices.push(Vertex { position: v1, normal, color });
                        vertices.push(Vertex { position: v2, normal, color });
                        vertices.push(Vertex { position: v3, normal, color });
                        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                    }
                }

                // Append to full mesh (with offset indices).
                let vert_offset = all_vertices.len() as u32;
                all_vertices.extend_from_slice(&vertices);
                for &idx in &indices {
                    all_indices.push(idx + vert_offset);
                }

                // Compute bounding sphere.
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
                chunk_infos.push(TerrainChunkInfo { mesh_type, center, radius });
                chunk_meshes.push((vertices, indices));
            }
        }

        (chunk_meshes, chunk_infos, (all_vertices, all_indices))
    }

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

fn height_color(y: f32) -> Vec3 {
    if y <= 2.0 {
        Vec3::new(0.4, 0.25, 0.1)     // underwater bed (brown)
    } else if y <= 6.0 {
        Vec3::new(0.76, 0.70, 0.50)    // shore (sand)
    } else if y <= 15.0 {
        Vec3::new(0.3, 0.55, 0.2)      // grass (green)
    } else if y <= 25.0 {
        Vec3::new(0.35, 0.42, 0.28)    // highland (dark green)
    } else {
        Vec3::new(0.65, 0.65, 0.62)    // mountain (gray)
    }
}
