use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::renderer::mesh::Vertex;

const TERRAIN_HALF: i32 = 450;
const CELL_SIZE: i32 = 3;
const MIN_HEIGHT: f64 = -15.0;
const MAX_HEIGHT: f64 = 40.0;

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

    pub fn generate_mesh(&self) -> (Vec<Vertex>, Vec<u32>) {
        let grid_half = TERRAIN_HALF / CELL_SIZE; // number of cells in each direction
        let step = CELL_SIZE as f32;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for gx in -grid_half..grid_half {
            for gz in -grid_half..grid_half {
                let x = gx as f32 * step;
                let z = gz as f32 * step;

                let h00 = self.sample(x, z);
                let h10 = self.sample(x + step, z);
                let h01 = self.sample(x, z + step);
                let h11 = self.sample(x + step, z + step);

                let v0 = Vec3::new(x, h00, z);
                let v1 = Vec3::new(x + step, h10, z);
                let v2 = Vec3::new(x + step, h11, z + step);
                let v3 = Vec3::new(x, h01, z + step);

                let normal = (v1 - v0).cross(v3 - v0).normalize();

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

        (vertices, indices)
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
    if y <= -8.0 {
        Vec3::new(0.4, 0.25, 0.1)     // deep pit (brown)
    } else if y <= -2.0 {
        Vec3::new(0.76, 0.70, 0.50)    // shore (sand)
    } else if y <= 12.0 {
        Vec3::new(0.3, 0.55, 0.2)      // grass (green)
    } else if y <= 25.0 {
        Vec3::new(0.35, 0.42, 0.28)    // highland (dark green)
    } else {
        Vec3::new(0.65, 0.65, 0.62)    // mountain (gray)
    }
}
