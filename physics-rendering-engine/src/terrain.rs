use std::collections::HashMap;

use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::renderer::mesh::Vertex;

const TERRAIN_HALF: i32 = 45;
const MIN_HEIGHT: i32 = -5;
const MAX_HEIGHT: i32 = 20;

pub struct TerrainGrid {
    heights: HashMap<(i32, i32), f32>,
}

impl TerrainGrid {
    pub fn generate(seed: u32) -> Self {
        let fbm = Fbm::<Perlin>::new(seed)
            .set_octaves(5)
            .set_frequency(0.02)
            .set_persistence(0.5);

        let mut heights = HashMap::new();

        for x in -TERRAIN_HALF..=TERRAIN_HALF {
            for z in -TERRAIN_HALF..=TERRAIN_HALF {
                let val = fbm.get([x as f64, z as f64]);
                let range = (MAX_HEIGHT - MIN_HEIGHT) as f64;
                let h = MIN_HEIGHT as f64 + (val + 1.0) * 0.5 * range;
                let h = h.clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as f32;
                heights.insert((x, z), h);
            }
        }

        Self { heights }
    }

    pub fn get_height(&self, x: i32, z: i32) -> f32 {
        self.heights.get(&(x, z)).copied().unwrap_or(0.0)
    }

    /// Generate a smooth heightmap mesh: one quad per grid cell, vertices at terrain height.
    pub fn generate_mesh(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for cx in -TERRAIN_HALF..TERRAIN_HALF {
            for cz in -TERRAIN_HALF..TERRAIN_HALF {
                let x = cx as f32;
                let z = cz as f32;

                // Four corner heights for this cell.
                let h00 = self.get_height(cx, cz);
                let h10 = self.get_height(cx + 1, cz);
                let h01 = self.get_height(cx, cz + 1);
                let h11 = self.get_height(cx + 1, cz + 1);

                let v0 = Vec3::new(x, h00, z);
                let v1 = Vec3::new(x + 1.0, h10, z);
                let v2 = Vec3::new(x + 1.0, h11, z + 1.0);
                let v3 = Vec3::new(x, h01, z + 1.0);

                // Compute face normal from cross product.
                let normal = (v1 - v0).cross(v3 - v0).normalize();

                // Color based on average height of the quad.
                let avg_h = (h00 + h10 + h01 + h11) * 0.25;
                let color = height_color(avg_h);

                let base = vertices.len() as u32;
                vertices.push(Vertex { position: v0, normal, color });
                vertices.push(Vertex { position: v1, normal, color });
                vertices.push(Vertex { position: v2, normal, color });
                vertices.push(Vertex { position: v3, normal, color });

                // Two triangles per quad.
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }

        (vertices, indices)
    }

    /// Extract physics trimesh data from the terrain mesh.
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
    if y <= -3.0 {
        Vec3::new(0.4, 0.25, 0.1)
    } else if y <= -1.0 {
        Vec3::new(0.76, 0.70, 0.50)
    } else if y <= 5.0 {
        Vec3::new(0.3, 0.55, 0.2)
    } else if y <= 10.0 {
        Vec3::new(0.35, 0.42, 0.28)
    } else {
        Vec3::new(0.65, 0.65, 0.62)
    }
}
