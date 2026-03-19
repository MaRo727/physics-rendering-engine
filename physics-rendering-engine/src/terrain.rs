use std::collections::{HashMap, HashSet};

use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::building::push_quad;
use crate::renderer::mesh::Vertex;

const TERRAIN_HALF: i32 = 45;
const MIN_HEIGHT: i32 = -5;
const MAX_HEIGHT: i32 = 20;

pub struct TerrainGrid {
    heights: HashMap<(i32, i32), i32>,
    cells: HashSet<(i32, i32, i32)>,
}

impl TerrainGrid {
    pub fn generate(seed: u32) -> Self {
        let fbm = Fbm::<Perlin>::new(seed)
            .set_octaves(5)
            .set_frequency(0.02)
            .set_persistence(0.5);

        let mut heights = HashMap::new();
        let mut cells = HashSet::new();

        for x in -TERRAIN_HALF..TERRAIN_HALF {
            for z in -TERRAIN_HALF..TERRAIN_HALF {
                let val = fbm.get([x as f64, z as f64]);
                // Map noise [-1, 1] to [MIN_HEIGHT, MAX_HEIGHT]
                let range = (MAX_HEIGHT - MIN_HEIGHT) as f64;
                let h = MIN_HEIGHT + ((val + 1.0) * 0.5 * range) as i32;
                let h = h.clamp(MIN_HEIGHT, MAX_HEIGHT);
                heights.insert((x, z), h);

                for y in MIN_HEIGHT..=h {
                    cells.insert((x, y, z));
                }
            }
        }

        Self { heights, cells }
    }

    pub fn get_height(&self, x: i32, z: i32) -> i32 {
        self.heights.get(&(x, z)).copied().unwrap_or(0)
    }

    pub fn generate_mesh(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for &(cx, cy, cz) in &self.cells {
            let x = cx as f32;
            let y = cy as f32;
            let z = cz as f32;

            let top_color = height_color(cy);
            let side_color = top_color * 0.8;

            // +X face
            if !self.cells.contains(&(cx + 1, cy, cz)) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x + 1.0, y,       z + 1.0),
                    Vec3::new(x + 1.0, y + 1.0, z + 1.0),
                    Vec3::new(x + 1.0, y + 1.0, z),
                    Vec3::new(x + 1.0, y,       z),
                    Vec3::X,
                    side_color,
                );
            }
            // -X face
            if !self.cells.contains(&(cx - 1, cy, cz)) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x, y,       z),
                    Vec3::new(x, y + 1.0, z),
                    Vec3::new(x, y + 1.0, z + 1.0),
                    Vec3::new(x, y,       z + 1.0),
                    Vec3::NEG_X,
                    side_color,
                );
            }
            // +Y face
            if !self.cells.contains(&(cx, cy + 1, cz)) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x,       y + 1.0, z + 1.0),
                    Vec3::new(x + 1.0, y + 1.0, z + 1.0),
                    Vec3::new(x + 1.0, y + 1.0, z),
                    Vec3::new(x,       y + 1.0, z),
                    Vec3::Y,
                    top_color,
                );
            }
            // -Y face
            if !self.cells.contains(&(cx, cy - 1, cz)) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x,       y, z),
                    Vec3::new(x + 1.0, y, z),
                    Vec3::new(x + 1.0, y, z + 1.0),
                    Vec3::new(x,       y, z + 1.0),
                    Vec3::NEG_Y,
                    side_color,
                );
            }
            // +Z face
            if !self.cells.contains(&(cx, cy, cz + 1)) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x,       y,       z + 1.0),
                    Vec3::new(x + 1.0, y,       z + 1.0),
                    Vec3::new(x + 1.0, y + 1.0, z + 1.0),
                    Vec3::new(x,       y + 1.0, z + 1.0),
                    Vec3::Z,
                    side_color,
                );
            }
            // -Z face
            if !self.cells.contains(&(cx, cy, cz - 1)) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x + 1.0, y,       z),
                    Vec3::new(x,       y,       z),
                    Vec3::new(x,       y + 1.0, z),
                    Vec3::new(x + 1.0, y + 1.0, z),
                    Vec3::NEG_Z,
                    side_color,
                );
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

fn height_color(y: i32) -> Vec3 {
    if y <= -3 {
        Vec3::new(0.4, 0.25, 0.1)    // lake bed (brown)
    } else if y <= -1 {
        Vec3::new(0.76, 0.70, 0.50)   // shore (sand)
    } else if y <= 5 {
        Vec3::new(0.3, 0.55, 0.2)     // grass (green)
    } else if y <= 10 {
        Vec3::new(0.35, 0.42, 0.28)   // highland (dark green)
    } else {
        Vec3::new(0.65, 0.65, 0.62)   // mountain (gray)
    }
}
