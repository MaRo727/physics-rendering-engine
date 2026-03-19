use std::collections::HashMap;

use glam::Vec3;
use rapier3d::prelude::{ColliderHandle, RigidBodyHandle};

use crate::physics::world::PhysicsWorld;
use crate::renderer::mesh::Vertex;

const BUILDING_COLOR: Vec3 = Vec3::new(0.7, 0.7, 0.65);

/// Data stored per occupied grid cell.
struct CellData {
    rigid_body: RigidBodyHandle,
    collider: ColliderHandle,
}

/// A grid-based building system. Cubes snap to integer grid positions.
/// Cell (cx, cy, cz) occupies [cx, cx+1] × [cy, cy+1] × [cz, cz+1].
pub struct BuildingGrid {
    cells: HashMap<(i32, i32, i32), CellData>,
    dirty: bool,
}

impl BuildingGrid {
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            dirty: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    pub fn is_occupied(&self, x: i32, y: i32, z: i32) -> bool {
        self.cells.contains_key(&(x, y, z))
    }

    /// Place a cube at grid position (cx, cy, cz). Returns true if placed.
    pub fn place(&mut self, physics: &mut PhysicsWorld, cx: i32, cy: i32, cz: i32) -> bool {
        if self.is_occupied(cx, cy, cz) {
            return false;
        }

        let center = cell_center(cx, cy, cz);
        let half = Vec3::splat(0.5);
        let (rigid_body, collider) = physics.add_static_box(center, half);

        self.cells.insert((cx, cy, cz), CellData { rigid_body, collider });
        self.dirty = true;
        true
    }

    /// Remove the cube at grid position (cx, cy, cz). Returns true if removed.
    pub fn remove(&mut self, physics: &mut PhysicsWorld, cx: i32, cy: i32, cz: i32) -> bool {
        if let Some(cell) = self.cells.remove(&(cx, cy, cz)) {
            physics.remove_body(cell.rigid_body, cell.collider);
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Generate an optimized mesh with only externally-visible faces.
    pub fn generate_mesh(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for &(cx, cy, cz) in self.cells.keys() {
            let x = cx as f32;
            let y = cy as f32;
            let z = cz as f32;

            // +X face
            if !self.is_occupied(cx + 1, cy, cz) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x + 1.0, y,       z + 1.0),
                    Vec3::new(x + 1.0, y + 1.0, z + 1.0),
                    Vec3::new(x + 1.0, y + 1.0, z),
                    Vec3::new(x + 1.0, y,       z),
                    Vec3::X,
                );
            }
            // -X face
            if !self.is_occupied(cx - 1, cy, cz) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x, y,       z),
                    Vec3::new(x, y + 1.0, z),
                    Vec3::new(x, y + 1.0, z + 1.0),
                    Vec3::new(x, y,       z + 1.0),
                    Vec3::NEG_X,
                );
            }
            // +Y face
            if !self.is_occupied(cx, cy + 1, cz) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x,       y + 1.0, z + 1.0),
                    Vec3::new(x + 1.0, y + 1.0, z + 1.0),
                    Vec3::new(x + 1.0, y + 1.0, z),
                    Vec3::new(x,       y + 1.0, z),
                    Vec3::Y,
                );
            }
            // -Y face
            if !self.is_occupied(cx, cy - 1, cz) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x,       y, z),
                    Vec3::new(x + 1.0, y, z),
                    Vec3::new(x + 1.0, y, z + 1.0),
                    Vec3::new(x,       y, z + 1.0),
                    Vec3::NEG_Y,
                );
            }
            // +Z face
            if !self.is_occupied(cx, cy, cz + 1) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x,       y,       z + 1.0),
                    Vec3::new(x + 1.0, y,       z + 1.0),
                    Vec3::new(x + 1.0, y + 1.0, z + 1.0),
                    Vec3::new(x,       y + 1.0, z + 1.0),
                    Vec3::Z,
                );
            }
            // -Z face
            if !self.is_occupied(cx, cy, cz - 1) {
                push_quad(
                    &mut vertices, &mut indices,
                    Vec3::new(x + 1.0, y,       z),
                    Vec3::new(x,       y,       z),
                    Vec3::new(x,       y + 1.0, z),
                    Vec3::new(x + 1.0, y + 1.0, z),
                    Vec3::NEG_Z,
                );
            }
        }

        (vertices, indices)
    }
}

/// Center of a grid cell in world space.
fn cell_center(cx: i32, cy: i32, cz: i32) -> Vec3 {
    Vec3::new(cx as f32 + 0.5, cy as f32 + 0.5, cz as f32 + 0.5)
}

/// Snap a world position to grid coordinates.
/// The position should be slightly inside the target cell (offset by normal).
pub fn snap_to_grid(pos: Vec3) -> (i32, i32, i32) {
    (pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32)
}

fn push_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    a: Vec3, b: Vec3, c: Vec3, d: Vec3,
    normal: Vec3,
) {
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: a, normal, color: BUILDING_COLOR });
    vertices.push(Vertex { position: b, normal, color: BUILDING_COLOR });
    vertices.push(Vertex { position: c, normal, color: BUILDING_COLOR });
    vertices.push(Vertex { position: d, normal, color: BUILDING_COLOR });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}
