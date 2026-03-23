use std::collections::HashMap;

use glam::Vec3;
use crate::physics::body::{ColliderHandle, Isometry, RigidBodyHandle, SharedShape};
use crate::physics::world::PhysicsWorld;
use crate::renderer::mesh::Vertex;

const BUILDING_COLOR: Vec3 = Vec3::new(0.7, 0.7, 0.65);
const INTERIOR_COLOR: Vec3 = Vec3::new(0.55, 0.52, 0.48);

/// Sub-blocks per axis within each cell (4×4×4 = 64 bits = u64).
const SUBS: i32 = 4;
const SUB_SIZE: f32 = 1.0 / SUBS as f32;
const SUB_HALF: f32 = SUB_SIZE / 2.0;
const ALL_SUBS: u64 = u64::MAX;

/// Radius around a pickaxe hit in which sub-blocks are removed.
const MINE_RADIUS: f32 = 0.35;

// ---------------------------------------------------------------------------
// Sub-block helpers
// ---------------------------------------------------------------------------

#[inline]
fn sub_bit(sx: i32, sy: i32, sz: i32) -> u64 {
    1u64 << (sy * 16 + sz * 4 + sx)
}

#[inline]
fn has_sub(bits: u64, sx: i32, sy: i32, sz: i32) -> bool {
    bits & sub_bit(sx, sy, sz) != 0
}

/// World position of a sub-block center.
fn sub_world_pos(cx: i32, cy: i32, cz: i32, sx: i32, sy: i32, sz: i32) -> Vec3 {
    Vec3::new(
        cx as f32 + (sx as f32 + 0.5) * SUB_SIZE,
        cy as f32 + (sy as f32 + 0.5) * SUB_SIZE,
        cz as f32 + (sz as f32 + 0.5) * SUB_SIZE,
    )
}

/// Wrap a sub-block coordinate into [0, SUBS), adjusting the cell index.
fn wrap(cell: i32, sub: i32) -> (i32, i32) {
    if sub < 0 {
        (cell - 1, sub + SUBS)
    } else if sub >= SUBS {
        (cell + 1, sub - SUBS)
    } else {
        (cell, sub)
    }
}

/// Build a compound physics shape from the remaining sub-blocks in a cell.
fn build_compound_shape(sub_blocks: u64) -> SharedShape {
    let mut shapes = Vec::new();
    for sy in 0..SUBS {
        for sz in 0..SUBS {
            for sx in 0..SUBS {
                if has_sub(sub_blocks, sx, sy, sz) {
                    let iso = Isometry::translation(
                        (sx as f32 + 0.5) * SUB_SIZE - 0.5,
                        (sy as f32 + 0.5) * SUB_SIZE - 0.5,
                        (sz as f32 + 0.5) * SUB_SIZE - 0.5,
                    );
                    shapes.push((iso, SharedShape::cuboid(SUB_HALF, SUB_HALF, SUB_HALF)));
                }
            }
        }
    }
    SharedShape::compound(shapes)
}

// ---------------------------------------------------------------------------
// Cell data
// ---------------------------------------------------------------------------

/// Data stored per occupied grid cell.
struct CellData {
    rigid_body: RigidBodyHandle,
    collider: ColliderHandle,
    /// 64-bit mask for the 4×4×4 sub-block grid. All 1s = fully intact.
    sub_blocks: u64,
}

// ---------------------------------------------------------------------------
// Building grid
// ---------------------------------------------------------------------------

/// A grid-based building system. Cubes snap to integer grid positions.
/// Cell (cx, cy, cz) occupies [cx, cx+1] × [cy, cy+1] × [cz, cz+1].
/// Each cell is subdivided into 4×4×4 sub-blocks that can be individually mined.
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

        self.cells.insert(
            (cx, cy, cz),
            CellData {
                rigid_body,
                collider,
                sub_blocks: ALL_SUBS,
            },
        );
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

    /// Check if a rigid body belongs to a building cell.
    pub fn has_body(&self, rb: RigidBodyHandle) -> bool {
        self.cells.values().any(|c| c.rigid_body == rb)
    }

    /// Mine sub-blocks near `hit_pos`. Returns true if any sub-blocks were removed.
    pub fn mine_at(&mut self, physics: &mut PhysicsWorld, hit_pos: Vec3) -> bool {
        let radius_sq = MINE_RADIUS * MINE_RADIUS;

        // Collect which bits to clear per cell (avoids borrow issues).
        let cx_min = (hit_pos.x - MINE_RADIUS).floor() as i32;
        let cx_max = (hit_pos.x + MINE_RADIUS).floor() as i32;
        let cy_min = (hit_pos.y - MINE_RADIUS).floor() as i32;
        let cy_max = (hit_pos.y + MINE_RADIUS).floor() as i32;
        let cz_min = (hit_pos.z - MINE_RADIUS).floor() as i32;
        let cz_max = (hit_pos.z + MINE_RADIUS).floor() as i32;

        let mut changes: Vec<((i32, i32, i32), u64)> = Vec::new();

        for cy in cy_min..=cy_max {
            for cz in cz_min..=cz_max {
                for cx in cx_min..=cx_max {
                    if let Some(cell) = self.cells.get(&(cx, cy, cz)) {
                        let mut clear_mask = 0u64;
                        for sy in 0..SUBS {
                            for sz in 0..SUBS {
                                for sx in 0..SUBS {
                                    let bit = sub_bit(sx, sy, sz);
                                    if cell.sub_blocks & bit == 0 {
                                        continue;
                                    }
                                    let center = sub_world_pos(cx, cy, cz, sx, sy, sz);
                                    if (center - hit_pos).length_squared() < radius_sq {
                                        clear_mask |= bit;
                                    }
                                }
                            }
                        }
                        if clear_mask != 0 {
                            changes.push(((cx, cy, cz), clear_mask));
                        }
                    }
                }
            }
        }

        if changes.is_empty() {
            return false;
        }

        // Apply changes.
        for ((cx, cy, cz), clear_mask) in changes {
            let cell = self.cells.get_mut(&(cx, cy, cz)).unwrap();
            cell.sub_blocks &= !clear_mask;

            if cell.sub_blocks == 0 {
                // Fully destroyed — remove the cell.
                let cell = self.cells.remove(&(cx, cy, cz)).unwrap();
                physics.remove_body(cell.rigid_body, cell.collider);
            } else {
                // Rebuild collider as compound shape of remaining sub-blocks.
                let shape = build_compound_shape(cell.sub_blocks);
                cell.collider = physics.replace_collider(cell.rigid_body, cell.collider, shape);
            }
        }

        self.dirty = true;
        true
    }

    // -----------------------------------------------------------------------
    // Sub-block queries (for mesh generation)
    // -----------------------------------------------------------------------

    /// Check if a sub-block is solid, handling cross-cell boundaries.
    fn is_solid(&self, cx: i32, cy: i32, cz: i32, sx: i32, sy: i32, sz: i32) -> bool {
        let (cx, sx) = wrap(cx, sx);
        let (cy, sy) = wrap(cy, sy);
        let (cz, sz) = wrap(cz, sz);
        match self.cells.get(&(cx, cy, cz)) {
            Some(cell) => has_sub(cell.sub_blocks, sx, sy, sz),
            None => false,
        }
    }

    // -----------------------------------------------------------------------
    // Mesh generation
    // -----------------------------------------------------------------------

    /// Generate an optimized mesh with only externally-visible faces.
    /// Each sub-block face is emitted only when the neighbor sub-block is empty.
    pub fn generate_mesh(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let s = SUB_SIZE;

        for (&(cx, cy, cz), cell) in &self.cells {
            for sy in 0..SUBS {
                for sz in 0..SUBS {
                    for sx in 0..SUBS {
                        if !has_sub(cell.sub_blocks, sx, sy, sz) {
                            continue;
                        }

                        let x = cx as f32 + sx as f32 * s;
                        let y = cy as f32 + sy as f32 * s;
                        let z = cz as f32 + sz as f32 * s;

                        // Faces exposed to the cell exterior get normal color;
                        // interior (broken) faces get a darker shade.
                        let is_interior = cell.sub_blocks != ALL_SUBS;

                        // +X face
                        if !self.is_solid(cx, cy, cz, sx + 1, sy, sz) {
                            let on_edge = sx == SUBS - 1;
                            let color = if on_edge && !is_interior { BUILDING_COLOR } else { INTERIOR_COLOR };
                            push_quad(
                                &mut vertices, &mut indices,
                                Vec3::new(x + s, y,     z + s),
                                Vec3::new(x + s, y + s, z + s),
                                Vec3::new(x + s, y + s, z),
                                Vec3::new(x + s, y,     z),
                                Vec3::X, color,
                            );
                        }
                        // -X face
                        if !self.is_solid(cx, cy, cz, sx - 1, sy, sz) {
                            let on_edge = sx == 0;
                            let color = if on_edge && !is_interior { BUILDING_COLOR } else { INTERIOR_COLOR };
                            push_quad(
                                &mut vertices, &mut indices,
                                Vec3::new(x, y,     z),
                                Vec3::new(x, y + s, z),
                                Vec3::new(x, y + s, z + s),
                                Vec3::new(x, y,     z + s),
                                Vec3::NEG_X, color,
                            );
                        }
                        // +Y face
                        if !self.is_solid(cx, cy, cz, sx, sy + 1, sz) {
                            let on_edge = sy == SUBS - 1;
                            let color = if on_edge && !is_interior { BUILDING_COLOR } else { INTERIOR_COLOR };
                            push_quad(
                                &mut vertices, &mut indices,
                                Vec3::new(x,     y + s, z + s),
                                Vec3::new(x + s, y + s, z + s),
                                Vec3::new(x + s, y + s, z),
                                Vec3::new(x,     y + s, z),
                                Vec3::Y, color,
                            );
                        }
                        // -Y face
                        if !self.is_solid(cx, cy, cz, sx, sy - 1, sz) {
                            let on_edge = sy == 0;
                            let color = if on_edge && !is_interior { BUILDING_COLOR } else { INTERIOR_COLOR };
                            push_quad(
                                &mut vertices, &mut indices,
                                Vec3::new(x,     y, z),
                                Vec3::new(x + s, y, z),
                                Vec3::new(x + s, y, z + s),
                                Vec3::new(x,     y, z + s),
                                Vec3::NEG_Y, color,
                            );
                        }
                        // +Z face
                        if !self.is_solid(cx, cy, cz, sx, sy, sz + 1) {
                            let on_edge = sz == SUBS - 1;
                            let color = if on_edge && !is_interior { BUILDING_COLOR } else { INTERIOR_COLOR };
                            push_quad(
                                &mut vertices, &mut indices,
                                Vec3::new(x,     y,     z + s),
                                Vec3::new(x + s, y,     z + s),
                                Vec3::new(x + s, y + s, z + s),
                                Vec3::new(x,     y + s, z + s),
                                Vec3::Z, color,
                            );
                        }
                        // -Z face
                        if !self.is_solid(cx, cy, cz, sx, sy, sz - 1) {
                            let on_edge = sz == 0;
                            let color = if on_edge && !is_interior { BUILDING_COLOR } else { INTERIOR_COLOR };
                            push_quad(
                                &mut vertices, &mut indices,
                                Vec3::new(x + s, y,     z),
                                Vec3::new(x,     y,     z),
                                Vec3::new(x,     y + s, z),
                                Vec3::new(x + s, y + s, z),
                                Vec3::NEG_Z, color,
                            );
                        }
                    }
                }
            }
        }

        (vertices, indices)
    }
}

/// Center of a grid cell in world space.
pub fn cell_center(cx: i32, cy: i32, cz: i32) -> Vec3 {
    Vec3::new(cx as f32 + 0.5, cy as f32 + 0.5, cz as f32 + 0.5)
}

/// Snap a world position to grid coordinates.
/// The position should be slightly inside the target cell (offset by normal).
pub fn snap_to_grid(pos: Vec3) -> (i32, i32, i32) {
    (pos.x.floor() as i32, pos.y.floor() as i32, pos.z.floor() as i32)
}

pub(crate) fn push_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    a: Vec3, b: Vec3, c: Vec3, d: Vec3,
    normal: Vec3,
    color: Vec3,
) {
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: a, normal, color });
    vertices.push(Vertex { position: b, normal, color });
    vertices.push(Vertex { position: c, normal, color });
    vertices.push(Vertex { position: d, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}
