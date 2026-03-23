use std::collections::{HashMap, HashSet, VecDeque};

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

// Face masks for the 4×4×4 sub-block grid.  Bit index = sy*16 + sz*4 + sx.
const TOP_LAYER_MASK: u64    = 0xFFFF_0000_0000_0000; // sy = 3
const BOTTOM_LAYER_MASK: u64 = 0x0000_0000_0000_FFFF; // sy = 0
const POS_X_FACE_MASK: u64   = 0x8888_8888_8888_8888; // sx = 3
const NEG_X_FACE_MASK: u64   = 0x1111_1111_1111_1111; // sx = 0
const POS_Z_FACE_MASK: u64   = 0xF000_F000_F000_F000; // sz = 3
const NEG_Z_FACE_MASK: u64   = 0x000F_000F_000F_000F; // sz = 0

/// (neighbor offset, this cell's face mask, neighbor's face mask)
const SUPPORT_NEIGHBORS: [((i32, i32, i32), u64, u64); 5] = [
    ((0, -1, 0), BOTTOM_LAYER_MASK, TOP_LAYER_MASK),   // below
    ((1,  0, 0), POS_X_FACE_MASK,   NEG_X_FACE_MASK),  // +X
    ((-1, 0, 0), NEG_X_FACE_MASK,   POS_X_FACE_MASK),  // -X
    ((0,  0, 1), POS_Z_FACE_MASK,   NEG_Z_FACE_MASK),  // +Z
    ((0,  0,-1), NEG_Z_FACE_MASK,   POS_Z_FACE_MASK),  // -Z
];

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

    /// Check if a cell is supported by any neighbor (below or sideways) or terrain.
    /// `terrain_height` is the terrain surface height at the cell's center XZ.
    /// For cells that don't exist yet (placement check), assumes a full block.
    pub fn is_supported(&self, cx: i32, cy: i32, cz: i32, terrain_height: f32) -> bool {
        // Sub-blocks of this cell (if it doesn't exist yet, assume full block).
        let self_subs = match self.cells.get(&(cx, cy, cz)) {
            Some(cell) => cell.sub_blocks,
            None => ALL_SUBS,
        };

        // Check each neighbor: below + 4 horizontal sides.
        for &((dx, dy, dz), self_face, neighbor_face) in &SUPPORT_NEIGHBORS {
            // This cell must have solid sub-blocks on the connecting face.
            if self_subs & self_face == 0 {
                continue;
            }
            if let Some(neighbor) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                if neighbor.sub_blocks & neighbor_face != 0 {
                    return true;
                }
            }
        }

        // Supported if the terrain surface reaches the bottom of this cell.
        if self_subs & BOTTOM_LAYER_MASK != 0 && terrain_height >= cy as f32 {
            return true;
        }

        false
    }

    /// Place a cube at grid position (cx, cy, cz). Returns true if placed.
    /// The block must be supported from below (another block or terrain).
    pub fn place(&mut self, physics: &mut PhysicsWorld, cx: i32, cy: i32, cz: i32, terrain_height: f32) -> bool {
        if self.is_occupied(cx, cy, cz) {
            return false;
        }
        if !self.is_supported(cx, cy, cz, terrain_height) {
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

    /// Mine sub-blocks near `hit_pos`. Returns all affected cell coordinates
    /// (both partially and fully destroyed) so the caller can check for collapse.
    pub fn mine_at(&mut self, physics: &mut PhysicsWorld, hit_pos: Vec3) -> Vec<(i32, i32, i32)> {
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
            return Vec::new();
        }

        // Apply changes.
        let mut affected = Vec::new();
        for ((cx, cy, cz), clear_mask) in changes {
            let cell = self.cells.get_mut(&(cx, cy, cz)).unwrap();
            cell.sub_blocks &= !clear_mask;
            affected.push((cx, cy, cz));

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
        affected
    }

    /// Collect all cells reachable from `start` via face-connected neighbors.
    fn flood_connected(&self, start: (i32, i32, i32)) -> HashSet<(i32, i32, i32)> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start);
        queue.push_back(start);

        while let Some((cx, cy, cz)) = queue.pop_front() {
            let self_subs = match self.cells.get(&(cx, cy, cz)) {
                Some(cell) => cell.sub_blocks,
                None => continue,
            };

            // Check all 6 neighbors (including below for downward connectivity).
            const DIRS: [((i32, i32, i32), u64, u64); 6] = [
                ((0,  1, 0), TOP_LAYER_MASK,    BOTTOM_LAYER_MASK),
                ((0, -1, 0), BOTTOM_LAYER_MASK,  TOP_LAYER_MASK),
                ((1,  0, 0), POS_X_FACE_MASK,    NEG_X_FACE_MASK),
                ((-1, 0, 0), NEG_X_FACE_MASK,    POS_X_FACE_MASK),
                ((0,  0, 1), POS_Z_FACE_MASK,    NEG_Z_FACE_MASK),
                ((0,  0,-1), NEG_Z_FACE_MASK,    POS_Z_FACE_MASK),
            ];

            for &((dx, dy, dz), self_face, neighbor_face) in &DIRS {
                if self_subs & self_face == 0 {
                    continue;
                }
                let nb = (cx + dx, cy + dy, cz + dz);
                if visited.contains(&nb) {
                    continue;
                }
                if let Some(neighbor) = self.cells.get(&nb) {
                    if neighbor.sub_blocks & neighbor_face != 0 {
                        visited.insert(nb);
                        queue.push_back(nb);
                    }
                }
            }
        }
        visited
    }

    /// Remove blocks that lost support after cells were destroyed or terrain lowered.
    /// Uses flood-fill from ground-anchored blocks to find truly unsupported cells.
    /// Returns world-space centers of every block that was removed.
    pub fn collapse_unsupported(
        &mut self,
        physics: &mut PhysicsWorld,
        seeds: &[(i32, i32, i32)],
        terrain_height: impl Fn(f32, f32) -> f32,
    ) -> Vec<Vec3> {
        // Gather all cells that might be affected: neighbors of each seed.
        let mut candidates = HashSet::new();
        for &(cx, cy, cz) in seeds {
            for &(dx, dy, dz) in &[(0,1,0),(0,-1,0),(1,0,0),(-1,0,0),(0,0,1),(0,0,-1),(0,0,0)] {
                let nb = (cx + dx, cy + dy, cz + dz);
                if self.is_occupied(nb.0, nb.1, nb.2) {
                    candidates.insert(nb);
                }
            }
        }

        if candidates.is_empty() {
            return Vec::new();
        }

        // Expand candidates to include all cells connected to them, since
        // a disconnection can affect an entire connected structure.
        let mut full_region = HashSet::new();
        for &start in &candidates {
            if full_region.contains(&start) {
                continue;
            }
            let connected = self.flood_connected(start);
            full_region.extend(connected);
        }

        // Find which cells in the region are anchored (supported by terrain
        // or have a solid bottom resting on the ground).
        let mut anchored = HashSet::new();
        let mut queue = VecDeque::new();
        for &(cx, cy, cz) in &full_region {
            let self_subs = self.cells[&(cx, cy, cz)].sub_blocks;
            if self_subs & BOTTOM_LAYER_MASK != 0 {
                let th = terrain_height(cx as f32 + 0.5, cz as f32 + 0.5);
                if th >= cy as f32 {
                    anchored.insert((cx, cy, cz));
                    queue.push_back((cx, cy, cz));
                }
            }
        }

        // Flood-fill from anchored cells through the region.
        while let Some((cx, cy, cz)) = queue.pop_front() {
            let self_subs = match self.cells.get(&(cx, cy, cz)) {
                Some(cell) => cell.sub_blocks,
                None => continue,
            };

            const DIRS: [((i32, i32, i32), u64, u64); 6] = [
                ((0,  1, 0), TOP_LAYER_MASK,    BOTTOM_LAYER_MASK),
                ((0, -1, 0), BOTTOM_LAYER_MASK,  TOP_LAYER_MASK),
                ((1,  0, 0), POS_X_FACE_MASK,    NEG_X_FACE_MASK),
                ((-1, 0, 0), NEG_X_FACE_MASK,    POS_X_FACE_MASK),
                ((0,  0, 1), POS_Z_FACE_MASK,    NEG_Z_FACE_MASK),
                ((0,  0,-1), NEG_Z_FACE_MASK,    POS_Z_FACE_MASK),
            ];

            for &((dx, dy, dz), self_face, neighbor_face) in &DIRS {
                if self_subs & self_face == 0 {
                    continue;
                }
                let nb = (cx + dx, cy + dy, cz + dz);
                if anchored.contains(&nb) || !full_region.contains(&nb) {
                    continue;
                }
                if let Some(neighbor) = self.cells.get(&nb) {
                    if neighbor.sub_blocks & neighbor_face != 0 {
                        anchored.insert(nb);
                        queue.push_back(nb);
                    }
                }
            }
        }

        // Anything in the region that isn't anchored falls.
        let mut fallen = Vec::new();
        for &(cx, cy, cz) in &full_region {
            if !anchored.contains(&(cx, cy, cz)) {
                self.remove(physics, cx, cy, cz);
                fallen.push(cell_center(cx, cy, cz));
            }
        }

        fallen
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
