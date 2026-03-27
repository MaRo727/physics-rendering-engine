use std::collections::HashMap;

use glam::Vec3;
use crate::physics::body::{ColliderHandle, Isometry, NaPoint3, RigidBodyHandle, SharedShape};
use crate::physics::world::PhysicsWorld;
use crate::renderer::mesh::Vertex;

const BUILDING_COLOR: Vec3 = Vec3::new(0.7, 0.7, 0.65);

/// Sub-blocks per axis within each cell (4×4×4 = 64 bits = u64).
const SUBS: i32 = 4;
const SUB_SIZE: f32 = 1.0 / SUBS as f32;
const SUB_HALF: f32 = SUB_SIZE / 2.0;
const ALL_SUBS: u64 = u64::MAX;

/// Radius around a pickaxe hit in which sub-blocks are removed.
const MINE_RADIUS: f32 = 0.35;

// ---------------------------------------------------------------------------
// Block types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BlockType {
    Cube = 0,
    Slab = 1,
    VerticalSlab = 2,
    Slope = 3,
    InnerCornerSlope = 4,
    Stairs = 5,
    Fence = 6,
}

impl BlockType {
    pub fn name(self) -> &'static str {
        match self {
            BlockType::Cube => "Cube",
            BlockType::Slab => "Slab",
            BlockType::VerticalSlab => "VSlab",
            BlockType::Slope => "Slope",
            BlockType::InnerCornerSlope => "Corner",
            BlockType::Stairs => "Stairs",
            BlockType::Fence => "Fence",
        }
    }

    pub fn next(self) -> Self {
        match self {
            BlockType::Cube => BlockType::Slab,
            BlockType::Slab => BlockType::VerticalSlab,
            BlockType::VerticalSlab => BlockType::Slope,
            BlockType::Slope => BlockType::InnerCornerSlope,
            BlockType::InnerCornerSlope => BlockType::Stairs,
            BlockType::Stairs => BlockType::Fence,
            BlockType::Fence => BlockType::Cube,
        }
    }

    /// Returns the renderer mesh ID for this block type's preview shape.
    pub fn mesh_id(self) -> u32 {
        use crate::renderer::*;
        match self {
            BlockType::Cube => MESH_CUBE,
            BlockType::Slab => MESH_BLOCK_SLAB,
            BlockType::VerticalSlab => MESH_BLOCK_VSLAB,
            BlockType::Slope => MESH_BLOCK_SLOPE,
            BlockType::InnerCornerSlope => MESH_BLOCK_INNER_CORNER,
            BlockType::Stairs => MESH_BLOCK_STAIRS,
            BlockType::Fence => MESH_BLOCK_FENCE,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => BlockType::Slab,
            2 => BlockType::VerticalSlab,
            3 => BlockType::Slope,
            4 => BlockType::InnerCornerSlope,
            5 => BlockType::Stairs,
            6 => BlockType::Fence,
            _ => BlockType::Cube,
        }
    }
}

// Face masks for the 4×4×4 sub-block grid.  Bit index = sy*16 + sz*4 + sx.
const TOP_LAYER_MASK: u64    = 0xFFFF_0000_0000_0000; // sy = 3
const THIRD_LAYER_MASK: u64  = 0x0000_FFFF_0000_0000; // sy = 2
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

/// Get the initial sub-block mask for a block type (before rotation).
pub fn initial_sub_blocks(bt: BlockType, rotation: u8) -> u64 {
    match bt {
        BlockType::Cube => ALL_SUBS,
        BlockType::Slab if rotation == 1 => {
            // Top 2 layers (sy=2,3)
            THIRD_LAYER_MASK | TOP_LAYER_MASK
        }
        BlockType::Slab => {
            // Bottom 2 layers (sy=0,1): bits where sy*16 < 32
            BOTTOM_LAYER_MASK | SECOND_LAYER_MASK
        }
        BlockType::VerticalSlab => {
            // Front half: sz=0,1 (all sx, all sy)
            NEG_Z_HALF_MASK
        }
        BlockType::Slope => {
            // Ramp: full at back (sz=3), decreasing toward front
            // sy < 4: all; sy < 3 where sz >= 1; sy < 2 where sz >= 2; sy < 1 where sz >= 3
            // Layer sy=0: all sz (16 bits)
            // Layer sy=1: sz >= 1
            // Layer sy=2: sz >= 2
            // Layer sy=3: sz >= 3 (back column only)
            slope_mask()
        }
        BlockType::InnerCornerSlope => {
            // Concave corner: intersection of two slopes
            inner_corner_slope_mask()
        }
        BlockType::Stairs => {
            // Bottom half full, top half back half only
            // sy=0,1: all sub-blocks
            // sy=2,3: sz=2,3 only
            stairs_mask()
        }
        BlockType::Fence => {
            // Center pillar: sx=1..2, sz=1..2, all y
            fence_mask()
        }
    }
}

// Second Y layer mask: sy=1
const SECOND_LAYER_MASK: u64 = 0x0000_0000_FFFF_0000;
// Front half (sz=0,1) for all sx,sy
const NEG_Z_HALF_MASK: u64 = {
    // sz=0: bits 0..3 per row; sz=1: bits 4..7 per row
    // Per y-layer (16 bits): 0x00FF (sz=0,1 for all sx)
    // 4 layers: 0x00FF_00FF_00FF_00FF
    0x00FF_00FF_00FF_00FFu64
};

fn slope_mask() -> u64 {
    let mut mask = 0u64;
    for sy in 0..4i32 {
        for sz in 0..4i32 {
            // Block is solid where sy <= (sz) — taller at back (high sz)
            // sy=0: all sz; sy=1: sz>=1; sy=2: sz>=2; sy=3: sz>=3
            if sz >= sy {
                for sx in 0..4i32 {
                    mask |= sub_bit(sx, sy, sz);
                }
            }
        }
    }
    mask
}

fn inner_corner_slope_mask() -> u64 {
    // L-shaped concave corner: solid where sy <= sz OR sy <= sx
    // This creates a valley at the corner where both sx and sz are low
    let mut mask = 0u64;
    for sy in 0..4i32 {
        for sz in 0..4i32 {
            for sx in 0..4i32 {
                if sz >= sy || sx >= sy {
                    mask |= sub_bit(sx, sy, sz);
                }
            }
        }
    }
    mask
}

fn stairs_mask() -> u64 {
    let mut mask = 0u64;
    for sy in 0..4i32 {
        for sz in 0..4i32 {
            for sx in 0..4i32 {
                // Bottom half (sy 0,1): always solid
                // Top half (sy 2,3): only back half (sz 2,3)
                if sy < 2 || sz >= 2 {
                    mask |= sub_bit(sx, sy, sz);
                }
            }
        }
    }
    mask
}

fn fence_mask() -> u64 {
    let mut mask = 0u64;
    for sy in 0..4i32 {
        for sx in 1..3i32 {
            for sz in 1..3i32 {
                mask |= sub_bit(sx, sy, sz);
            }
        }
    }
    mask
}

/// Rotate a sub-block bitmask by `rotation` * 90° around Y (clockwise when viewed from above).
pub fn rotate_sub_blocks(mask: u64, rotation: u8) -> u64 {
    if rotation == 0 { return mask; }
    let mut result = 0u64;
    for sy in 0..4i32 {
        for sz in 0..4i32 {
            for sx in 0..4i32 {
                if mask & sub_bit(sx, sy, sz) == 0 { continue; }
                let (nx, nz) = rotate_xz(sx, sz, rotation);
                result |= sub_bit(nx, sy, nz);
            }
        }
    }
    result
}

/// Rotate (sx, sz) in a 4x4 grid by rotation*90° CW.
fn rotate_xz(sx: i32, sz: i32, rotation: u8) -> (i32, i32) {
    match rotation % 4 {
        0 => (sx, sz),
        1 => (3 - sz, sx),       // 90° CW
        2 => (3 - sx, 3 - sz),   // 180°
        3 => (sz, 3 - sx),       // 270° CW
        _ => unreachable!(),
    }
}

/// Build the physics collision shape for a pristine block type.
pub fn build_block_shape(block_type: BlockType, rotation: u8) -> SharedShape {
    match block_type {
        BlockType::Cube => SharedShape::cuboid(0.5, 0.5, 0.5),
        BlockType::Slab => {
            let y_off = if rotation == 1 { 0.25 } else { -0.25 };
            let shapes = vec![(
                Isometry::translation(0.0, y_off, 0.0),
                SharedShape::cuboid(0.5, 0.25, 0.5),
            )];
            SharedShape::compound(shapes)
        }
        BlockType::VerticalSlab => {
            // Half-depth wall at front, rotated
            let shapes = vec![(
                rotate_isometry(Isometry::translation(0.0, 0.0, -0.25), rotation),
                SharedShape::cuboid(0.5, 0.5, 0.25),
            )];
            SharedShape::compound(shapes)
        }
        BlockType::Slope => {
            // Wedge: use convex hull
            let pts = rotate_points(&[
                // Bottom face (y = -0.5)
                [-0.5, -0.5, -0.5],
                [ 0.5, -0.5, -0.5],
                [ 0.5, -0.5,  0.5],
                [-0.5, -0.5,  0.5],
                // Top back edge (y = 0.5, z = 0.5)
                [-0.5,  0.5,  0.5],
                [ 0.5,  0.5,  0.5],
            ], rotation);
            let pts_na: Vec<_> = pts.iter().map(|p| NaPoint3::new(p[0], p[1], p[2])).collect();
            SharedShape::convex_hull(&pts_na).unwrap_or_else(|| SharedShape::cuboid(0.5, 0.5, 0.5))
        }
        BlockType::InnerCornerSlope => {
            // Approximate with compound shapes — two wedges forming L
            // Use sub-block compound for accuracy
            let subs = rotate_sub_blocks(initial_sub_blocks(block_type, rotation), rotation);
            build_compound_shape(subs)
        }
        BlockType::Stairs => {
            // Two cuboids: bottom step + top step
            let shapes = vec![
                (rotate_isometry(Isometry::translation(0.0, -0.25, 0.0), rotation),
                 SharedShape::cuboid(0.5, 0.25, 0.5)),
                (rotate_isometry(Isometry::translation(0.0, 0.25, 0.25), rotation),
                 SharedShape::cuboid(0.5, 0.25, 0.25)),
            ];
            SharedShape::compound(shapes)
        }
        BlockType::Fence => SharedShape::cuboid(0.125, 0.5, 0.125),
    }
}

/// Rotate an isometry around the Y axis by rotation*90° CW.
fn rotate_isometry(iso: Isometry<f32>, rotation: u8) -> Isometry<f32> {
    if rotation == 0 { return iso; }
    let t = iso.translation.vector;
    let (sin, cos) = match rotation % 4 {
        1 => (1.0f32, 0.0f32),
        2 => (0.0, -1.0),
        3 => (-1.0, 0.0),
        _ => return iso,
    };
    let nx = t.x * cos - t.z * sin;
    let nz = t.x * sin + t.z * cos;
    Isometry::translation(nx, t.y, nz)
}

/// Rotate an array of [x,y,z] points around Y by rotation*90° CW.
fn rotate_points(pts: &[[f32; 3]], rotation: u8) -> Vec<[f32; 3]> {
    if rotation == 0 { return pts.to_vec(); }
    let (sin, cos) = match rotation % 4 {
        1 => (1.0f32, 0.0f32),
        2 => (0.0, -1.0),
        3 => (-1.0, 0.0),
        _ => return pts.to_vec(),
    };
    pts.iter().map(|p| {
        let nx = p[0] * cos - p[2] * sin;
        let nz = p[0] * sin + p[2] * cos;
        [nx, p[1], nz]
    }).collect()
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
    block_type: BlockType,
    rotation: u8,
    color: Vec3,
}

// ---------------------------------------------------------------------------
// Baked groups
// ---------------------------------------------------------------------------

/// A baked group — many blocks merged into a single object.
/// One physics body, rendered as a ghost in the editor, destroyed as a unit.
pub struct BakedGroup {
    pub blocks: Vec<crate::blueprint::BlockEntry>,
    rigid_body: Option<RigidBodyHandle>,
    collider: Option<ColliderHandle>,
}

// ---------------------------------------------------------------------------
// Building grid
// ---------------------------------------------------------------------------

/// A grid-based building system. Cubes snap to integer grid positions.
/// Cell (cx, cy, cz) occupies [cx, cx+1] × [cy, cy+1] × [cz, cz+1].
/// Each cell is subdivided into 4×4×4 sub-blocks that can be individually mined.
pub struct BuildingGrid {
    cells: HashMap<(i32, i32, i32), CellData>,
    groups: Vec<BakedGroup>,
    /// Reverse lookup: rigid body handle → index into `groups`.
    group_body_map: HashMap<RigidBodyHandle, usize>,
    dirty: bool,
    /// Temporary preview cells for drag-to-fill (position + color). No physics bodies.
    preview_cells: Vec<((i32, i32, i32), BlockType, u8, Vec3)>,
}

impl BuildingGrid {
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            groups: Vec::new(),
            group_body_map: HashMap::new(),
            dirty: false,
            preview_cells: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty() && self.groups.is_empty()
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn is_occupied(&self, x: i32, y: i32, z: i32) -> bool {
        self.cells.contains_key(&(x, y, z))
    }

    /// Set the drag-to-fill preview (visual only, no physics).
    pub fn set_preview(&mut self, cells: Vec<((i32, i32, i32), BlockType, u8, Vec3)>) {
        if self.preview_cells != cells {
            self.preview_cells = cells;
            self.dirty = true;
        }
    }

    /// Clear the drag preview.
    pub fn clear_preview(&mut self) {
        if !self.preview_cells.is_empty() {
            self.preview_cells.clear();
            self.dirty = true;
        }
    }

    /// Whether the grid has any preview cells (used for empty-check in rendering).
    pub fn has_preview(&self) -> bool {
        !self.preview_cells.is_empty()
    }

    /// Return the sub_blocks mask of a baked group block at the given cell, or 0.
    fn group_sub_blocks_at(&self, cx: i32, cy: i32, cz: i32) -> u64 {
        for group in &self.groups {
            for b in &group.blocks {
                if b.x == cx && b.y == cy && b.z == cz {
                    return b.sub_blocks;
                }
            }
        }
        0
    }

    /// Check if a cell is supported by any neighbor (below or sideways) or terrain.
    /// `terrain_height` is the terrain surface height at the cell's center XZ.
    /// For cells that don't exist yet (placement check), uses the given block type's mask.
    pub fn is_supported_with(&self, cx: i32, cy: i32, cz: i32, terrain_height: f32,
                              block_type: BlockType, rotation: u8) -> bool {
        // Sub-blocks of this cell (if it doesn't exist yet, use the block type's initial mask).
        let self_subs = match self.cells.get(&(cx, cy, cz)) {
            Some(cell) => cell.sub_blocks,
            None => rotate_sub_blocks(initial_sub_blocks(block_type, rotation), rotation),
        };

        // Check each neighbor: below + 4 horizontal sides.
        for &((dx, dy, dz), self_face, neighbor_face) in &SUPPORT_NEIGHBORS {
            // This cell must have solid sub-blocks on the connecting face.
            if self_subs & self_face == 0 {
                continue;
            }
            let nx = cx + dx;
            let ny = cy + dy;
            let nz = cz + dz;
            // Check loose cells.
            if let Some(neighbor) = self.cells.get(&(nx, ny, nz)) {
                if neighbor.sub_blocks & neighbor_face != 0 {
                    return true;
                }
            }
            // Check baked group blocks.
            if self.group_sub_blocks_at(nx, ny, nz) & neighbor_face != 0 {
                return true;
            }
        }

        // Supported if the terrain surface reaches the bottom of this cell.
        if self_subs & BOTTOM_LAYER_MASK != 0 && terrain_height >= cy as f32 {
            return true;
        }

        false
    }

    /// Place a block at grid position (cx, cy, cz). Returns true if placed.
    /// The block must be supported from below (another block or terrain).
    pub fn place(&mut self, physics: &mut PhysicsWorld, cx: i32, cy: i32, cz: i32,
                 terrain_height: f32, block_type: BlockType, rotation: u8) -> bool {
        self.place_colored(physics, cx, cy, cz, terrain_height, block_type, rotation, BUILDING_COLOR)
    }

    /// Place a block with a specific color. Returns true if placed.
    pub fn place_colored(&mut self, physics: &mut PhysicsWorld, cx: i32, cy: i32, cz: i32,
                         terrain_height: f32, block_type: BlockType, rotation: u8, color: Vec3) -> bool {
        if self.is_occupied(cx, cy, cz) {
            return false;
        }
        if !self.is_supported_with(cx, cy, cz, terrain_height, block_type, rotation) {
            return false;
        }

        let sub_blocks = rotate_sub_blocks(initial_sub_blocks(block_type, rotation), rotation);
        let center = cell_center(cx, cy, cz);
        let shape = build_block_shape(block_type, rotation);
        let (rigid_body, collider) = physics.add_static_shape(center, shape);

        self.cells.insert(
            (cx, cy, cz),
            CellData {
                rigid_body,
                collider,
                sub_blocks,
                block_type,
                rotation,
                color,
            },
        );
        self.dirty = true;
        true
    }

    /// Place a block without support checks (for editor mode).
    pub fn place_unsupported(&mut self, physics: &mut PhysicsWorld, cx: i32, cy: i32, cz: i32,
                              block_type: BlockType, rotation: u8, color: Vec3) -> bool {
        if self.is_occupied(cx, cy, cz) {
            return false;
        }

        let sub_blocks = rotate_sub_blocks(initial_sub_blocks(block_type, rotation), rotation);
        let center = cell_center(cx, cy, cz);
        let shape = build_block_shape(block_type, rotation);
        let (rigid_body, collider) = physics.add_static_shape(center, shape);

        self.cells.insert(
            (cx, cy, cz),
            CellData {
                rigid_body,
                collider,
                sub_blocks,
                block_type,
                rotation,
                color,
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

    /// Check if a rigid body belongs to a building cell or baked group.
    pub fn has_body(&self, rb: RigidBodyHandle) -> bool {
        self.cells.values().any(|c| c.rigid_body == rb)
            || self.groups.iter().any(|g| g.rigid_body == Some(rb))
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

    /// Remove the single sub-block closest to `hit_pos`. Returns the affected
    /// cell coordinate (if any) so the caller can check for collapse.
    pub fn chisel_at(&mut self, physics: &mut PhysicsWorld, hit_pos: Vec3) -> Vec<(i32, i32, i32)> {
        // Find which cell the hit landed in.
        let cx = hit_pos.x.floor() as i32;
        let cy = hit_pos.y.floor() as i32;
        let cz = hit_pos.z.floor() as i32;

        // Search the hit cell and its immediate neighbors (hit may land on a
        // boundary) for the closest existing sub-block.
        let mut best: Option<((i32, i32, i32), u64, f32)> = None; // (cell, bit, dist_sq)

        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(cell) = self.cells.get(&key) {
                        for sy in 0..SUBS {
                            for sz in 0..SUBS {
                                for sx in 0..SUBS {
                                    let bit = sub_bit(sx, sy, sz);
                                    if cell.sub_blocks & bit == 0 {
                                        continue;
                                    }
                                    let center = sub_world_pos(key.0, key.1, key.2, sx, sy, sz);
                                    let dist_sq = (center - hit_pos).length_squared();
                                    if best.map_or(true, |(_, _, d)| dist_sq < d) {
                                        best = Some((key, bit, dist_sq));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let (key, bit, _) = match best {
            Some(b) => b,
            None => return Vec::new(),
        };

        let cell = self.cells.get_mut(&key).unwrap();
        cell.sub_blocks &= !bit;

        if cell.sub_blocks == 0 {
            let cell = self.cells.remove(&key).unwrap();
            physics.remove_body(cell.rigid_body, cell.collider);
        } else {
            let shape = build_compound_shape(cell.sub_blocks);
            cell.collider = physics.replace_collider(cell.rigid_body, cell.collider, shape);
        }

        self.dirty = true;
        vec![key]
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
    /// Pristine (unmined) blocks use clean geometric meshes; mined blocks iterate sub-blocks.
    /// Baked groups are also included (with full color in world, ghost tint in editor).
    pub fn generate_mesh(&self) -> (Vec<Vertex>, Vec<u32>) {
        self.generate_mesh_inner(false, None)
    }

    /// Generate mesh with ghost tint on baked groups (for editor).
    /// The selected group (if any) is highlighted brighter.
    pub fn generate_mesh_editor(&self, selected_group: Option<usize>) -> (Vec<Vertex>, Vec<u32>) {
        self.generate_mesh_inner(true, selected_group)
    }

    fn generate_mesh_inner(&self, ghost_groups: bool, selected_group: Option<usize>) -> (Vec<Vertex>, Vec<u32>) {
        let estimate = self.cells.len() * 24;
        let mut vertices = Vec::with_capacity(estimate);
        let mut indices = Vec::with_capacity(estimate);

        for (&(cx, cy, cz), cell) in &self.cells {
            let pristine_mask = rotate_sub_blocks(initial_sub_blocks(cell.block_type, cell.rotation), cell.rotation);
            if cell.sub_blocks == pristine_mask {
                // Clean mesh path — emit the geometric shape with neighbor face culling.
                emit_block_mesh(
                    cell.block_type, cell.rotation, cx, cy, cz,
                    &self.cells, self, cell.color, &mut vertices, &mut indices,
                );
            } else {
                // Mined — fall back to sub-block iteration.
                self.emit_sub_block_mesh(cx, cy, cz, cell, &mut vertices, &mut indices);
            }
        }

        // Append drag-to-fill preview blocks (tinted, no neighbor culling).
        for &((cx, cy, cz), block_type, rotation, color) in &self.preview_cells {
            if !self.cells.contains_key(&(cx, cy, cz)) {
                let preview_color = color * 0.55 + Vec3::splat(0.25);
                emit_block_mesh(
                    block_type, rotation, cx, cy, cz,
                    &self.cells, self, preview_color,
                    &mut vertices, &mut indices,
                );
            }
        }

        // Append baked group meshes.
        self.generate_group_meshes_with_selection(&mut vertices, &mut indices, ghost_groups, selected_group);

        (vertices, indices)
    }

    /// Emit sub-block based mesh for a mined cell (existing behavior).
    fn emit_sub_block_mesh(
        &self, cx: i32, cy: i32, cz: i32, cell: &CellData,
        vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
    ) {
        let s = SUB_SIZE;
        let ext_color = cell.color;
        let int_color = cell.color * 0.78;
        let pristine_mask = rotate_sub_blocks(initial_sub_blocks(cell.block_type, cell.rotation), cell.rotation);
        for sy in 0..SUBS {
            for sz in 0..SUBS {
                for sx in 0..SUBS {
                    if !has_sub(cell.sub_blocks, sx, sy, sz) {
                        continue;
                    }

                    let x = cx as f32 + sx as f32 * s;
                    let y = cy as f32 + sy as f32 * s;
                    let z = cz as f32 + sz as f32 * s;

                    let is_interior = cell.sub_blocks != pristine_mask;

                    // +X face
                    if !self.is_solid(cx, cy, cz, sx + 1, sy, sz) {
                        let on_edge = sx == SUBS - 1;
                        let color = if on_edge && !is_interior { ext_color } else { int_color };
                        push_quad(
                            vertices, indices,
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
                        let color = if on_edge && !is_interior { ext_color } else { int_color };
                        push_quad(
                            vertices, indices,
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
                        let color = if on_edge && !is_interior { ext_color } else { int_color };
                        push_quad(
                            vertices, indices,
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
                        let color = if on_edge && !is_interior { ext_color } else { int_color };
                        push_quad(
                            vertices, indices,
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
                        let color = if on_edge && !is_interior { ext_color } else { int_color };
                        push_quad(
                            vertices, indices,
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
                        let color = if on_edge && !is_interior { ext_color } else { int_color };
                        push_quad(
                            vertices, indices,
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

    /// Get block type, rotation, sub_blocks, and color for a cell (used by save).
    pub fn cell_info(&self, cx: i32, cy: i32, cz: i32) -> Option<(BlockType, u8, u64, Vec3)> {
        self.cells.get(&(cx, cy, cz)).map(|c| (c.block_type, c.rotation, c.sub_blocks, c.color))
    }

    /// Return iterator over all occupied cell coordinates.
    pub fn occupied_cells(&self) -> impl Iterator<Item = &(i32, i32, i32)> {
        self.cells.keys()
    }

    /// Number of occupied cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Load a cell with full data (used by save/load).
    pub fn load_cell(&mut self, physics: &mut PhysicsWorld,
                     cx: i32, cy: i32, cz: i32,
                     block_type: BlockType, rotation: u8, sub_blocks: u64, color: Vec3) {
        let center = cell_center(cx, cy, cz);
        let pristine = rotate_sub_blocks(initial_sub_blocks(block_type, rotation), rotation);
        let shape = if sub_blocks == pristine {
            build_block_shape(block_type, rotation)
        } else {
            build_compound_shape(sub_blocks)
        };
        let (rigid_body, collider) = physics.add_static_shape(center, shape);
        self.cells.insert((cx, cy, cz), CellData {
            rigid_body, collider, sub_blocks, block_type, rotation, color,
        });
        self.dirty = true;
    }

    /// Remove all cells and physics bodies.
    pub fn clear(&mut self, physics: &mut PhysicsWorld) {
        let keys: Vec<_> = self.cells.keys().copied().collect();
        for (x, y, z) in keys {
            if let Some(cell) = self.cells.remove(&(x, y, z)) {
                physics.remove_body(cell.rigid_body, cell.collider);
            }
        }
        self.clear_groups(physics);
        self.dirty = true;
    }

    // -------------------------------------------------------------------
    // Baked group methods
    // -------------------------------------------------------------------

    /// Bake all current cells into a new group. Removes individual cell physics
    /// bodies and creates one compound body for the group. Returns false if empty.
    pub fn bake_group(&mut self, physics: &mut PhysicsWorld) -> bool {
        if self.cells.is_empty() {
            return false;
        }

        // Collect block entries from current cells.
        let blocks: Vec<crate::blueprint::BlockEntry> = self.cells.keys().map(|&(x, y, z)| {
            let c = &self.cells[&(x, y, z)];
            crate::blueprint::BlockEntry {
                x, y, z,
                block_type: c.block_type as u8,
                rotation: c.rotation,
                sub_blocks: c.sub_blocks,
                color: [c.color.x, c.color.y, c.color.z],
            }
        }).collect();

        // Remove individual physics bodies.
        let keys: Vec<_> = self.cells.keys().copied().collect();
        for (x, y, z) in keys {
            if let Some(cell) = self.cells.remove(&(x, y, z)) {
                physics.remove_body(cell.rigid_body, cell.collider);
            }
        }

        // Create compound physics body for the group.
        let (rb, col) = build_group_physics(physics, &blocks);

        let idx = self.groups.len();
        self.groups.push(BakedGroup {
            blocks,
            rigid_body: Some(rb),
            collider: Some(col),
        });
        self.group_body_map.insert(rb, idx);
        self.dirty = true;
        true
    }

    /// Add a pre-built group (from blueprint load). Creates compound physics.
    pub fn load_group(&mut self, physics: &mut PhysicsWorld, blocks: Vec<crate::blueprint::BlockEntry>) {
        let (rb, col) = build_group_physics(physics, &blocks);
        let idx = self.groups.len();
        self.groups.push(BakedGroup {
            blocks,
            rigid_body: Some(rb),
            collider: Some(col),
        });
        self.group_body_map.insert(rb, idx);
        self.dirty = true;
    }

    /// Add a pre-built group with a world offset applied to each block position.
    pub fn load_group_offset(&mut self, physics: &mut PhysicsWorld,
                              blocks: &[crate::blueprint::BlockEntry],
                              ox: i32, oy: i32, oz: i32) {
        let offset_blocks: Vec<crate::blueprint::BlockEntry> = blocks.iter().map(|b| {
            crate::blueprint::BlockEntry {
                x: b.x + ox,
                y: b.y + oy,
                z: b.z + oz,
                block_type: b.block_type,
                rotation: b.rotation,
                sub_blocks: b.sub_blocks,
                color: b.color,
            }
        }).collect();
        self.load_group(physics, offset_blocks);
    }

    /// Check if a rigid body belongs to a baked group. Returns group index if found.
    pub fn group_for_body(&self, rb: RigidBodyHandle) -> Option<usize> {
        self.group_body_map.get(&rb).copied()
    }

    /// Rebuild the group_body_map after a removal shifts indices.
    fn rebuild_group_body_map(&mut self) {
        self.group_body_map.clear();
        for (i, g) in self.groups.iter().enumerate() {
            if let Some(rb) = g.rigid_body {
                self.group_body_map.insert(rb, i);
            }
        }
    }

    /// Unbake a group back into individual editable cells. Returns number of blocks restored.
    pub fn unbake_group(&mut self, physics: &mut PhysicsWorld, group_idx: usize) -> usize {
        let group = self.groups.remove(group_idx);
        // Remove the compound physics body.
        if let (Some(rb), Some(col)) = (group.rigid_body, group.collider) {
            self.group_body_map.remove(&rb);
            physics.remove_body(rb, col);
        }
        self.rebuild_group_body_map();
        // Restore blocks as individual cells.
        let count = group.blocks.len();
        for b in &group.blocks {
            let bt = BlockType::from_u8(b.block_type);
            let color = Vec3::new(b.color[0], b.color[1], b.color[2]);
            self.load_cell(physics, b.x, b.y, b.z, bt, b.rotation, b.sub_blocks, color);
        }
        self.dirty = true;
        count
    }

    /// Destroy an entire baked group by index, removing its physics body.
    pub fn destroy_group(&mut self, physics: &mut PhysicsWorld, group_idx: usize) {
        let group = self.groups.remove(group_idx);
        if let (Some(rb), Some(col)) = (group.rigid_body, group.collider) {
            self.group_body_map.remove(&rb);
            physics.remove_body(rb, col);
        }
        self.rebuild_group_body_map();
        self.dirty = true;
    }

    /// Number of baked groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Access baked groups (for save).
    pub fn groups(&self) -> &[BakedGroup] {
        &self.groups
    }

    /// Remove all baked groups and their physics.
    fn clear_groups(&mut self, physics: &mut PhysicsWorld) {
        for group in self.groups.drain(..) {
            if let (Some(rb), Some(col)) = (group.rigid_body, group.collider) {
                physics.remove_body(rb, col);
            }
        }
    }

    /// Generate mesh for baked groups. In ghost mode, groups are dimmed;
    /// the selected group (if any) is highlighted brighter.
    fn generate_group_meshes_with_selection(
        &self, vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>,
        ghost: bool, selected: Option<usize>,
    ) {
        for (gi, group) in self.groups.iter().enumerate() {
            // Determine tint: ghost mode dims groups, selected group is brighter.
            let tint = if ghost {
                if selected == Some(gi) { 0.85 } else { 0.45 }
            } else {
                1.0
            };

            // Build a temporary cell map for neighbor culling within the group.
            let mut group_cells: HashMap<(i32, i32, i32), (u64, BlockType, u8, Vec3)> = HashMap::new();
            for b in &group.blocks {
                let bt = BlockType::from_u8(b.block_type);
                let color = Vec3::new(b.color[0], b.color[1], b.color[2]);
                group_cells.insert((b.x, b.y, b.z), (b.sub_blocks, bt, b.rotation, color));
            }

            for b in &group.blocks {
                let bt = BlockType::from_u8(b.block_type);
                let color = Vec3::new(b.color[0], b.color[1], b.color[2]) * tint;
                let pristine = rotate_sub_blocks(initial_sub_blocks(bt, b.rotation), b.rotation);

                if b.sub_blocks == pristine {
                    emit_block_mesh_with_cells(
                        bt, b.rotation, b.x, b.y, b.z,
                        &group_cells, color, vertices, indices,
                    );
                } else {
                    emit_sub_block_mesh_standalone(
                        b.x, b.y, b.z, b.sub_blocks, bt, b.rotation, color,
                        &group_cells, vertices, indices,
                    );
                }
            }
        }
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

/// Build a single compound physics body for a baked group.
fn build_group_physics(physics: &mut PhysicsWorld, blocks: &[crate::blueprint::BlockEntry]) -> (RigidBodyHandle, ColliderHandle) {
    // Compute centroid for the rigid body position.
    let n = blocks.len() as f32;
    let cx = blocks.iter().map(|b| b.x as f32 + 0.5).sum::<f32>() / n;
    let cy = blocks.iter().map(|b| b.y as f32 + 0.5).sum::<f32>() / n;
    let cz = blocks.iter().map(|b| b.z as f32 + 0.5).sum::<f32>() / n;
    let center = Vec3::new(cx, cy, cz);

    // Build compound shape with one sub-shape per block, offset from centroid.
    let mut shapes = Vec::new();
    for b in blocks {
        let bx = b.x as f32 + 0.5 - cx;
        let by = b.y as f32 + 0.5 - cy;
        let bz = b.z as f32 + 0.5 - cz;

        let bt = BlockType::from_u8(b.block_type);
        let pristine = rotate_sub_blocks(initial_sub_blocks(bt, b.rotation), b.rotation);

        if b.sub_blocks == pristine {
            // Use optimized block shape.
            let block_shape = build_block_shape(bt, b.rotation);
            // The block shape may itself be compound — extract its children.
            if let Some(compound) = block_shape.as_compound() {
                for (child_iso, child_shape) in compound.shapes() {
                    let mut iso = child_iso.clone();
                    iso.translation.vector.x += bx;
                    iso.translation.vector.y += by;
                    iso.translation.vector.z += bz;
                    shapes.push((iso, child_shape.clone()));
                }
            } else {
                shapes.push((Isometry::translation(bx, by, bz), block_shape));
            }
        } else {
            // Mined block — add remaining sub-blocks.
            for sy in 0..SUBS {
                for sz in 0..SUBS {
                    for sx in 0..SUBS {
                        if has_sub(b.sub_blocks, sx, sy, sz) {
                            let iso = Isometry::translation(
                                bx + (sx as f32 + 0.5) * SUB_SIZE - 0.5,
                                by + (sy as f32 + 0.5) * SUB_SIZE - 0.5,
                                bz + (sz as f32 + 0.5) * SUB_SIZE - 0.5,
                            );
                            shapes.push((iso, SharedShape::cuboid(SUB_HALF, SUB_HALF, SUB_HALF)));
                        }
                    }
                }
            }
        }
    }

    let compound = SharedShape::compound(shapes);
    physics.add_static_shape(center, compound)
}

/// Emit block mesh using a group-local cell map for neighbor culling.
fn emit_block_mesh_with_cells(
    block_type: BlockType, rotation: u8,
    cx: i32, cy: i32, cz: i32,
    group_cells: &HashMap<(i32, i32, i32), (u64, BlockType, u8, Vec3)>,
    color: Vec3,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) {
    // Helper: check if neighbor face is fully solid within the group.
    let is_face_solid = |nx: i32, ny: i32, nz: i32, face_mask: u64| -> bool {
        group_cells.get(&(nx, ny, nz))
            .map_or(false, |(subs, _, _, _)| subs & face_mask == face_mask)
    };

    let bx = cx as f32;
    let by = cy as f32;
    let bz = cz as f32;

    match block_type {
        BlockType::Cube => {
            if !is_face_solid(cx, cy + 1, cz, BOTTOM_LAYER_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx, by + 1.0, bz + 1.0), Vec3::new(bx + 1.0, by + 1.0, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz), Vec3::new(bx, by + 1.0, bz),
                    Vec3::Y, color);
            }
            if !is_face_solid(cx, cy - 1, cz, TOP_LAYER_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx, by, bz), Vec3::new(bx + 1.0, by, bz),
                    Vec3::new(bx + 1.0, by, bz + 1.0), Vec3::new(bx, by, bz + 1.0),
                    Vec3::NEG_Y, color);
            }
            if !is_face_solid(cx + 1, cy, cz, NEG_X_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx + 1.0, by, bz + 1.0), Vec3::new(bx + 1.0, by + 1.0, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz), Vec3::new(bx + 1.0, by, bz),
                    Vec3::X, color);
            }
            if !is_face_solid(cx - 1, cy, cz, POS_X_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx, by, bz), Vec3::new(bx, by + 1.0, bz),
                    Vec3::new(bx, by + 1.0, bz + 1.0), Vec3::new(bx, by, bz + 1.0),
                    Vec3::NEG_X, color);
            }
            if !is_face_solid(cx, cy, cz + 1, NEG_Z_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx, by, bz + 1.0), Vec3::new(bx + 1.0, by, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz + 1.0), Vec3::new(bx, by + 1.0, bz + 1.0),
                    Vec3::Z, color);
            }
            if !is_face_solid(cx, cy, cz - 1, POS_Z_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx + 1.0, by, bz), Vec3::new(bx, by, bz),
                    Vec3::new(bx, by + 1.0, bz), Vec3::new(bx + 1.0, by + 1.0, bz),
                    Vec3::NEG_Z, color);
            }
        }
        _ => {
            // For non-cube block types in groups, fall back to sub-block mesh
            // to keep the implementation simple (slopes, slabs, etc. are rare in groups).
            let subs = rotate_sub_blocks(initial_sub_blocks(block_type, rotation), rotation);
            emit_sub_block_mesh_standalone(
                cx, cy, cz, subs, block_type, rotation, color,
                group_cells, vertices, indices,
            );
        }
    }
}

/// Emit sub-block mesh for a single cell, using a group cell map for visibility.
fn emit_sub_block_mesh_standalone(
    cx: i32, cy: i32, cz: i32,
    sub_blocks: u64,
    block_type: BlockType, rotation: u8,
    color: Vec3,
    group_cells: &HashMap<(i32, i32, i32), (u64, BlockType, u8, Vec3)>,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) {
    let s = SUB_SIZE;
    let pristine_mask = rotate_sub_blocks(initial_sub_blocks(block_type, rotation), rotation);
    let is_interior = sub_blocks != pristine_mask;
    let ext_color = color;
    let int_color = color * 0.78;

    // Check if a sub-block is solid, handling cross-cell boundaries within the group.
    let is_solid_group = |cx: i32, cy: i32, cz: i32, sx: i32, sy: i32, sz: i32| -> bool {
        let (cx, sx) = wrap(cx, sx);
        let (cy, sy) = wrap(cy, sy);
        let (cz, sz) = wrap(cz, sz);
        match group_cells.get(&(cx, cy, cz)) {
            Some((subs, _, _, _)) => has_sub(*subs, sx, sy, sz),
            None => false,
        }
    };

    for sy in 0..SUBS {
        for sz in 0..SUBS {
            for sx in 0..SUBS {
                if !has_sub(sub_blocks, sx, sy, sz) {
                    continue;
                }
                let x = cx as f32 + sx as f32 * s;
                let y = cy as f32 + sy as f32 * s;
                let z = cz as f32 + sz as f32 * s;

                if !is_solid_group(cx, cy, cz, sx + 1, sy, sz) {
                    let on_edge = sx == SUBS - 1;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x + s, y, z + s), Vec3::new(x + s, y + s, z + s),
                        Vec3::new(x + s, y + s, z), Vec3::new(x + s, y, z),
                        Vec3::X, c);
                }
                if !is_solid_group(cx, cy, cz, sx - 1, sy, sz) {
                    let on_edge = sx == 0;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x, y, z), Vec3::new(x, y + s, z),
                        Vec3::new(x, y + s, z + s), Vec3::new(x, y, z + s),
                        Vec3::NEG_X, c);
                }
                if !is_solid_group(cx, cy, cz, sx, sy + 1, sz) {
                    let on_edge = sy == SUBS - 1;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x, y + s, z + s), Vec3::new(x + s, y + s, z + s),
                        Vec3::new(x + s, y + s, z), Vec3::new(x, y + s, z),
                        Vec3::Y, c);
                }
                if !is_solid_group(cx, cy, cz, sx, sy - 1, sz) {
                    let on_edge = sy == 0;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x, y, z), Vec3::new(x + s, y, z),
                        Vec3::new(x + s, y, z + s), Vec3::new(x, y, z + s),
                        Vec3::NEG_Y, c);
                }
                if !is_solid_group(cx, cy, cz, sx, sy, sz + 1) {
                    let on_edge = sz == SUBS - 1;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x, y, z + s), Vec3::new(x + s, y, z + s),
                        Vec3::new(x + s, y + s, z + s), Vec3::new(x, y + s, z + s),
                        Vec3::Z, c);
                }
                if !is_solid_group(cx, cy, cz, sx, sy, sz - 1) {
                    let on_edge = sz == 0;
                    let c = if on_edge && !is_interior { ext_color } else { int_color };
                    push_quad(vertices, indices,
                        Vec3::new(x + s, y, z), Vec3::new(x, y, z),
                        Vec3::new(x, y + s, z), Vec3::new(x + s, y + s, z),
                        Vec3::NEG_Z, c);
                }
            }
        }
    }
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

fn push_tri(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    a: Vec3, b: Vec3, c: Vec3,
    normal: Vec3,
    color: Vec3,
) {
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: a, normal, color });
    vertices.push(Vertex { position: b, normal, color });
    vertices.push(Vertex { position: c, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2]);
}

/// Rotate a vertex position around cell center (0.5, y, 0.5) by rotation*90° CW.
fn rotate_vert(v: Vec3, center: Vec3, rotation: u8) -> Vec3 {
    if rotation == 0 { return v; }
    let dx = v.x - center.x;
    let dz = v.z - center.z;
    let (nx, nz) = match rotation % 4 {
        1 => (-dz, dx),
        2 => (-dx, -dz),
        3 => (dz, -dx),
        _ => (dx, dz),
    };
    Vec3::new(center.x + nx, v.y, center.z + nz)
}

/// Rotate a normal vector around Y by rotation*90° CW.
fn rotate_normal(n: Vec3, rotation: u8) -> Vec3 {
    if rotation == 0 { return n; }
    match rotation % 4 {
        1 => Vec3::new(-n.z, n.y, n.x),
        2 => Vec3::new(-n.x, n.y, -n.z),
        3 => Vec3::new(n.z, n.y, -n.x),
        _ => n,
    }
}

/// Check if a full face of a neighbor cell is solid (all sub-blocks on that face present).
fn is_neighbor_face_solid(
    cells: &HashMap<(i32, i32, i32), CellData>,
    cx: i32, cy: i32, cz: i32,
    face_mask: u64,
) -> bool {
    cells.get(&(cx, cy, cz))
        .map_or(false, |c| c.sub_blocks & face_mask == face_mask)
}

/// Emit a clean geometric mesh for a pristine (unmined) block.
fn emit_block_mesh(
    block_type: BlockType, rotation: u8,
    cx: i32, cy: i32, cz: i32,
    cells: &HashMap<(i32, i32, i32), CellData>,
    grid: &BuildingGrid,
    color: Vec3,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
) {
    let bx = cx as f32;
    let by = cy as f32;
    let bz = cz as f32;
    let center = Vec3::new(bx + 0.5, by + 0.5, bz + 0.5);

    match block_type {
        BlockType::Cube => {
            // 6 quads with neighbor culling
            // +Y
            if !is_neighbor_face_solid(cells, cx, cy + 1, cz, BOTTOM_LAYER_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx,     by + 1.0, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz),
                    Vec3::new(bx,     by + 1.0, bz),
                    Vec3::Y, color);
            }
            // -Y
            if !is_neighbor_face_solid(cells, cx, cy - 1, cz, TOP_LAYER_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx,     by, bz),
                    Vec3::new(bx + 1.0, by, bz),
                    Vec3::new(bx + 1.0, by, bz + 1.0),
                    Vec3::new(bx,     by, bz + 1.0),
                    Vec3::NEG_Y, color);
            }
            // +X
            if !is_neighbor_face_solid(cells, cx + 1, cy, cz, NEG_X_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx + 1.0, by,     bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz),
                    Vec3::new(bx + 1.0, by,     bz),
                    Vec3::X, color);
            }
            // -X
            if !is_neighbor_face_solid(cells, cx - 1, cy, cz, POS_X_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx, by,     bz),
                    Vec3::new(bx, by + 1.0, bz),
                    Vec3::new(bx, by + 1.0, bz + 1.0),
                    Vec3::new(bx, by,     bz + 1.0),
                    Vec3::NEG_X, color);
            }
            // +Z
            if !is_neighbor_face_solid(cells, cx, cy, cz + 1, NEG_Z_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx,     by,     bz + 1.0),
                    Vec3::new(bx + 1.0, by,     bz + 1.0),
                    Vec3::new(bx + 1.0, by + 1.0, bz + 1.0),
                    Vec3::new(bx,     by + 1.0, bz + 1.0),
                    Vec3::Z, color);
            }
            // -Z
            if !is_neighbor_face_solid(cells, cx, cy, cz - 1, POS_Z_FACE_MASK) {
                push_quad(vertices, indices,
                    Vec3::new(bx + 1.0, by,     bz),
                    Vec3::new(bx,     by,     bz),
                    Vec3::new(bx,     by + 1.0, bz),
                    Vec3::new(bx + 1.0, by + 1.0, bz),
                    Vec3::NEG_Z, color);
            }
        }
        BlockType::Slab => {
            // rotation 0 = bottom slab [by, by+0.5], rotation 1 = top slab [by+0.5, by+1.0]
            let h = 0.5;
            let y_lo = if rotation == 1 { by + h } else { by };
            let y_hi = y_lo + h;
            // Slab is horizontally symmetric so no XZ rotation needed
            // Top face
            if rotation == 1 {
                if !neighbor_face_solid_check(grid, cx, cy, cz, 0, 1, 0, 0) {
                    push_quad(vertices, indices,
                        Vec3::new(bx,     y_hi, bz + 1.0),
                        Vec3::new(bx + 1.0, y_hi, bz + 1.0),
                        Vec3::new(bx + 1.0, y_hi, bz),
                        Vec3::new(bx,     y_hi, bz),
                        Vec3::Y, color);
                }
            } else {
                push_quad(vertices, indices,
                    Vec3::new(bx,     y_hi, bz + 1.0),
                    Vec3::new(bx + 1.0, y_hi, bz + 1.0),
                    Vec3::new(bx + 1.0, y_hi, bz),
                    Vec3::new(bx,     y_hi, bz),
                    Vec3::Y, color);
            }
            // Bottom face
            if rotation == 0 {
                if !neighbor_face_solid_check(grid, cx, cy, cz, 0, -1, 0, 0) {
                    push_quad(vertices, indices,
                        Vec3::new(bx,     y_lo, bz),
                        Vec3::new(bx + 1.0, y_lo, bz),
                        Vec3::new(bx + 1.0, y_lo, bz + 1.0),
                        Vec3::new(bx,     y_lo, bz + 1.0),
                        Vec3::NEG_Y, color);
                }
            } else {
                push_quad(vertices, indices,
                    Vec3::new(bx,     y_lo, bz),
                    Vec3::new(bx + 1.0, y_lo, bz),
                    Vec3::new(bx + 1.0, y_lo, bz + 1.0),
                    Vec3::new(bx,     y_lo, bz + 1.0),
                    Vec3::NEG_Y, color);
            }
            // Front face (-Z)
            push_quad(vertices, indices,
                Vec3::new(bx + 1.0, y_lo, bz),
                Vec3::new(bx,     y_lo, bz),
                Vec3::new(bx,     y_hi, bz),
                Vec3::new(bx + 1.0, y_hi, bz),
                Vec3::NEG_Z, color);
            // Back face (+Z)
            push_quad(vertices, indices,
                Vec3::new(bx,     y_lo, bz + 1.0),
                Vec3::new(bx + 1.0, y_lo, bz + 1.0),
                Vec3::new(bx + 1.0, y_hi, bz + 1.0),
                Vec3::new(bx,     y_hi, bz + 1.0),
                Vec3::Z, color);
            // Left face (-X)
            push_quad(vertices, indices,
                Vec3::new(bx, y_lo,     bz),
                Vec3::new(bx, y_hi, bz),
                Vec3::new(bx, y_hi, bz + 1.0),
                Vec3::new(bx, y_lo,     bz + 1.0),
                Vec3::NEG_X, color);
            // Right face (+X)
            push_quad(vertices, indices,
                Vec3::new(bx + 1.0, y_lo,     bz + 1.0),
                Vec3::new(bx + 1.0, y_hi, bz + 1.0),
                Vec3::new(bx + 1.0, y_hi, bz),
                Vec3::new(bx + 1.0, y_lo,     bz),
                Vec3::X, color);
        }
        BlockType::VerticalSlab => {
            // Front half wall: z range [bz, bz+0.5], full height
            let d = 0.5;
            let rv = |v: Vec3| rotate_vert(v, center, rotation);
            let rn = |n: Vec3| rotate_normal(n, rotation);
            // Front face (-Z)
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rv(Vec3::new(bx,     by,     bz)),
                rv(Vec3::new(bx,     by + 1.0, bz)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz)),
                rn(Vec3::NEG_Z), color);
            // Back face (at z+0.5)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by,     bz + d)),
                rv(Vec3::new(bx + 1.0, by,     bz + d)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + d)),
                rv(Vec3::new(bx,     by + 1.0, bz + d)),
                rn(Vec3::Z), color);
            // Top
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by + 1.0, bz + d)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + d)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz)),
                rv(Vec3::new(bx,     by + 1.0, bz)),
                rn(Vec3::Y), color);
            // Bottom
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz + d)),
                rv(Vec3::new(bx,     by, bz + d)),
                rn(Vec3::NEG_Y), color);
            // Left (-X)
            push_quad(vertices, indices,
                rv(Vec3::new(bx, by,     bz)),
                rv(Vec3::new(bx, by + 1.0, bz)),
                rv(Vec3::new(bx, by + 1.0, bz + d)),
                rv(Vec3::new(bx, by,     bz + d)),
                rn(Vec3::NEG_X), color);
            // Right (+X)
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by,     bz + d)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + d)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rn(Vec3::X), color);
        }
        BlockType::Slope => {
            // Wedge: full at back (+Z), zero at front top
            // Bottom, back, left, right quads + sloped top surface
            let rv = |v: Vec3| rotate_vert(v, center, rotation);
            let rn = |n: Vec3| rotate_normal(n, rotation);

            // Bottom face
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz + 1.0)),
                rv(Vec3::new(bx,     by, bz + 1.0)),
                rn(Vec3::NEG_Y), color);
            // Back face (+Z, full height)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rn(Vec3::Z), color);
            // Left face (-X, triangle)
            push_tri(vertices, indices,
                rv(Vec3::new(bx, by,     bz)),
                rv(Vec3::new(bx, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx, by,     bz + 1.0)),
                rn(Vec3::NEG_X), color);
            // Right face (+X, triangle)
            push_tri(vertices, indices,
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rn(Vec3::X), color);
            // Sloped top face
            let slope_normal = Vec3::new(0.0, 1.0, -1.0).normalize();
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by,     bz)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rn(slope_normal), color);
        }
        BlockType::InnerCornerSlope => {
            // L-shaped concave corner: two slopes meeting
            let rv = |v: Vec3| rotate_vert(v, center, rotation);
            let rn = |n: Vec3| rotate_normal(n, rotation);

            // Bottom face (full)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz + 1.0)),
                rv(Vec3::new(bx,     by, bz + 1.0)),
                rn(Vec3::NEG_Y), color);
            // Back face (+Z, full height)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rn(Vec3::Z), color);
            // Right face (+X, full height)
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rn(Vec3::X), color);
            // Slope from -Z side (triangle): front-left edge rises to back
            let slope_nz = Vec3::new(0.0, 1.0, -1.0).normalize();
            push_tri(vertices, indices,
                rv(Vec3::new(bx,     by,     bz)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rn(slope_nz), color);
            // Slope from -X side (triangle): front-left edge rises to right
            let slope_nx = Vec3::new(-1.0, 1.0, 0.0).normalize();
            push_tri(vertices, indices,
                rv(Vec3::new(bx,     by,     bz)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rn(slope_nx), color);
            // Diagonal slope face (the valley)
            let diag_n = Vec3::new(-1.0, 1.0, -1.0).normalize();
            push_tri(vertices, indices,
                rv(Vec3::new(bx,     by,     bz)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rn(diag_n), color);
        }
        BlockType::Stairs => {
            // Two steps: bottom half full, top half back only
            let rv = |v: Vec3| rotate_vert(v, center, rotation);
            let rn = |n: Vec3| rotate_normal(n, rotation);
            let h = 0.5;

            // Bottom face
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz)),
                rv(Vec3::new(bx + 1.0, by, bz + 1.0)),
                rv(Vec3::new(bx,     by, bz + 1.0)),
                rn(Vec3::NEG_Y), color);
            // Bottom step top face (front half, y=0.5)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by + h, bz + h)),
                rv(Vec3::new(bx + 1.0, by + h, bz + h)),
                rv(Vec3::new(bx + 1.0, by + h, bz)),
                rv(Vec3::new(bx,     by + h, bz)),
                rn(Vec3::Y), color);
            // Top step top face (back half, y=1.0)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + h)),
                rv(Vec3::new(bx,     by + 1.0, bz + h)),
                rn(Vec3::Y), color);
            // Front riser (-Z, bottom step)
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by, bz)),
                rv(Vec3::new(bx,     by, bz)),
                rv(Vec3::new(bx,     by + h, bz)),
                rv(Vec3::new(bx + 1.0, by + h, bz)),
                rn(Vec3::NEG_Z), color);
            // Middle riser (-Z, top step at z=0.5)
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by + h, bz + h)),
                rv(Vec3::new(bx,     by + h, bz + h)),
                rv(Vec3::new(bx,     by + 1.0, bz + h)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + h)),
                rn(Vec3::NEG_Z), color);
            // Back face (+Z, full height)
            push_quad(vertices, indices,
                rv(Vec3::new(bx,     by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx,     by + 1.0, bz + 1.0)),
                rn(Vec3::Z), color);
            // Left side (-X): stair profile
            push_quad(vertices, indices,
                rv(Vec3::new(bx, by,     bz)),
                rv(Vec3::new(bx, by + h, bz)),
                rv(Vec3::new(bx, by + h, bz + h)),
                rv(Vec3::new(bx, by,     bz + h)),
                rn(Vec3::NEG_X), color);
            push_quad(vertices, indices,
                rv(Vec3::new(bx, by,     bz + h)),
                rv(Vec3::new(bx, by + 1.0, bz + h)),
                rv(Vec3::new(bx, by + 1.0, bz + 1.0)),
                rv(Vec3::new(bx, by,     bz + 1.0)),
                rn(Vec3::NEG_X), color);
            // Right side (+X): stair profile
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by + h, bz)),
                rv(Vec3::new(bx + 1.0, by,     bz)),
                rv(Vec3::new(bx + 1.0, by,     bz + h)),
                rv(Vec3::new(bx + 1.0, by + h, bz + h)),
                rn(Vec3::X), color);
            push_quad(vertices, indices,
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + h)),
                rv(Vec3::new(bx + 1.0, by,     bz + h)),
                rv(Vec3::new(bx + 1.0, by,     bz + 1.0)),
                rv(Vec3::new(bx + 1.0, by + 1.0, bz + 1.0)),
                rn(Vec3::X), color);
        }
        BlockType::Fence => {
            // Thin center pillar: 0.25x1x0.25 centered at (0.5, 0.5, 0.5)
            let lo = 0.25;
            let hi = 0.75;
            let fx0 = bx + lo;
            let fx1 = bx + hi;
            let fz0 = bz + lo;
            let fz1 = bz + hi;
            // Always visible (thin shape, no culling needed)
            // +Y
            push_quad(vertices, indices,
                Vec3::new(fx0, by + 1.0, fz1),
                Vec3::new(fx1, by + 1.0, fz1),
                Vec3::new(fx1, by + 1.0, fz0),
                Vec3::new(fx0, by + 1.0, fz0),
                Vec3::Y, color);
            // -Y
            push_quad(vertices, indices,
                Vec3::new(fx0, by, fz0),
                Vec3::new(fx1, by, fz0),
                Vec3::new(fx1, by, fz1),
                Vec3::new(fx0, by, fz1),
                Vec3::NEG_Y, color);
            // +X
            push_quad(vertices, indices,
                Vec3::new(fx1, by,     fz1),
                Vec3::new(fx1, by + 1.0, fz1),
                Vec3::new(fx1, by + 1.0, fz0),
                Vec3::new(fx1, by,     fz0),
                Vec3::X, color);
            // -X
            push_quad(vertices, indices,
                Vec3::new(fx0, by,     fz0),
                Vec3::new(fx0, by + 1.0, fz0),
                Vec3::new(fx0, by + 1.0, fz1),
                Vec3::new(fx0, by,     fz1),
                Vec3::NEG_X, color);
            // +Z
            push_quad(vertices, indices,
                Vec3::new(fx0, by,     fz1),
                Vec3::new(fx1, by,     fz1),
                Vec3::new(fx1, by + 1.0, fz1),
                Vec3::new(fx0, by + 1.0, fz1),
                Vec3::Z, color);
            // -Z
            push_quad(vertices, indices,
                Vec3::new(fx1, by,     fz0),
                Vec3::new(fx0, by,     fz0),
                Vec3::new(fx0, by + 1.0, fz0),
                Vec3::new(fx1, by + 1.0, fz0),
                Vec3::NEG_Z, color);
        }
    }
}

/// Neighbor face culling helper — checks if the neighbor in the given direction is fully solid.
fn neighbor_face_solid_check(
    grid: &BuildingGrid,
    cx: i32, cy: i32, cz: i32,
    dx: i32, dy: i32, dz: i32,
    _rotation: u8,
) -> bool {
    let (_, _, neighbor_face) = match (dx, dy, dz) {
        (0, -1, 0) => (BOTTOM_LAYER_MASK, TOP_LAYER_MASK, TOP_LAYER_MASK),
        (0,  1, 0) => (TOP_LAYER_MASK, BOTTOM_LAYER_MASK, BOTTOM_LAYER_MASK),
        (1,  0, 0) => (POS_X_FACE_MASK, NEG_X_FACE_MASK, NEG_X_FACE_MASK),
        (-1, 0, 0) => (NEG_X_FACE_MASK, POS_X_FACE_MASK, POS_X_FACE_MASK),
        (0,  0, 1) => (POS_Z_FACE_MASK, NEG_Z_FACE_MASK, NEG_Z_FACE_MASK),
        (0,  0,-1) => (NEG_Z_FACE_MASK, POS_Z_FACE_MASK, POS_Z_FACE_MASK),
        _ => return false,
    };
    is_neighbor_face_solid(&grid.cells, cx + dx, cy + dy, cz + dz, neighbor_face)
}
