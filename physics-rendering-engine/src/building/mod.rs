mod block_type;
mod shapes;
mod mesh;
mod groups;

pub use block_type::*;
pub use shapes::build_block_shape;
pub(crate) use mesh::push_quad;
pub use groups::BakedGroup;

use std::collections::{HashMap, HashSet};

use glam::Vec3;
use crate::physics::body::{ColliderHandle, RigidBodyHandle};
use crate::physics::world::PhysicsWorld;
use crate::renderer::mesh::Vertex;

use shapes::build_compound_shape;
use mesh::{emit_block_mesh, greedy_mesh_cubes};
use groups::{build_group_physics, generate_group_meshes_with_selection};

// ---------------------------------------------------------------------------
// Cell data
// ---------------------------------------------------------------------------

/// Data stored per occupied grid cell.
pub(super) struct CellData {
    rigid_body: RigidBodyHandle,
    collider: ColliderHandle,
    /// 64-bit mask for the 4x4x4 sub-block grid. All 1s = fully intact.
    pub(super) sub_blocks: u64,
    pub(super) block_type: BlockType,
    pub(super) rotation: u8,
    pub(super) color: Vec3,
}

// ---------------------------------------------------------------------------
// Building grid
// ---------------------------------------------------------------------------

/// A grid-based building system. Cubes snap to integer grid positions.
/// Cell (cx, cy, cz) occupies [cx, cx+1] x [cy, cy+1] x [cz, cz+1].
/// Each cell is subdivided into 4x4x4 sub-blocks that can be individually mined.
pub struct BuildingGrid {
    pub(super) cells: HashMap<(i32, i32, i32), CellData>,
    groups: Vec<BakedGroup>,
    /// Reverse lookup: rigid body handle -> index into `groups`.
    group_body_map: HashMap<RigidBodyHandle, usize>,
    /// O(1) lookup for whether a rigid body belongs to any cell (not group).
    cell_body_set: HashSet<RigidBodyHandle>,
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
            cell_body_set: HashSet::new(),
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
        let (rigid_body, collider) = physics.add_static_shape(center, shape, crate::physics::world::cg_building());

        self.cell_body_set.insert(rigid_body);
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
        let (rigid_body, collider) = physics.add_static_shape(center, shape, crate::physics::world::cg_building());

        self.cell_body_set.insert(rigid_body);
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
            self.cell_body_set.remove(&cell.rigid_body);
            physics.remove_body(cell.rigid_body, cell.collider);
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Check if a rigid body belongs to a building cell or baked group.
    pub fn has_body(&self, rb: RigidBodyHandle) -> bool {
        self.cell_body_set.contains(&rb)
            || self.group_body_map.contains_key(&rb)
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
                // Fully destroyed -- remove the cell.
                let cell = self.cells.remove(&(cx, cy, cz)).unwrap();
                self.cell_body_set.remove(&cell.rigid_body);
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
            self.cell_body_set.remove(&cell.rigid_body);
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

        // Build maps for greedy meshing of unbaked pristine cubes.
        let mut all_cells_map: HashMap<(i32, i32, i32), (u64, BlockType, u8, Vec3)> =
            HashMap::with_capacity(self.cells.len());
        let mut cube_colors: HashMap<(i32, i32, i32), Vec3> = HashMap::new();

        for (&(cx, cy, cz), cell) in &self.cells {
            all_cells_map.insert(
                (cx, cy, cz),
                (cell.sub_blocks, cell.block_type, cell.rotation, cell.color),
            );
            let pristine_mask = rotate_sub_blocks(initial_sub_blocks(cell.block_type, cell.rotation), cell.rotation);
            if cell.sub_blocks == pristine_mask && cell.block_type == BlockType::Cube {
                cube_colors.insert((cx, cy, cz), cell.color);
            } else if cell.sub_blocks == pristine_mask {
                // Non-cube pristine block -- emit with per-block neighbor culling.
                emit_block_mesh(
                    cell.block_type, cell.rotation, cx, cy, cz,
                    &self.cells, self, cell.color, &mut vertices, &mut indices,
                );
            } else {
                // Mined -- fall back to sub-block iteration.
                self.emit_sub_block_mesh(cx, cy, cz, cell, &mut vertices, &mut indices);
            }
        }

        // Greedy-mesh the pristine cubes.
        greedy_mesh_cubes(&all_cells_map, &cube_colors, &mut vertices, &mut indices);

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
        generate_group_meshes_with_selection(&self.groups, &mut vertices, &mut indices, ghost_groups, selected_group);

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
        let (rigid_body, collider) = physics.add_static_shape(center, shape, crate::physics::world::cg_building());
        self.cell_body_set.insert(rigid_body);
        self.cells.insert((cx, cy, cz), CellData {
            rigid_body, collider, sub_blocks, block_type, rotation, color,
        });
        self.dirty = true;
    }

    /// Remove all cells and physics bodies.
    pub fn clear(&mut self, physics: &mut PhysicsWorld) {
        self.cell_body_set.clear();
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
        let blocks: Vec<crate::persistence::blueprint::BlockEntry> = self.cells.keys().map(|&(x, y, z)| {
            let c = &self.cells[&(x, y, z)];
            crate::persistence::blueprint::BlockEntry {
                x, y, z,
                block_type: c.block_type as u8,
                rotation: c.rotation,
                sub_blocks: c.sub_blocks,
                color: [c.color.x, c.color.y, c.color.z],
            }
        }).collect();

        // Remove individual physics bodies.
        self.cell_body_set.clear();
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
    pub fn load_group(&mut self, physics: &mut PhysicsWorld, blocks: Vec<crate::persistence::blueprint::BlockEntry>) {
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
                              blocks: &[crate::persistence::blueprint::BlockEntry],
                              ox: i32, oy: i32, oz: i32) {
        let offset_blocks: Vec<crate::persistence::blueprint::BlockEntry> = blocks.iter().map(|b| {
            crate::persistence::blueprint::BlockEntry {
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
