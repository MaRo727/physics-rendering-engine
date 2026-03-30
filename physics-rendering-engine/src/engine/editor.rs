use super::*;

/// Editor color palette (12 colors).
pub(crate) const EDITOR_PALETTE: [Vec3; 12] = [
    Vec3::new(0.9, 0.9, 0.9),   // 0: white
    Vec3::new(0.3, 0.3, 0.3),   // 1: dark gray
    Vec3::new(0.85, 0.2, 0.2),  // 2: red
    Vec3::new(0.2, 0.7, 0.2),   // 3: green
    Vec3::new(0.2, 0.3, 0.85),  // 4: blue
    Vec3::new(0.9, 0.85, 0.2),  // 5: yellow
    Vec3::new(0.55, 0.35, 0.15),// 6: brown
    Vec3::new(0.6, 0.2, 0.7),   // 7: purple
    Vec3::new(0.9, 0.5, 0.1),   // 8: orange
    Vec3::new(0.2, 0.8, 0.8),   // 9: cyan
    Vec3::new(0.7, 0.7, 0.65),  // 10: building tan (default)
    Vec3::new(0.5, 0.5, 0.5),   // 11: medium gray
];

/// Which plane the drag-to-fill rectangle lives on.
#[derive(Debug, Clone, Copy)]
pub(crate) enum DragPlane {
    /// Floor: fill XZ at fixed Y.
    FloorXZ(i32),
    /// Wall: fill XY at fixed Z.
    WallXY(i32),
    /// Wall: fill YZ at fixed X.
    WallYZ(i32),
}

/// Maximum undo stack depth.
const MAX_UNDO: usize = 100;

// ---------------------------------------------------------------------------
// Undo/redo operations
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) enum EditorOp {
    /// A single block was placed.
    PlaceBlock { pos: (i32, i32, i32), block_type: BlockType, rotation: u8, color: Vec3 },
    /// A single block was removed (stores its data for restore).
    RemoveBlock { pos: (i32, i32, i32), block_type: BlockType, rotation: u8, sub_blocks: u64, color: Vec3 },
    /// A fill region was placed (stores only positions that were actually placed).
    Fill { blocks: Vec<((i32, i32, i32), BlockType, u8, Vec3)> },
    /// Cells were baked into a group (stores snapshot of cells for unbake on undo).
    BakeGroup { cell_snapshot: Vec<blueprint::BlockEntry> },
    /// A group was unbaked (stores the group blocks for re-bake on undo).
    UnbakeGroup { group_blocks: Vec<blueprint::BlockEntry> },
    /// Colors were replaced on loose cells.
    RecolorCells { changes: Vec<((i32, i32, i32), Vec3)>, new_color: Vec3 },
    /// Colors were replaced on a baked group.
    RecolorGroup { group_idx: usize, changes: Vec<(usize, [f32; 3])>, new_color: Vec3 },
    /// A baked group was moved.
    MoveGroup { group_idx: usize, dx: i32, dy: i32, dz: i32 },
    /// A group was pasted (stores the group index for removal on undo).
    PasteGroup { group_idx: usize },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Intersect a ray with the drag plane, returning the snapped grid coordinates
/// on the two free axes (the fixed axis comes from the plane).
pub(super) fn ray_plane_intersect(origin: Vec3, dir: Vec3, plane: DragPlane) -> Option<(i32, i32, i32)> {
    let (plane_val, axis_component) = match plane {
        DragPlane::FloorXZ(y) => (y as f32 + 0.5, dir.y),
        DragPlane::WallXY(z) => (z as f32 + 0.5, dir.z),
        DragPlane::WallYZ(x) => (x as f32 + 0.5, dir.x),
    };
    // Ray nearly parallel to plane -- skip.
    if axis_component.abs() < 1e-6 {
        return None;
    }
    let t = match plane {
        DragPlane::FloorXZ(_) => (plane_val - origin.y) / dir.y,
        DragPlane::WallXY(_) => (plane_val - origin.z) / dir.z,
        DragPlane::WallYZ(_) => (plane_val - origin.x) / dir.x,
    };
    if t < 0.0 || t > 200.0 {
        return None;
    }
    let hit = origin + dir * t;
    Some(match plane {
        DragPlane::FloorXZ(y) => (hit.x.floor() as i32, y, hit.z.floor() as i32),
        DragPlane::WallXY(z) => (hit.x.floor() as i32, hit.y.floor() as i32, z),
        DragPlane::WallYZ(x) => (x, hit.y.floor() as i32, hit.z.floor() as i32),
    })
}

/// Build the list of cells that fill the rectangle between `start` and `end` on the given plane.
pub(super) fn build_fill_region(
    start: (i32, i32, i32),
    end: (i32, i32, i32),
    plane: DragPlane,
    block_type: BlockType,
    rotation: u8,
    color: Vec3,
) -> Vec<((i32, i32, i32), BlockType, u8, Vec3)> {
    let mut cells = Vec::new();
    match plane {
        DragPlane::FloorXZ(y) => {
            let (x0, x1) = (start.0.min(end.0), start.0.max(end.0));
            let (z0, z1) = (start.2.min(end.2), start.2.max(end.2));
            for x in x0..=x1 {
                for z in z0..=z1 {
                    cells.push(((x, y, z), block_type, rotation, color));
                }
            }
        }
        DragPlane::WallXY(z) => {
            let (x0, x1) = (start.0.min(end.0), start.0.max(end.0));
            let (y0, y1) = (start.1.min(end.1), start.1.max(end.1));
            for x in x0..=x1 {
                for y in y0..=y1 {
                    cells.push(((x, y, z), block_type, rotation, color));
                }
            }
        }
        DragPlane::WallYZ(x) => {
            let (y0, y1) = (start.1.min(end.1), start.1.max(end.1));
            let (z0, z1) = (start.2.min(end.2), start.2.max(end.2));
            for y in y0..=y1 {
                for z in z0..=z1 {
                    cells.push(((x, y, z), block_type, rotation, color));
                }
            }
        }
    }
    cells
}

/// Mirror a block's rotation when reflecting on the X axis.
fn mirror_rotation_x(block_type: BlockType, rotation: u8) -> u8 {
    match block_type {
        BlockType::Cube | BlockType::Slab | BlockType::Fence => rotation,
        // For directional blocks: mirror swaps rotation 1 <-> 3
        _ => match rotation {
            1 => 3,
            3 => 1,
            other => other,
        }
    }
}

/// Mirror a block's rotation when reflecting on the Z axis.
fn mirror_rotation_z(block_type: BlockType, rotation: u8) -> u8 {
    match block_type {
        BlockType::Cube | BlockType::Slab | BlockType::Fence => rotation,
        // For directional blocks: mirror swaps rotation 0 <-> 2
        _ => match rotation {
            0 => 2,
            2 => 0,
            other => other,
        }
    }
}

impl Engine {
    /// Push an operation onto the undo stack, clearing the redo stack.
    fn push_undo(&mut self, op: EditorOp) {
        self.undo_stack.push(op);
        self.redo_stack.clear();
        if self.undo_stack.len() > MAX_UNDO {
            self.undo_stack.remove(0);
        }
    }

    /// Clear extrude state (called when a non-extrude operation happens).
    fn clear_extrude(&mut self) {
        self.editor_last_fill = None;
        self.editor_extrude_height = 0;
    }

    // -----------------------------------------------------------------------
    // Undo / Redo
    // -----------------------------------------------------------------------

    fn apply_undo(&mut self) {
        let op = match self.undo_stack.pop() {
            Some(op) => op,
            None => {
                self.editor_status = Some(("Nothing to undo".to_string(), 1.5));
                return;
            }
        };

        match op.clone() {
            EditorOp::PlaceBlock { pos, .. } => {
                self.editor_grid.remove(&mut self.editor_physics, pos.0, pos.1, pos.2);
            }
            EditorOp::RemoveBlock { pos, block_type, rotation, sub_blocks, color } => {
                self.editor_grid.load_cell(
                    &mut self.editor_physics, pos.0, pos.1, pos.2,
                    block_type, rotation, sub_blocks, color,
                );
            }
            EditorOp::Fill { ref blocks } => {
                for &((x, y, z), _, _, _) in blocks {
                    self.editor_grid.remove(&mut self.editor_physics, x, y, z);
                }
            }
            EditorOp::BakeGroup { ref cell_snapshot } => {
                // Undo bake: remove the last group, restore cells.
                let gc = self.editor_grid.group_count();
                if gc > 0 {
                    self.editor_grid.destroy_group(&mut self.editor_physics, gc - 1);
                    for b in cell_snapshot {
                        let bt = BlockType::from_u8(b.block_type);
                        let color = Vec3::new(b.color[0], b.color[1], b.color[2]);
                        self.editor_grid.load_cell(
                            &mut self.editor_physics, b.x, b.y, b.z,
                            bt, b.rotation, b.sub_blocks, color,
                        );
                    }
                }
            }
            EditorOp::UnbakeGroup { ref group_blocks } => {
                // Undo unbake: remove the cells that were restored, re-bake the group.
                for b in group_blocks {
                    self.editor_grid.remove(&mut self.editor_physics, b.x, b.y, b.z);
                }
                self.editor_grid.load_group(&mut self.editor_physics, group_blocks.clone());
            }
            EditorOp::RecolorCells { ref changes, .. } => {
                for &(pos, old_color) in changes {
                    if let Some(cell) = self.editor_grid.cells.get_mut(&pos) {
                        cell.color = old_color;
                    }
                }
                self.editor_grid.mark_dirty();
            }
            EditorOp::RecolorGroup { group_idx, ref changes, .. } => {
                if group_idx < self.editor_grid.group_count() {
                    let group = &mut self.editor_grid.groups_mut()[group_idx];
                    for &(block_idx, old_color) in changes {
                        if block_idx < group.blocks.len() {
                            group.blocks[block_idx].color = old_color;
                        }
                    }
                    self.editor_grid.mark_dirty();
                }
            }
            EditorOp::MoveGroup { group_idx, dx, dy, dz } => {
                if group_idx < self.editor_grid.group_count() {
                    self.editor_grid.move_group(&mut self.editor_physics, group_idx, -dx, -dy, -dz);
                }
            }
            EditorOp::PasteGroup { group_idx } => {
                if group_idx < self.editor_grid.group_count() {
                    self.editor_grid.destroy_group(&mut self.editor_physics, group_idx);
                    self.editor_selected_group = None;
                }
            }
        }

        self.editor_physics.step(0.0);
        self.redo_stack.push(op);
        self.editor_status = Some((format!("Undo ({})", self.undo_stack.len()), 1.5));
    }

    fn apply_redo(&mut self) {
        let op = match self.redo_stack.pop() {
            Some(op) => op,
            None => {
                self.editor_status = Some(("Nothing to redo".to_string(), 1.5));
                return;
            }
        };

        match op.clone() {
            EditorOp::PlaceBlock { pos, block_type, rotation, color } => {
                self.editor_grid.place_unsupported(
                    &mut self.editor_physics, pos.0, pos.1, pos.2,
                    block_type, rotation, color,
                );
            }
            EditorOp::RemoveBlock { pos, .. } => {
                self.editor_grid.remove(&mut self.editor_physics, pos.0, pos.1, pos.2);
            }
            EditorOp::Fill { ref blocks } => {
                for &((x, y, z), bt, rot, col) in blocks {
                    self.editor_grid.place_unsupported(
                        &mut self.editor_physics, x, y, z, bt, rot, col,
                    );
                }
            }
            EditorOp::BakeGroup { .. } => {
                self.editor_grid.bake_group(&mut self.editor_physics);
            }
            EditorOp::UnbakeGroup { .. } => {
                let gc = self.editor_grid.group_count();
                if gc > 0 {
                    self.editor_grid.unbake_group(&mut self.editor_physics, gc - 1);
                }
            }
            EditorOp::RecolorCells { ref changes, new_color } => {
                for &(pos, _) in changes {
                    if let Some(cell) = self.editor_grid.cells.get_mut(&pos) {
                        cell.color = new_color;
                    }
                }
                self.editor_grid.mark_dirty();
            }
            EditorOp::RecolorGroup { group_idx, ref changes, new_color } => {
                if group_idx < self.editor_grid.group_count() {
                    let group = &mut self.editor_grid.groups_mut()[group_idx];
                    for &(block_idx, _) in changes {
                        if block_idx < group.blocks.len() {
                            group.blocks[block_idx].color = [new_color.x, new_color.y, new_color.z];
                        }
                    }
                    self.editor_grid.mark_dirty();
                }
            }
            EditorOp::MoveGroup { group_idx, dx, dy, dz } => {
                if group_idx < self.editor_grid.group_count() {
                    self.editor_grid.move_group(&mut self.editor_physics, group_idx, dx, dy, dz);
                }
            }
            EditorOp::PasteGroup { .. } => {
                // Re-paste from clipboard if available.
                if let Some(ref clip) = self.editor_clipboard {
                    self.editor_grid.load_group(&mut self.editor_physics, clip.clone());
                }
            }
        }

        self.editor_physics.step(0.0);
        self.undo_stack.push(op);
        self.editor_status = Some((format!("Redo ({})", self.redo_stack.len()), 1.5));
    }

    // -----------------------------------------------------------------------
    // Structure editor
    // -----------------------------------------------------------------------

    pub(crate) fn update_editor(&mut self, dt: f32, input: &InputState) {
        // Ensure editor physics has a ground plane (lazy init).
        if !self.editor_ground_inited {
            self.editor_ground_inited = true;
            use crate::physics::body::SharedShape;
            let half_ext = Vec3::new(200.0, 0.5, 200.0);
            self.editor_physics.add_static_shape(
                Vec3::new(0.0, -0.5, 0.0),
                SharedShape::cuboid(half_ext.x, half_ext.y, half_ext.z),
                crate::physics::world::cg_terrain(),
            );
            self.editor_physics.step(0.0); // build query pipeline
        }

        // Camera movement.
        self.editor_camera.update(dt, input);

        // Color selection via number keys.
        if let Some(slot) = input.editor_color_slot {
            let idx = slot as usize;
            if idx < EDITOR_PALETTE.len() {
                self.editor_color_idx = idx;
            }
        }

        // Scroll wheel cycles through palette.
        if input.scroll_delta.abs() > 0.1 {
            let len = EDITOR_PALETTE.len() as i32;
            let dir = if input.scroll_delta > 0.0 { 1 } else { -1 };
            self.editor_color_idx = ((self.editor_color_idx as i32 + dir).rem_euclid(len)) as usize;
        }

        // Block type cycling (B).
        let cycle_block = input.cycle_block_type && !self.cycle_block_prev;
        self.cycle_block_prev = input.cycle_block_type;
        if cycle_block {
            self.selected_block_type = self.selected_block_type.next();
            self.selected_rotation = 0;
        }

        // Block rotation (V) — only if not Ctrl+V (paste).
        let rotate = input.rotate_block && !self.rotate_prev && !input.ctrl_held;
        self.rotate_prev = input.rotate_block;
        if rotate {
            if self.selected_block_type == BlockType::Slab {
                self.selected_rotation = if self.selected_rotation == 0 { 1 } else { 0 };
            } else {
                self.selected_rotation = (self.selected_rotation + 1) % 4;
            }
        }

        let cam_eye = self.editor_camera.eye;
        let (sy, cy_cos) = self.editor_camera.yaw.sin_cos();
        let (sp, cp) = self.editor_camera.pitch.sin_cos();
        let look_dir = Vec3::new(-sy * cp, sp, -cy_cos * cp);

        // -------------------------------------------------------------------
        // Undo / Redo (Ctrl+Z / Ctrl+Y)
        // -------------------------------------------------------------------
        if input.editor_undo && !self.editor_undo_prev {
            self.clear_extrude();
            self.apply_undo();
        }
        self.editor_undo_prev = input.editor_undo;

        if input.editor_redo && !self.editor_redo_prev {
            self.clear_extrude();
            self.apply_redo();
        }
        self.editor_redo_prev = input.editor_redo;

        // -------------------------------------------------------------------
        // Copy (Ctrl+C)
        // -------------------------------------------------------------------
        if input.editor_copy && !self.editor_copy_prev {
            if let Some(gi) = self.editor_selected_group {
                let blocks = self.editor_grid.groups()[gi].blocks.clone();
                let count = blocks.len();
                self.editor_clipboard = Some(blocks);
                self.editor_status = Some((format!("Copied {} blocks", count), 2.0));
            } else if self.editor_grid.cell_count() > 0 {
                // Copy loose cells as clipboard.
                let blocks: Vec<blueprint::BlockEntry> = self.editor_grid.occupied_cells().filter_map(|&(x, y, z)| {
                    self.editor_grid.cell_info(x, y, z).map(|(bt, rot, subs, col)| {
                        blueprint::BlockEntry {
                            x, y, z,
                            block_type: bt as u8,
                            rotation: rot,
                            sub_blocks: subs,
                            color: [col.x, col.y, col.z],
                        }
                    })
                }).collect();
                let count = blocks.len();
                self.editor_clipboard = Some(blocks);
                self.editor_status = Some((format!("Copied {} loose blocks", count), 2.0));
            } else {
                self.editor_status = Some(("Nothing to copy".to_string(), 2.0));
            }
        }
        self.editor_copy_prev = input.editor_copy;

        // -------------------------------------------------------------------
        // Paste (Ctrl+V)
        // -------------------------------------------------------------------
        if input.editor_paste && !self.editor_paste_prev {
            if let Some(ref clip) = self.editor_clipboard.clone() {
                // Place at camera target position.
                let target = if let Some((_rb, hit_pos, _normal)) = self.editor_physics.cast_ray_unfiltered(cam_eye, look_dir, 50.0) {
                    let (cx, cy, cz) = building::snap_to_grid(hit_pos + Vec3::Y * 0.01);
                    (cx, cy, cz)
                } else {
                    // Place in front of camera.
                    let pos = cam_eye + look_dir * 10.0;
                    building::snap_to_grid(pos)
                };

                // Compute clipboard min corner.
                let min_x = clip.iter().map(|b| b.x).min().unwrap_or(0);
                let min_y = clip.iter().map(|b| b.y).min().unwrap_or(0);
                let min_z = clip.iter().map(|b| b.z).min().unwrap_or(0);

                let ox = target.0 - min_x;
                let oy = target.1 - min_y;
                let oz = target.2 - min_z;

                self.editor_grid.load_group_offset(&mut self.editor_physics, clip, ox, oy, oz);
                self.editor_physics.step(0.0);

                let group_idx = self.editor_grid.group_count() - 1;
                self.editor_selected_group = Some(group_idx);
                self.push_undo(EditorOp::PasteGroup { group_idx });
                self.clear_extrude();

                self.editor_status = Some((format!("Pasted {} blocks", clip.len()), 2.0));
            } else {
                self.editor_status = Some(("Clipboard empty".to_string(), 2.0));
            }
        }
        self.editor_paste_prev = input.editor_paste;

        // -------------------------------------------------------------------
        // Mirror clipboard (X / Z)
        // -------------------------------------------------------------------
        if input.editor_mirror_x && !self.editor_mirror_x_prev {
            if let Some(ref mut clip) = self.editor_clipboard {
                let max_x = clip.iter().map(|b| b.x).max().unwrap_or(0);
                let min_x = clip.iter().map(|b| b.x).min().unwrap_or(0);
                for b in clip.iter_mut() {
                    b.x = max_x - (b.x - min_x);
                    let bt = BlockType::from_u8(b.block_type);
                    b.rotation = mirror_rotation_x(bt, b.rotation);
                    b.sub_blocks = building::mirror_sub_blocks_x(b.sub_blocks);
                }
                self.editor_status = Some(("Mirrored X".to_string(), 2.0));
            } else {
                self.editor_status = Some(("No clipboard to mirror".to_string(), 2.0));
            }
        }
        self.editor_mirror_x_prev = input.editor_mirror_x;

        if input.editor_mirror_z && !self.editor_mirror_z_prev {
            if let Some(ref mut clip) = self.editor_clipboard {
                let max_z = clip.iter().map(|b| b.z).max().unwrap_or(0);
                let min_z = clip.iter().map(|b| b.z).min().unwrap_or(0);
                for b in clip.iter_mut() {
                    b.z = max_z - (b.z - min_z);
                    let bt = BlockType::from_u8(b.block_type);
                    b.rotation = mirror_rotation_z(bt, b.rotation);
                    b.sub_blocks = building::mirror_sub_blocks_z(b.sub_blocks);
                }
                self.editor_status = Some(("Mirrored Z".to_string(), 2.0));
            } else {
                self.editor_status = Some(("No clipboard to mirror".to_string(), 2.0));
            }
        }
        self.editor_mirror_z_prev = input.editor_mirror_z;

        // -------------------------------------------------------------------
        // Replace color (H)
        // -------------------------------------------------------------------
        if input.editor_replace_color && !self.editor_replace_color_prev {
            let new_color = EDITOR_PALETTE[self.editor_color_idx];
            if let Some(gi) = self.editor_selected_group {
                // Raycast to find target color within the group.
                if let Some((_rb, hit_pos, normal)) = self.editor_physics.cast_ray_unfiltered(cam_eye, look_dir, 50.0) {
                    let target = hit_pos - normal * 0.01;
                    let (cx, cy, cz) = building::snap_to_grid(target);
                    // Find the color of the block at this position in the group.
                    let old_color = {
                        let group = &self.editor_grid.groups()[gi];
                        group.blocks.iter().find(|b| b.x == cx && b.y == cy && b.z == cz)
                            .map(|b| Vec3::new(b.color[0], b.color[1], b.color[2]))
                    };
                    if let Some(old_color) = old_color {
                        let changes = self.editor_grid.recolor_group_matching(gi, old_color, new_color);
                        if !changes.is_empty() {
                            let count = changes.len();
                            self.push_undo(EditorOp::RecolorGroup { group_idx: gi, changes, new_color });
                            self.editor_status = Some((format!("Recolored {} blocks", count), 2.0));
                        }
                    }
                }
            } else {
                // Raycast to find target color among loose cells.
                if let Some((_rb, hit_pos, normal)) = self.editor_physics.cast_ray_unfiltered(cam_eye, look_dir, 50.0) {
                    let target = hit_pos - normal * 0.01;
                    let (cx, cy, cz) = building::snap_to_grid(target);
                    if let Some((_, _, _, old_color)) = self.editor_grid.cell_info(cx, cy, cz) {
                        let changes = self.editor_grid.recolor_matching(old_color, new_color);
                        if !changes.is_empty() {
                            let count = changes.len();
                            self.push_undo(EditorOp::RecolorCells { changes, new_color });
                            self.editor_status = Some((format!("Recolored {} blocks", count), 2.0));
                        }
                    }
                }
            }
        }
        self.editor_replace_color_prev = input.editor_replace_color;

        // -------------------------------------------------------------------
        // Drag-to-fill placement (RMB)
        // -------------------------------------------------------------------
        const DRAG_THRESHOLD: f32 = 0.5;
        let place_press = input.place && !self.place_prev;
        let place_release = !input.place && self.place_prev;
        self.place_prev = input.place;

        if place_press {
            self.clear_extrude();
            // Immediately place a single block on click.
            if let Some((_rb, hit_pos, normal)) = self.editor_physics.cast_ray_unfiltered(cam_eye, look_dir, 50.0) {
                let target = hit_pos + normal * 0.01;
                let (cx, cy, cz) = building::snap_to_grid(target);
                let color = EDITOR_PALETTE[self.editor_color_idx];
                if self.editor_grid.place_unsupported(
                    &mut self.editor_physics, cx, cy, cz,
                    self.selected_block_type, self.selected_rotation, color,
                ) {
                    self.editor_physics.step(0.0);
                    self.push_undo(EditorOp::PlaceBlock {
                        pos: (cx, cy, cz),
                        block_type: self.selected_block_type,
                        rotation: self.selected_rotation,
                        color,
                    });
                }
                // Record start for potential drag.
                let (ax, ay, az) = (normal.x.abs(), normal.y.abs(), normal.z.abs());
                let plane = if ay >= ax && ay >= az {
                    DragPlane::FloorXZ(cy)
                } else if ax >= az {
                    DragPlane::WallYZ(cx)
                } else {
                    DragPlane::WallXY(cz)
                };
                self.drag_start = Some((cx, cy, cz));
                self.drag_plane = Some(plane);
                self.drag_end = Some((cx, cy, cz));
                self.drag_hold_timer = 0.0;
                self.drag_active = false;
            }
        } else if input.place && self.drag_start.is_some() {
            // Holding RMB -- accumulate hold time, activate drag after threshold.
            self.drag_hold_timer += dt;
            if self.drag_hold_timer >= DRAG_THRESHOLD {
                self.drag_active = true;
            }
            if self.drag_active {
                if let (Some(start), Some(plane)) = (self.drag_start, self.drag_plane)
                    && let Some(end) = ray_plane_intersect(cam_eye, look_dir, plane)
                {
                    let prev_end = self.drag_end;
                    self.drag_end = Some(end);
                    if self.drag_end != prev_end {
                        let color = EDITOR_PALETTE[self.editor_color_idx];
                        let bt = self.selected_block_type;
                        let rot = self.selected_rotation;
                        let preview = build_fill_region(start, end, plane, bt, rot, color);
                        self.editor_grid.set_preview(preview);
                    }
                }
            }
        } else if place_release && self.drag_start.is_some() {
            // Release: if drag was active, place the fill region.
            if self.drag_active {
                if let (Some(start), Some(plane), Some(end)) = (self.drag_start, self.drag_plane, self.drag_end) {
                    let color = EDITOR_PALETTE[self.editor_color_idx];
                    let bt = self.selected_block_type;
                    let rot = self.selected_rotation;
                    let region = build_fill_region(start, end, plane, bt, rot, color);
                    let mut placed = Vec::new();
                    for &((cx, cy, cz), block_type, rotation, col) in &region {
                        if self.editor_grid.place_unsupported(
                            &mut self.editor_physics, cx, cy, cz,
                            block_type, rotation, col,
                        ) {
                            placed.push(((cx, cy, cz), block_type, rotation, col));
                        }
                    }
                    if !placed.is_empty() {
                        self.editor_physics.step(0.0);
                        // Save for extrude.
                        let positions: Vec<(i32, i32, i32)> = placed.iter().map(|&(p, _, _, _)| p).collect();
                        self.editor_last_fill = Some((positions, bt, rot, color));
                        self.editor_extrude_height = 0;
                        // Record undo.
                        let count = placed.len();
                        self.push_undo(EditorOp::Fill { blocks: placed });
                        if count > 1 {
                            self.editor_status = Some((format!("Placed {} blocks", count), 2.0));
                        }
                    }
                }
                self.editor_grid.clear_preview();
            }
            self.drag_start = None;
            self.drag_plane = None;
            self.drag_end = None;
            self.drag_active = false;
            self.drag_hold_timer = 0.0;
        }

        // -------------------------------------------------------------------
        // Remove block (LMB -- edge triggered)
        // -------------------------------------------------------------------
        let throw_edge = input.throw && !self.editor_throw_prev;
        self.editor_throw_prev = input.throw;
        if throw_edge {
            self.clear_extrude();
            if let Some((_rb, hit_pos, normal)) = self.editor_physics.cast_ray_unfiltered(cam_eye, look_dir, 50.0) {
                let target = hit_pos - normal * 0.01;
                let (cx, cy, cz) = building::snap_to_grid(target);
                if cy >= 0 { // don't remove ground
                    if let Some((bt, rot, subs, color)) = self.editor_grid.remove_and_return(&mut self.editor_physics, cx, cy, cz) {
                        self.editor_physics.step(0.0);
                        self.push_undo(EditorOp::RemoveBlock {
                            pos: (cx, cy, cz),
                            block_type: bt,
                            rotation: rot,
                            sub_blocks: subs,
                            color,
                        });
                    }
                }
            }
        }

        // -------------------------------------------------------------------
        // Arrow keys: extrude (no group) or move group (group selected)
        // -------------------------------------------------------------------
        let arrow_up_edge = input.editor_arrow_up && !self.editor_arrow_up_prev;
        let arrow_down_edge = input.editor_arrow_down && !self.editor_arrow_down_prev;
        let arrow_left_edge = input.editor_arrow_left && !self.editor_arrow_left_prev;
        let arrow_right_edge = input.editor_arrow_right && !self.editor_arrow_right_prev;
        self.editor_arrow_up_prev = input.editor_arrow_up;
        self.editor_arrow_down_prev = input.editor_arrow_down;
        self.editor_arrow_left_prev = input.editor_arrow_left;
        self.editor_arrow_right_prev = input.editor_arrow_right;

        if let Some(gi) = self.editor_selected_group {
            // Arrow keys move the selected group.
            if input.ctrl_held {
                // Ctrl+Up/Down: move in Y.
                if arrow_up_edge {
                    self.editor_grid.move_group(&mut self.editor_physics, gi, 0, 1, 0);
                    self.editor_physics.step(0.0);
                    self.push_undo(EditorOp::MoveGroup { group_idx: gi, dx: 0, dy: 1, dz: 0 });
                }
                if arrow_down_edge {
                    self.editor_grid.move_group(&mut self.editor_physics, gi, 0, -1, 0);
                    self.editor_physics.step(0.0);
                    self.push_undo(EditorOp::MoveGroup { group_idx: gi, dx: 0, dy: -1, dz: 0 });
                }
            } else {
                // Up/Down: move in Z.
                if arrow_up_edge {
                    self.editor_grid.move_group(&mut self.editor_physics, gi, 0, 0, -1);
                    self.editor_physics.step(0.0);
                    self.push_undo(EditorOp::MoveGroup { group_idx: gi, dx: 0, dy: 0, dz: -1 });
                }
                if arrow_down_edge {
                    self.editor_grid.move_group(&mut self.editor_physics, gi, 0, 0, 1);
                    self.editor_physics.step(0.0);
                    self.push_undo(EditorOp::MoveGroup { group_idx: gi, dx: 0, dy: 0, dz: 1 });
                }
            }
            // Left/Right: move in X.
            if arrow_left_edge {
                self.editor_grid.move_group(&mut self.editor_physics, gi, -1, 0, 0);
                self.editor_physics.step(0.0);
                self.push_undo(EditorOp::MoveGroup { group_idx: gi, dx: -1, dy: 0, dz: 0 });
            }
            if arrow_right_edge {
                self.editor_grid.move_group(&mut self.editor_physics, gi, 1, 0, 0);
                self.editor_physics.step(0.0);
                self.push_undo(EditorOp::MoveGroup { group_idx: gi, dx: 1, dy: 0, dz: 0 });
            }
        } else {
            // Vertical extrude (ArrowUp / ArrowDown after a fill).
            if arrow_up_edge {
                if let Some((ref positions, bt, rot, color)) = self.editor_last_fill.clone() {
                    self.editor_extrude_height += 1;
                    let h = self.editor_extrude_height;
                    let mut placed = Vec::new();
                    for &(x, y, z) in positions {
                        if self.editor_grid.place_unsupported(
                            &mut self.editor_physics, x, y + h, z, bt, rot, color,
                        ) {
                            placed.push(((x, y + h, z), bt, rot, color));
                        }
                    }
                    if !placed.is_empty() {
                        self.editor_physics.step(0.0);
                        let count = placed.len();
                        self.push_undo(EditorOp::Fill { blocks: placed });
                        self.editor_status = Some((format!("Extruded +{} ({} blocks)", h, count), 2.0));
                    }
                }
            }

            if arrow_down_edge && self.editor_extrude_height > 0 {
                if let Some((ref positions, _, _, _)) = self.editor_last_fill.clone() {
                    let h = self.editor_extrude_height;
                    for &(x, y, z) in positions {
                        self.editor_grid.remove(&mut self.editor_physics, x, y + h, z);
                    }
                    self.editor_physics.step(0.0);
                    self.editor_extrude_height -= 1;
                    // Pop the last undo entry (the fill we're removing).
                    if let Some(EditorOp::Fill { .. }) = self.undo_stack.last() {
                        self.undo_stack.pop();
                    }
                    self.editor_status = Some((format!("Extrude height: {}", self.editor_extrude_height), 2.0));
                }
            }
        }

        // -------------------------------------------------------------------
        // Save (F9)
        // -------------------------------------------------------------------
        if input.editor_save && !self.editor_save_prev {
            self.editor_save_blueprint();
        }
        self.editor_save_prev = input.editor_save;

        // Load (F10).
        if input.editor_load && !self.editor_load_prev {
            self.editor_load_blueprint();
        }
        self.editor_load_prev = input.editor_load;

        // -------------------------------------------------------------------
        // Bake group (G)
        // -------------------------------------------------------------------
        if input.toggle_ghost && !self.editor_bake_prev {
            let count = self.editor_grid.cell_count();
            if count > 0 {
                // Snapshot cells for undo.
                let cell_snapshot: Vec<blueprint::BlockEntry> = self.editor_grid.occupied_cells().filter_map(|&(x, y, z)| {
                    self.editor_grid.cell_info(x, y, z).map(|(bt, rot, subs, col)| {
                        blueprint::BlockEntry {
                            x, y, z,
                            block_type: bt as u8,
                            rotation: rot,
                            sub_blocks: subs,
                            color: [col.x, col.y, col.z],
                        }
                    })
                }).collect();

                if self.editor_grid.bake_group(&mut self.editor_physics) {
                    self.editor_physics.step(0.0);
                    let groups = self.editor_grid.group_count();
                    self.editor_selected_group = None;
                    self.push_undo(EditorOp::BakeGroup { cell_snapshot });
                    self.clear_extrude();
                    self.editor_status = Some((format!("Baked {} blocks into group ({})", count, groups), 3.0));
                }
            } else {
                self.editor_status = Some(("Nothing to bake".to_string(), 2.0));
            }
        }
        self.editor_bake_prev = input.toggle_ghost;

        // -------------------------------------------------------------------
        // Navigate baked groups (PageUp / PageDown)
        // -------------------------------------------------------------------
        let group_count = self.editor_grid.group_count();
        if input.editor_prev_group && !self.editor_prev_group_prev && group_count > 0 {
            let cur = self.editor_selected_group.unwrap_or(0);
            let new_idx = if cur == 0 { group_count - 1 } else { cur - 1 };
            self.editor_selected_group = Some(new_idx);
            self.editor_status = Some((format!("Group {}/{}", new_idx + 1, group_count), 2.0));
            self.editor_grid.mark_dirty();
        }

        if input.editor_next_group && !self.editor_next_group_prev && group_count > 0 {
            let cur = self.editor_selected_group.unwrap_or(group_count.wrapping_sub(1));
            let new_idx = (cur + 1) % group_count;
            self.editor_selected_group = Some(new_idx);
            self.editor_status = Some((format!("Group {}/{}", new_idx + 1, group_count), 2.0));
            self.editor_grid.mark_dirty();
        }
        self.editor_prev_group_prev = input.editor_prev_group;
        self.editor_next_group_prev = input.editor_next_group;

        // Clamp selection if groups were removed.
        if group_count == 0 {
            self.editor_selected_group = None;
        } else if let Some(sel) = self.editor_selected_group {
            if sel >= group_count {
                self.editor_selected_group = Some(group_count - 1);
            }
        }

        // -------------------------------------------------------------------
        // Unbake selected group (U)
        // -------------------------------------------------------------------
        if input.editor_unbake && !self.editor_unbake_prev {
            if let Some(gi) = self.editor_selected_group {
                let group_blocks = self.editor_grid.groups()[gi].blocks.clone();
                let count = self.editor_grid.unbake_group(&mut self.editor_physics, gi);
                self.editor_physics.step(0.0);
                self.editor_selected_group = None;
                self.push_undo(EditorOp::UnbakeGroup { group_blocks });
                self.clear_extrude();
                self.editor_status = Some((format!("Unbaked {} blocks", count), 3.0));
            } else if group_count > 0 {
                self.editor_status = Some(("Use Left/Right to select a group first".to_string(), 2.0));
            } else {
                self.editor_status = Some(("No groups to unbake".to_string(), 2.0));
            }
        }
        self.editor_unbake_prev = input.editor_unbake;

        // -------------------------------------------------------------------
        // Status message timer
        // -------------------------------------------------------------------
        if let Some((_, ref mut timer)) = self.editor_status {
            *timer -= dt;
            if *timer <= 0.0 {
                self.editor_status = None;
            }
        }
    }

    pub(crate) fn editor_save_blueprint(&mut self) {
        let cells: Vec<_> = self.editor_grid.occupied_cells().copied().collect();
        let has_groups = self.editor_grid.group_count() > 0;
        if cells.is_empty() && !has_groups {
            self.editor_status = Some(("Nothing to save".to_string(), 2.0));
            return;
        }

        // Collect all block positions (cells + groups) for normalization.
        let mut all_positions: Vec<(i32, i32, i32)> = cells.clone();
        for group in self.editor_grid.groups() {
            for b in &group.blocks {
                all_positions.push((b.x, b.y, b.z));
            }
        }

        // Normalize to origin.
        let min_x = all_positions.iter().map(|c| c.0).min().unwrap_or(0);
        let min_y = all_positions.iter().map(|c| c.1).min().unwrap_or(0);
        let min_z = all_positions.iter().map(|c| c.2).min().unwrap_or(0);

        let blocks: Vec<blueprint::BlockEntry> = cells.iter().filter_map(|&(x, y, z)| {
            self.editor_grid.cell_info(x, y, z).map(|(bt, rot, subs, col)| {
                blueprint::BlockEntry {
                    x: x - min_x,
                    y: y - min_y,
                    z: z - min_z,
                    block_type: bt as u8,
                    rotation: rot,
                    sub_blocks: subs,
                    color: [col.x, col.y, col.z],
                }
            })
        }).collect();

        // Save baked groups with normalized positions.
        let groups: Vec<Vec<blueprint::BlockEntry>> = self.editor_grid.groups().iter().map(|g| {
            g.blocks.iter().map(|b| {
                blueprint::BlockEntry {
                    x: b.x - min_x,
                    y: b.y - min_y,
                    z: b.z - min_z,
                    block_type: b.block_type,
                    rotation: b.rotation,
                    sub_blocks: b.sub_blocks,
                    color: b.color,
                }
            }).collect()
        }).collect();

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let name = format!("structure_{}", timestamp);

        let bp = blueprint::Blueprint { name: name.clone(), blocks, groups };
        match blueprint::save_blueprint(&bp) {
            Ok(path) => {
                let msg = format!("Saved: {}", path.display());
                self.editor_status = Some((msg, 3.0));
            }
            Err(e) => {
                self.editor_status = Some((format!("Save failed: {}", e), 3.0));
            }
        }
    }

    pub(crate) fn editor_load_blueprint(&mut self) {
        let files = blueprint::list_blueprints();
        if files.is_empty() {
            self.editor_status = Some(("No blueprints found".to_string(), 2.0));
            return;
        }

        // Cycle through available blueprints.
        let idx = self.editor_blueprint_idx % files.len();
        self.editor_blueprint_idx = idx + 1;

        match blueprint::load_blueprint(&files[idx]) {
            Ok(bp) => {
                // Clear current editor grid.
                self.editor_grid.clear(&mut self.editor_physics);
                self.undo_stack.clear();
                self.redo_stack.clear();
                self.clear_extrude();

                // Place all unbaked blocks from blueprint.
                for b in &bp.blocks {
                    let bt = BlockType::from_u8(b.block_type);
                    let color = Vec3::new(b.color[0], b.color[1], b.color[2]);
                    self.editor_grid.load_cell(
                        &mut self.editor_physics,
                        b.x, b.y, b.z,
                        bt, b.rotation, b.sub_blocks, color,
                    );
                }

                // Load baked groups.
                for group_blocks in &bp.groups {
                    self.editor_grid.load_group(&mut self.editor_physics, group_blocks.clone());
                }
                self.editor_physics.step(0.0);

                let name = files[idx].file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                self.editor_status = Some((format!("Loaded: {}", name), 3.0));
            }
            Err(e) => {
                self.editor_status = Some((format!("Load failed: {}", e), 3.0));
            }
        }
    }
}
