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

impl Engine {
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

        // Block rotation (V).
        let rotate = input.rotate_block && !self.rotate_prev;
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

        // Drag-to-fill placement (RMB).
        // Quick click places a single block. Hold for 0.5s to enter drag mode.
        const DRAG_THRESHOLD: f32 = 0.5;
        let place_press = input.place && !self.place_prev;
        let place_release = !input.place && self.place_prev;
        self.place_prev = input.place;

        if place_press {
            // Immediately place a single block on click.
            if let Some((_rb, hit_pos, normal)) = self.editor_physics.cast_ray_unfiltered(cam_eye, look_dir, 50.0) {
                let target = hit_pos + normal * 0.01;
                let (cx, cy, cz) = building::snap_to_grid(target);
                let color = EDITOR_PALETTE[self.editor_color_idx];
                self.editor_grid.place_unsupported(
                    &mut self.editor_physics, cx, cy, cz,
                    self.selected_block_type, self.selected_rotation, color,
                );
                self.editor_physics.step(0.0);
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
                    let mut placed = 0u32;
                    for &((cx, cy, cz), block_type, rotation, col) in &region {
                        if self.editor_grid.place_unsupported(
                            &mut self.editor_physics, cx, cy, cz,
                            block_type, rotation, col,
                        ) {
                            placed += 1;
                        }
                    }
                    if placed > 0 {
                        self.editor_physics.step(0.0);
                    }
                    if placed > 1 {
                        self.editor_status = Some((format!("Placed {} blocks", placed), 2.0));
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

        // Remove block (LMB -- edge triggered).
        let throw_edge = input.throw && !self.editor_throw_prev;
        self.editor_throw_prev = input.throw;
        if throw_edge {
            if let Some((_rb, hit_pos, normal)) = self.editor_physics.cast_ray_unfiltered(cam_eye, look_dir, 50.0) {
                let target = hit_pos - normal * 0.01;
                let (cx, cy, cz) = building::snap_to_grid(target);
                if cy >= 0 { // don't remove ground
                    if self.editor_grid.remove(&mut self.editor_physics, cx, cy, cz) {
                        self.editor_physics.step(0.0);
                    }
                }
            }
        }

        // Save (F8).
        if input.editor_save && !self.editor_save_prev {
            self.editor_save_blueprint();
        }
        self.editor_save_prev = input.editor_save;

        // Load (F9).
        if input.editor_load && !self.editor_load_prev {
            self.editor_load_blueprint();
        }
        self.editor_load_prev = input.editor_load;

        // Bake group (G) -- merge current cells into a single object.
        if input.toggle_ghost && !self.editor_bake_prev {
            let count = self.editor_grid.cell_count();
            if self.editor_grid.bake_group(&mut self.editor_physics) {
                self.editor_physics.step(0.0);
                let groups = self.editor_grid.group_count();
                self.editor_selected_group = None;
                self.editor_status = Some((format!("Baked {} blocks into group ({})", count, groups), 3.0));
            } else {
                self.editor_status = Some(("Nothing to bake".to_string(), 2.0));
            }
        }
        self.editor_bake_prev = input.toggle_ghost;

        // Navigate baked groups (Left/Right arrows).
        let group_count = self.editor_grid.group_count();
        if input.editor_prev_group && !self.editor_prev_group_prev && group_count > 0 {
            let cur = self.editor_selected_group.unwrap_or(0);
            let new_idx = if cur == 0 { group_count - 1 } else { cur - 1 };
            self.editor_selected_group = Some(new_idx);
            self.editor_status = Some((format!("Group {}/{}", new_idx + 1, group_count), 2.0));
            self.editor_grid.mark_dirty(); // redraw with highlight
        }
        self.editor_prev_group_prev = input.editor_prev_group;

        if input.editor_next_group && !self.editor_next_group_prev && group_count > 0 {
            let cur = self.editor_selected_group.unwrap_or(group_count.wrapping_sub(1));
            let new_idx = (cur + 1) % group_count;
            self.editor_selected_group = Some(new_idx);
            self.editor_status = Some((format!("Group {}/{}", new_idx + 1, group_count), 2.0));
            self.editor_grid.mark_dirty(); // redraw with highlight
        }
        self.editor_next_group_prev = input.editor_next_group;

        // Clamp selection if groups were removed.
        if group_count == 0 {
            self.editor_selected_group = None;
        } else if let Some(sel) = self.editor_selected_group {
            if sel >= group_count {
                self.editor_selected_group = Some(group_count - 1);
            }
        }

        // Unbake selected group (U) -- restore to editable cells.
        if input.editor_unbake && !self.editor_unbake_prev {
            if let Some(gi) = self.editor_selected_group {
                let count = self.editor_grid.unbake_group(&mut self.editor_physics, gi);
                self.editor_physics.step(0.0);
                self.editor_selected_group = None;
                self.editor_status = Some((format!("Unbaked {} blocks", count), 3.0));
            } else if group_count > 0 {
                self.editor_status = Some(("Use Left/Right to select a group first".to_string(), 2.0));
            } else {
                self.editor_status = Some(("No groups to unbake".to_string(), 2.0));
            }
        }
        self.editor_unbake_prev = input.editor_unbake;

        // Status message timer.
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
