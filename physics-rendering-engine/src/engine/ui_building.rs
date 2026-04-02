use std::fmt::Write as _;

use super::*;

use crate::game::items::item_by_id;
use crate::game::quest::{self, QuestState};
use crate::ui::UiPrimitive;
use editor::EDITOR_PALETTE;
use panels::world_panels;

impl Engine {
    // -----------------------------------------------------------------------
    // UI building -- immediate-mode, runs each frame
    // -----------------------------------------------------------------------

    pub(crate) fn build_editor_ui(&mut self) {
        let sw = self.surface_width as f32;
        let sh = self.surface_height as f32;
        self.ui.begin_frame(sw, sh);

        let scale = 2.0;
        let cell = 8.0 * scale;
        let white = [0.9, 0.9, 0.9, 1.0];

        // Title.
        let title = "STRUCTURE EDITOR";
        let title_w = title.len() as f32 * cell;
        self.ui.text((sw - title_w) * 0.5, 16.0, title, scale, white);

        // Color palette at bottom.
        let swatch = 24.0;
        let gap = 4.0;
        let palette_w = EDITOR_PALETTE.len() as f32 * (swatch + gap) - gap;
        let palette_x = (sw - palette_w) * 0.5;
        let palette_y = sh - swatch - 60.0;

        for (i, color) in EDITOR_PALETTE.iter().enumerate() {
            let x = palette_x + i as f32 * (swatch + gap);
            self.ui.rect(x, palette_y, swatch, swatch, [color.x, color.y, color.z, 1.0]);
            if i == self.editor_color_idx {
                self.ui.rect_border(x - 2.0, palette_y - 2.0, swatch + 4.0, swatch + 4.0, white);
            }
        }

        // Help text (two lines).
        let help1 = "RMB: Place  LMB: Remove  B: Block  V: Rotate  G: Bake  U: Unbake  PgUp/Dn: Groups  H: Recolor";
        let help2 = "Ctrl+Z/Y: Undo/Redo  Ctrl+C/V: Copy/Paste  X/Z: Mirror  Arrows: Move  Ctrl+Arrows: Y  F9: Save  F10: Load";
        let hs = scale * 0.5;
        let help1_w = help1.len() as f32 * 8.0 * hs;
        let help2_w = help2.len() as f32 * 8.0 * hs;
        let help_color = [0.7, 0.7, 0.7, 1.0];
        self.ui.text((sw - help1_w) * 0.5, sh - 40.0, help1, hs, help_color);
        self.ui.text((sw - help2_w) * 0.5, sh - 24.0, help2, hs, help_color);

        // Block count, type, and indicators.
        self.hud_buf.clear();
        let gc = self.editor_grid.group_count();
        if let Some(sel) = self.editor_selected_group {
            let _ = write!(self.hud_buf, "Blocks: {}  Group: {}/{}  [{}] r:{}", self.editor_grid.cell_count(), sel + 1, gc, self.selected_block_type.name(), self.selected_rotation);
        } else {
            let _ = write!(self.hud_buf, "Blocks: {}  Groups: {}  [{}] r:{}", self.editor_grid.cell_count(), gc, self.selected_block_type.name(), self.selected_rotation);
        }
        // Undo/redo depth.
        if !self.undo_stack.is_empty() || !self.redo_stack.is_empty() {
            let _ = write!(self.hud_buf, "  Undo:{} Redo:{}", self.undo_stack.len(), self.redo_stack.len());
        }
        // Clipboard indicator.
        if let Some(ref clip) = self.editor_clipboard {
            let _ = write!(self.hud_buf, "  Clip:{}", clip.len());
        }
        // Extrude indicator.
        if self.editor_extrude_height > 0 {
            let _ = write!(self.hud_buf, "  Ext:+{}", self.editor_extrude_height);
        }
        self.ui.text(16.0, 16.0, &self.hud_buf, scale, white);

        // Status message.
        if let Some((ref msg, _)) = self.editor_status {
            let msg_w = msg.len() as f32 * cell;
            self.ui.text((sw - msg_w) * 0.5, 50.0, msg, scale, [0.3, 1.0, 0.3, 1.0]);
        }

        // Crosshair.
        let cx = sw * 0.5;
        let cy = sh * 0.5;
        self.ui.rect(cx - 1.0, cy - 8.0, 2.0, 16.0, [1.0, 1.0, 1.0, 0.6]);
        self.ui.rect(cx - 8.0, cy - 1.0, 16.0, 2.0, [1.0, 1.0, 1.0, 0.6]);
    }

    pub(crate) fn build_menu_ui(&mut self) {
        let sw = self.surface_width as f32;
        let sh = self.surface_height as f32;
        self.ui.begin_frame(sw, sh);

        let scale = 3.0;
        let cell = 8.0 * scale;

        // Dark overlay.
        self.ui.rect(0.0, 0.0, sw, sh, [0.02, 0.02, 0.05, 0.95]);

        // Title.
        let title = "VOXEL REALMS";
        let title_w = title.len() as f32 * cell;
        self.ui.text(sw * 0.5 - title_w * 0.5, sh * 0.25, title, scale, [0.9, 0.75, 0.2, 1.0]);

        // Subtitle.
        let sub = "An Action RPG";
        let sub_scale = 2.0;
        let sub_cell = 8.0 * sub_scale;
        let sub_w = sub.len() as f32 * sub_cell;
        self.ui.text(sw * 0.5 - sub_w * 0.5, sh * 0.25 + cell + 8.0, sub, sub_scale, [0.6, 0.6, 0.7, 1.0]);

        // Menu options.
        let options: &[&str] = if self.has_save_file {
            &["New Game", "Continue", "Settings", "Quit"]
        } else {
            &["New Game", "Settings", "Quit"]
        };

        let menu_scale = 2.0;
        let menu_cell = 8.0 * menu_scale;
        let line_h = menu_cell + 12.0;
        let menu_y = sh * 0.5;

        for (i, opt) in options.iter().enumerate() {
            let opt_w = opt.len() as f32 * menu_cell;
            let x = sw * 0.5 - opt_w * 0.5;
            let y = menu_y + i as f32 * line_h;

            let selected = i as u8 == self.menu_selection;
            let color = if selected {
                [1.0, 0.85, 0.3, 1.0]
            } else {
                [0.5, 0.5, 0.5, 1.0]
            };

            if selected {
                // Selection indicator.
                self.ui.text(x - menu_cell * 2.0, y, ">", menu_scale, color);
            }
            self.ui.text(x, y, opt, menu_scale, color);
        }

        // Controls hint.
        let hint = "W/S Navigate   E Select   ESC Quit";
        let hint_w = hint.len() as f32 * 8.0 * 1.5;
        self.ui.text(sw * 0.5 - hint_w * 0.5, sh * 0.85, hint, 1.5, [0.4, 0.4, 0.4, 1.0]);
    }

    pub(crate) fn build_settings_ui(&mut self) {
        use super::settings::SETTINGS_ENTRIES;

        let sw = self.surface_width as f32;
        let sh = self.surface_height as f32;
        self.ui.begin_frame(sw, sh);

        // Dark overlay.
        self.ui.rect(0.0, 0.0, sw, sh, [0.02, 0.02, 0.05, 0.95]);

        // Title.
        let scale = 3.0;
        let cell = 8.0 * scale;
        let title = "SETTINGS";
        let title_w = title.len() as f32 * cell;
        self.ui.text(sw * 0.5 - title_w * 0.5, sh * 0.12, title, scale, [0.9, 0.75, 0.2, 1.0]);

        // Settings entries.
        let entry_scale = 2.0;
        let entry_cell = 8.0 * entry_scale;
        let line_h = entry_cell + 14.0;
        let start_y = sh * 0.25;
        let label_x = sw * 0.25;
        let value_x = sw * 0.65;
        let bar_w = sw * 0.2;
        let bar_h = entry_cell;

        let gold = [1.0, 0.85, 0.3, 1.0];
        let white = [0.85, 0.85, 0.85, 1.0];
        let gray = [0.5, 0.5, 0.5, 1.0];
        let bar_fill = [0.4, 0.6, 0.9, 1.0];
        let bar_bg = [0.15, 0.15, 0.2, 1.0];

        for (i, entry) in SETTINGS_ENTRIES.iter().enumerate() {
            let y = start_y + i as f32 * line_h;
            let selected = i as u8 == self.settings_selection;
            let label_color = if selected { gold } else { white };

            // Selection indicator.
            if selected {
                self.ui.text(label_x - entry_cell * 2.0, y, ">", entry_scale, gold);
            }

            // Label.
            self.ui.text(label_x, y, entry.label, entry_scale, label_color);

            // Value bar.
            let cur = (entry.get)(&self.settings);
            let frac = (cur - entry.min) / (entry.max - entry.min);
            self.ui.bar(value_x, y + 2.0, bar_w, bar_h - 4.0, frac, bar_fill, bar_bg);

            // Arrows for selected.
            if selected {
                self.ui.text(value_x - entry_cell * 1.5, y, "<", entry_scale, gold);
                self.ui.text(value_x + bar_w + entry_cell * 0.5, y, ">", entry_scale, gold);
            }

            // Value text.
            let val_str = (entry.format)(cur);
            let val_w = val_str.len() as f32 * entry_cell;
            self.ui.text(value_x + (bar_w - val_w) * 0.5, y, &val_str, entry_scale,
                if selected { gold } else { gray });
        }

        // Controls hint.
        let hint = "W/S Navigate   A/D Adjust   E Back";
        let hint_w = hint.len() as f32 * 8.0 * 1.5;
        self.ui.text(sw * 0.5 - hint_w * 0.5, sh * 0.90, hint, 1.5, [0.4, 0.4, 0.4, 1.0]);
    }

    pub(crate) fn build_ui(
        &mut self,
        hp_frac: f32,
        mana_frac: f32,
        stam_frac: f32,
        level: f32,
        biome_id: f32,
        player_pos: Vec3,
    ) {
        let sw = self.surface_width as f32;
        let sh = self.surface_height as f32;
        self.ui.begin_frame(sw, sh);

        // Performance mode indicator (top-right).
        if self.perf_mode {
            self.ui.text(sw - 80.0, 8.0, "PERF", 2.0, [1.0, 0.6, 0.2, 0.8]);
        }

        let scale = 2.0;
        let cell = 8.0 * scale;
        let bar_w = 140.0;
        let bar_h = cell - 2.0;

        // -- Always-on HUD (bottom-left) --
        let hud_x = 12.0;
        let hud_y = sh - 5.0 * (cell + 2.0) - 12.0;

        // HP bar.
        self.ui.labelled_bar(
            hud_x, hud_y,
            "HP", bar_w, bar_h, hp_frac,
            [0.9, 0.9, 0.9, 1.0],
            [0.8, 0.2, 0.2, 1.0],
            [0.2, 0.05, 0.05, 0.9],
            scale,
        );

        // Mana bar.
        self.ui.labelled_bar(
            hud_x, hud_y + cell + 2.0,
            "MP", bar_w, bar_h, mana_frac,
            [0.9, 0.9, 0.9, 1.0],
            [0.2, 0.3, 0.9, 1.0],
            [0.05, 0.05, 0.2, 0.9],
            scale,
        );

        // Stamina bar.
        self.ui.labelled_bar(
            hud_x, hud_y + 2.0 * (cell + 2.0),
            "SP", bar_w, bar_h, stam_frac,
            [0.9, 0.9, 0.9, 1.0],
            [0.2, 0.8, 0.3, 1.0],
            [0.05, 0.15, 0.05, 0.9],
            scale,
        );

        // XP bar.
        let (xp, xp_frac) = if let Some(stats) = &self.world.player().stats {
            let current = stats.xp;
            let base = progression::xp_for_level(stats.level);
            let needed = progression::xp_to_next(stats.level);
            let progress = current.saturating_sub(base);
            let frac = if needed > 0 { progress as f32 / needed as f32 } else { 1.0 };
            (current, frac)
        } else {
            (0, 0.0)
        };
        let _ = xp; // used in debug
        self.ui.labelled_bar(
            hud_x, hud_y + 3.0 * (cell + 2.0),
            "XP", bar_w, bar_h, xp_frac,
            [0.9, 0.9, 0.9, 1.0],
            [0.9, 0.75, 0.2, 1.0],
            [0.15, 0.12, 0.05, 0.9],
            scale,
        );

        // Level + gold.
        let gold = self.world.player().stats.as_ref().map_or(0, |s| s.gold);
        self.hud_buf.clear();
        let _ = write!(self.hud_buf, "Lv.{}  {}g", level as u32, gold);
        self.ui.text(hud_x, hud_y + 4.0 * (cell + 2.0), &self.hud_buf, scale, [1.0, 1.0, 0.5, 1.0]);

        // Block type + rotation (shown when not default Cube/0).
        if self.selected_block_type != BlockType::Cube || self.selected_rotation != 0 {
            self.hud_buf.clear();
            let _ = write!(self.hud_buf, "[{}] r:{}", self.selected_block_type.name(), self.selected_rotation);
            self.ui.text(hud_x, hud_y + 5.0 * (cell + 2.0), &self.hud_buf, scale, [0.8, 0.9, 1.0, 1.0]);
        }

        // -- Debug overlay (F3) --
        if self.show_debug_ui {
            let ox = 12.0;
            let oy = 12.0;
            let line_h = cell + 2.0;

            let panel_w = 22.0 * cell + 16.0;
            let panel_h = 10.0 * line_h + 12.0;
            self.ui.panel(ox - 6.0, oy - 6.0, panel_w, panel_h);

            let white = [0.9, 0.9, 0.9, 1.0];

            // Performance metrics.
            let avg_dt: f32 = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
            let avg_fps = if avg_dt > 0.0 { 1.0 / avg_dt } else { 0.0 };
            let avg_ms = avg_dt * 1000.0;
            let fps_color = if avg_fps >= 55.0 { [0.3, 0.9, 0.3, 1.0] }
                            else if avg_fps >= 30.0 { [0.9, 0.8, 0.2, 1.0] }
                            else { [0.9, 0.2, 0.2, 1.0] };
            self.hud_buf.clear();
            let _ = write!(self.hud_buf, "FPS: {:.0}  ({:.1}ms)", avg_fps, avg_ms);
            self.ui.text(ox, oy, &self.hud_buf, scale, fps_color);

            self.hud_buf.clear();
            let _ = write!(self.hud_buf, "INST: {}  PART: {}", self.frame_transforms.len(), self.particles.count());
            self.ui.text(ox, oy + line_h, &self.hud_buf, scale, [0.7, 0.7, 0.7, 1.0]);

            self.hud_buf.clear();
            let _ = write!(self.hud_buf, "TORCH: {}", self.torches.len());
            self.ui.text(ox, oy + 2.0 * line_h, &self.hud_buf, scale, [0.7, 0.7, 0.7, 1.0]);

            let perf_offset = 3.0; // lines used by perf section

            let biome_name = match biome_id as u32 {
                0 => "PLAINS",
                1 => "FOREST",
                2 => "DESERT",
                3 => "MOUNTAIN",
                4 => "DUNGEON",
                _ => "CRYSTAL",
            };
            let biome_color = match biome_id as u32 {
                0 => [0.5, 0.8, 0.3, 1.0],
                1 => [0.2, 0.6, 0.2, 1.0],
                2 => [0.9, 0.8, 0.4, 1.0],
                3 => [0.7, 0.7, 0.9, 1.0],
                4 => [0.6, 0.4, 0.7, 1.0],
                _ => [0.7, 0.8, 0.95, 1.0],
            };
            let dy = perf_offset;
            self.ui.text(ox, oy + dy * line_h, "BIOME: ", scale, white);
            self.ui.text(ox + 7.0 * cell, oy + dy * line_h, biome_name, scale, biome_color);

            self.hud_buf.clear();
            let _ = write!(self.hud_buf, "LEVEL: {}", level as u32);
            self.ui.text(ox, oy + (dy + 1.0) * line_h, &self.hud_buf, scale, white);

            self.ui.labelled_bar(
                ox, oy + (dy + 2.0) * line_h,
                "HP:  ", bar_w, bar_h, hp_frac,
                white,
                [0.8, 0.2, 0.2, 1.0],
                [0.2, 0.05, 0.05, 0.9],
                scale,
            );
            self.ui.labelled_bar(
                ox, oy + (dy + 3.0) * line_h,
                "MANA:", bar_w, bar_h, mana_frac,
                white,
                [0.2, 0.3, 0.9, 1.0],
                [0.05, 0.05, 0.2, 0.9],
                scale,
            );
            self.ui.labelled_bar(
                ox, oy + (dy + 4.0) * line_h,
                "STAM:", bar_w, bar_h, stam_frac,
                white,
                [0.2, 0.8, 0.3, 1.0],
                [0.05, 0.15, 0.05, 0.9],
                scale,
            );

            self.hud_buf.clear();
            let _ = write!(self.hud_buf, "POS: {} {}", player_pos.x as i32, player_pos.z as i32);
            self.ui.text(ox, oy + (dy + 5.0) * line_h, &self.hud_buf, scale, [0.8, 0.8, 0.8, 1.0]);

            let weather_name = match self.weather.kind {
                crate::world::WeatherKind::Clear => "CLEAR",
                crate::world::WeatherKind::Cloudy => "CLOUDY",
                crate::world::WeatherKind::Rain => "RAIN",
                crate::world::WeatherKind::Thunderstorm => "STORM",
                crate::world::WeatherKind::Fog => "FOG",
                crate::world::WeatherKind::Windy => "WINDY",
            };
            let weather_color = match self.weather.kind {
                crate::world::WeatherKind::Clear => [0.9, 0.9, 0.5, 1.0],
                crate::world::WeatherKind::Cloudy => [0.7, 0.7, 0.7, 1.0],
                crate::world::WeatherKind::Rain => [0.4, 0.6, 0.9, 1.0],
                crate::world::WeatherKind::Thunderstorm => [0.9, 0.4, 0.9, 1.0],
                crate::world::WeatherKind::Fog => [0.6, 0.65, 0.7, 1.0],
                crate::world::WeatherKind::Windy => [0.5, 0.9, 0.7, 1.0],
            };
            self.ui.text(ox, oy + (dy + 6.0) * line_h, "WEATHER: ", scale, white);
            self.ui.text(ox + 9.0 * cell, oy + (dy + 6.0) * line_h, weather_name, scale, weather_color);
        }

        // -- Minimap (top-right corner) --
        {
            let map_cells = 20;
            let map_pixel = 5.0 * scale; // size of each cell on screen
            let map_size = map_cells as f32 * map_pixel;
            let map_x = sw - map_size - 12.0;
            let map_y = 12.0;

            // Background.
            self.ui.rect(map_x - 2.0, map_y - 2.0, map_size + 4.0, map_size + 4.0, [0.0, 0.0, 0.0, 0.7]);

            // Terrain grid centered on player (cached, recomputed when player moves a cell).
            let half = map_cells as f32 * 0.5;
            let world_scale = 20.0; // each map cell = 20 world units
            let cell_x = (player_pos.x / world_scale).floor() as i32;
            let cell_z = (player_pos.z / world_scale).floor() as i32;
            let minimap_dirty = (cell_x, cell_z) != self.minimap_last_cell
                || sw != self.minimap_screen_w;
            if minimap_dirty {
                self.minimap_last_cell = (cell_x, cell_z);
                self.minimap_screen_w = sw;
                self.minimap_prims.clear();
                for cy in 0..map_cells {
                    for cx in 0..map_cells {
                        let wx = player_pos.x + (cx as f32 - half) * world_scale;
                        let wz = player_pos.z + (cy as f32 - half) * world_scale;
                        let biome = self.terrain.biome_at_world(wx, wz);
                        self.minimap_biome_cache[cy * map_cells + cx] = biome as u8;
                        let color = match biome as u8 {
                            0 => [0.35, 0.55, 0.25, 0.8], // Plains
                            1 => [0.15, 0.35, 0.12, 0.8], // Forest
                            2 => [0.7, 0.6, 0.35, 0.8],   // Desert
                            3 => [0.5, 0.5, 0.55, 0.8],   // Mountains
                            4 => [0.3, 0.2, 0.35, 0.8],   // Dungeon
                            _ => [0.55, 0.5, 0.75, 0.8],   // Crystal
                        };
                        self.minimap_prims.push(UiPrimitive {
                            rect: [
                                map_x + cx as f32 * map_pixel,
                                map_y + cy as f32 * map_pixel,
                                map_pixel, map_pixel,
                            ],
                            color,
                            glyph: 0,
                            flags: 0,
                            _pad: [0; 2],
                        });
                    }
                }
            }
            self.ui.extend_prims(&self.minimap_prims);

            // Player dot (center).
            let pc = map_x + half * map_pixel;
            let pr = map_y + half * map_pixel;
            self.ui.rect(pc - 2.0, pr - 2.0, 4.0, 4.0, [1.0, 1.0, 1.0, 1.0]);

            // NPC and enemy dots -- precompute world-space radius to skip distant entities cheaply.
            let minimap_world_radius = half * world_scale; // 200 world units
            for e in self.world.entities.iter() {
                let (size, color) = match e.kind {
                    EntityKind::Npc => (2.0, [0.2, 0.8, 1.0, 1.0]),
                    EntityKind::Enemy => (1.5, [0.9, 0.2, 0.2, 1.0]),
                    _ => continue,
                };
                let epos = self.physics.body_position(e.body.rigid_body);
                // Quick world-space distance reject (avoids division for far entities).
                let wx = epos.x - player_pos.x;
                let wz = epos.z - player_pos.z;
                if wx.abs() > minimap_world_radius || wz.abs() > minimap_world_radius {
                    continue;
                }
                let dx = wx / world_scale;
                let dz = wz / world_scale;
                let mx = map_x + (dx + half) * map_pixel;
                let my = map_y + (dz + half) * map_pixel;
                self.ui.rect(mx - size, my - size, size * 2.0, size * 2.0, color);
            }
        }

        // -- Quest tracker (right side, below minimap) --
        {
            let qt_x = sw - 260.0;
            let qt_y = 12.0 + 20.0 * 5.0 * scale + 20.0; // below minimap
            let line_h_qt = cell + 2.0;
            let mut y = qt_y;

            let has_active = self.quests.iter()
                .any(|q| q.state == QuestState::Active || q.state == QuestState::Complete);

            if has_active {
                self.ui.text(qt_x, y, "QUESTS", scale, [1.0, 0.85, 0.3, 1.0]);
                y += line_h_qt;

                for q in self.quests.iter()
                    .filter(|q| q.state == QuestState::Active || q.state == QuestState::Complete)
                {
                    let status = if q.state == QuestState::Complete { "[DONE] " } else { "" };
                    self.hud_buf.clear();
                    match &q.objective {
                        quest::QuestObjective::Kill { done, needed, .. } => {
                            let _ = write!(self.hud_buf, "{}{} ({}/{})", status, q.name, done, needed);
                        }
                        quest::QuestObjective::Collect { item_id, needed } => {
                            let have = self.world.player().inventory.as_ref()
                                .map_or(0, |inv| inv.count(*item_id));
                            let _ = write!(self.hud_buf, "{}{} ({}/{})", status, q.name, have, needed);
                        }
                        quest::QuestObjective::Reach { reached, .. } => {
                            let mark = if *reached { "Y" } else { "N" };
                            let _ = write!(self.hud_buf, "{}{} ({})", status, q.name, mark);
                        }
                    }
                    let color = if q.state == QuestState::Complete {
                        [0.4, 1.0, 0.4, 1.0]
                    } else {
                        [0.8, 0.8, 0.8, 1.0]
                    };
                    // Truncate to fit.
                    let max_chars = 28;
                    let display = if self.hud_buf.len() > max_chars {
                        &self.hud_buf[..max_chars]
                    } else {
                        &self.hud_buf
                    };
                    self.ui.text(qt_x, y, display, scale, color);
                    y += line_h_qt;
                }
            }
        }

        // -- NPC Dialogue box --
        if let Some(ref dialogue) = self.active_dialogue {
            let dw = 500.0;
            let dh = 100.0;
            let dx = sw * 0.5 - dw * 0.5;
            let dy = sh - dh - 60.0;

            self.ui.panel(dx, dy, dw, dh);
            self.ui.text(dx + 8.0, dy + 4.0, dialogue.npc_name, scale, [1.0, 0.85, 0.3, 1.0]);

            let lines = dialogue.current_text();
            for (i, line) in lines.iter().enumerate() {
                self.ui.text(
                    dx + 8.0,
                    dy + 4.0 + (i as f32 + 1.5) * (cell + 2.0),
                    line,
                    scale,
                    [0.9, 0.9, 0.9, 1.0],
                );
            }
            self.ui.text(dx + 8.0, dy + dh - cell - 4.0, "[E] Continue", scale, [0.6, 0.6, 0.6, 1.0]);
        }

        // -- Teleport menu (P key) --
        if self.show_teleport {
            self.build_teleport_ui(scale, cell);
        }

        // -- Inventory screen (I key) --
        if self.show_inventory {
            self.build_inventory_ui(scale, cell);
        }

        // -- Dev menu (F12) --
        if self.show_dev_menu {
            self.build_dev_menu_ui(scale, cell);
        }
    }

    pub(crate) fn build_teleport_ui(&mut self, scale: f32, cell: f32) {
        let (sw, sh) = self.ui.screen_size();
        let line_h = cell + 2.0;
        let panels = world_panels();

        let panel_w = 20.0 * cell;
        let panel_h = (panels.len() as f32 + 2.5) * line_h + 12.0;
        let px = sw * 0.5 - panel_w * 0.5;
        let py = sh * 0.5 - panel_h * 0.5;

        self.ui.panel(px - 6.0, py - 6.0, panel_w + 12.0, panel_h + 12.0);

        let yellow = [1.0, 1.0, 0.5, 1.0];
        let white = [0.9, 0.9, 0.9, 1.0];
        let gray = [0.5, 0.5, 0.5, 1.0];
        let cyan = [0.4, 0.9, 1.0, 1.0];

        self.ui.text(px, py, "= TELEPORT =", scale, yellow);

        for (i, panel) in panels.iter().enumerate() {
            let y = py + (i as f32 + 1.5) * line_h;
            let is_current = panel.grid_x == self.panel_x && panel.grid_z == self.panel_z;

            self.hud_buf.clear();
            let _ = write!(self.hud_buf, "{}. {}", i + 1, panel.name);
            if is_current {
                self.hud_buf.push_str(" (here)");
            }

            let color = if is_current { cyan } else { white };
            self.ui.text(px, y, &self.hud_buf, scale, color);
        }

        self.ui.text(
            px,
            py + panel_h - line_h,
            "Press 1-9 to teleport, P to close",
            scale,
            gray,
        );
    }

    pub(crate) fn build_inventory_ui(&mut self, scale: f32, cell: f32) {
        let (sw, sh) = self.ui.screen_size();
        let line_h = cell + 2.0;

        // -- Equipment panel (left half) --
        let eq_w = 18.0 * cell;
        let eq_h = 10.0 * line_h + 12.0;
        let eq_x = sw * 0.25 - eq_w * 0.5;
        let eq_y = sh * 0.5 - eq_h * 0.5;

        self.ui.panel(eq_x - 6.0, eq_y - 6.0, eq_w + 12.0, eq_h + 12.0);

        let white = [0.9, 0.9, 0.9, 1.0];
        let gray = [0.5, 0.5, 0.5, 1.0];
        let yellow = [1.0, 1.0, 0.5, 1.0];

        self.ui.text(eq_x, eq_y, "= EQUIPMENT =", scale, yellow);

        let player = self.world.player();
        let eq = player.equipment.as_ref();

        let slots: [(&str, Option<u16>); 7] = if let Some(eq) = eq {
            [
                ("Weapon: ", eq.weapon),
                ("Head:   ", eq.head),
                ("Chest:  ", eq.chest),
                ("Legs:   ", eq.legs),
                ("Boots:  ", eq.boots),
                ("Ring 1: ", eq.accessory1),
                ("Ring 2: ", eq.accessory2),
            ]
        } else {
            [
                ("Weapon: ", None),
                ("Head:   ", None),
                ("Chest:  ", None),
                ("Legs:   ", None),
                ("Boots:  ", None),
                ("Ring 1: ", None),
                ("Ring 2: ", None),
            ]
        };

        for (i, (label, item_id)) in slots.iter().enumerate() {
            let y = eq_y + (i as f32 + 1.5) * line_h;
            self.ui.text(eq_x, y, label, scale, white);
            let name = item_id
                .and_then(|id| item_by_id(id))
                .map_or("---", |def| def.name);
            self.ui.text(eq_x + 8.0 * cell, y, name, scale, if item_id.is_some() { white } else { gray });
        }

        // Stats summary.
        if let Some(stats) = &player.stats {
            let stats_y = eq_y + 9.0 * line_h;
            self.hud_buf.clear();
            let _ = write!(self.hud_buf, "STR:{} INT:{} DEX:{}", stats.strength, stats.intelligence, stats.dexterity);
            self.ui.text(eq_x, stats_y, &self.hud_buf, scale, [0.7, 0.8, 0.9, 1.0]);
        }

        // -- Inventory grid (right half) --
        let inv_cols = 6;
        let inv_rows = 4;
        let slot_w = 10.0 * cell; // enough for item name
        let slot_h = line_h;
        let inv_w = inv_cols as f32 * slot_w;
        let inv_h = (inv_rows as f32 + 1.5) * slot_h + 12.0;
        let inv_x = sw * 0.75 - inv_w * 0.5;
        let inv_y = sh * 0.5 - inv_h * 0.5;

        self.ui.panel(inv_x - 6.0, inv_y - 6.0, inv_w + 12.0, inv_h + 12.0);
        self.ui.text(inv_x, inv_y, "= INVENTORY =", scale, yellow);

        let inv = player.inventory.as_ref();
        for slot_idx in 0..24 {
            let col = slot_idx % inv_cols;
            let row = slot_idx / inv_cols;
            let sx = inv_x + col as f32 * slot_w;
            let sy = inv_y + (row as f32 + 1.5) * slot_h;

            // Draw slot background.
            self.ui.rect_border(sx, sy, slot_w - 2.0, slot_h - 2.0, [0.1, 0.1, 0.1, 0.6]);

            // Build item text into hud_buf to avoid per-slot heap allocations.
            self.hud_buf.clear();
            if let Some(inv) = inv {
                if let Some(stack) = inv.slot(slot_idx) {
                    if let Some(def) = item_by_id(stack.item_id) {
                        if stack.count > 1 {
                            let _ = write!(self.hud_buf, "{}x{}", def.name, stack.count);
                        } else {
                            self.hud_buf.push_str(def.name);
                        }
                    }
                }
            }

            if !self.hud_buf.is_empty() {
                // Truncate long names to fit slot.
                let max_chars = (slot_w / cell) as usize;
                let display: &str = if self.hud_buf.len() > max_chars {
                    &self.hud_buf[..max_chars]
                } else {
                    &self.hud_buf
                };
                self.ui.text(sx + 2.0, sy + 1.0, display, scale, white);
            }
        }

        // Hint text at bottom.
        self.ui.text(
            sw * 0.5 - 10.0 * cell,
            inv_y + inv_h + 8.0,
            "Press I to close",
            scale,
            gray,
        );
    }

    pub(crate) fn build_dev_menu_ui(&mut self, scale: f32, cell: f32) {
        use super::DevMenuToggles;

        let (sw, _sh) = self.ui.screen_size();
        let line_h = cell + 4.0;
        let labels = DevMenuToggles::LABELS;
        let count = labels.len();

        let panel_w = 24.0 * cell;
        let panel_h = (count as f32 + 5.0) * line_h + 12.0;
        let px = sw - panel_w - 12.0;
        let py = 12.0;

        // Panel background.
        self.ui.panel(px - 6.0, py - 6.0, panel_w + 12.0, panel_h + 12.0);

        let gold = [1.0, 0.85, 0.3, 1.0];
        let white = [0.9, 0.9, 0.9, 1.0];
        let green = [0.3, 1.0, 0.3, 1.0];
        let red = [1.0, 0.3, 0.3, 1.0];
        let gray = [0.5, 0.5, 0.5, 1.0];

        // Title.
        self.ui.text(px, py, "DEV RENDER MENU", scale, gold);

        // FPS + frame time.
        let avg_dt: f32 = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
        let avg_fps = if avg_dt > 0.0 { 1.0 / avg_dt } else { 0.0 };
        let avg_ms = avg_dt * 1000.0;
        self.hud_buf.clear();
        let _ = write!(self.hud_buf, "{:.0} FPS  {:.2}ms  Inst: {}", avg_fps, avg_ms, self.frame_transforms.len());
        let fps_color = if avg_fps >= 55.0 { green }
                        else if avg_fps >= 30.0 { [0.9, 0.8, 0.2, 1.0] }
                        else { red };
        self.ui.text(px, py + line_h, &self.hud_buf, scale, fps_color);

        // Toggle entries.
        for (i, label) in labels.iter().enumerate() {
            let y = py + (i as f32 + 2.5) * line_h;
            let on = self.dev_toggles.get(i);
            let selected = i as u8 == self.dev_menu_selection;

            // Key hint (1-9, 0).
            let key = if i == 9 { '0' } else { (b'1' + i as u8) as char };
            self.hud_buf.clear();
            let _ = write!(self.hud_buf, "{}: {}", key, label);

            let label_color = if selected { gold } else { white };
            self.ui.text(px, y, &self.hud_buf, scale, label_color);

            // ON/OFF indicator.
            let status = if on { "ON" } else { "OFF" };
            let status_color = if on { green } else { red };
            let status_x = px + panel_w - 4.0 * cell;
            self.ui.text(status_x, y, status, scale, status_color);
        }

        // VSync toggle (separate from render categories).
        let vsync_y = py + (count as f32 + 2.5) * line_h;
        let vsync_on = self.renderer.vsync();
        self.ui.text(px, vsync_y, "V: VSync", scale, white);
        let vsync_status = if vsync_on { "ON" } else { "OFF" };
        let vsync_color = if vsync_on { green } else { red };
        self.ui.text(px + panel_w - 4.0 * cell, vsync_y, vsync_status, scale, vsync_color);

        // Hint.
        let hint_y = py + (count as f32 + 3.5) * line_h;
        self.ui.text(px, hint_y, "1-0: Toggle  V: VSync", scale, gray);
        self.ui.text(px, hint_y + line_h, "F12: Close", scale, gray);
    }
}
