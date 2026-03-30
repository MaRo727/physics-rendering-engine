use super::*;

use crate::game::player_model::FP_PART_COUNT;
use crate::renderer::{pack_instance_id, SHADOW_ONLY_BIT, MESH_CUBE, MESH_WATER, MESH_TORCH};
use crate::game::player::{extract_frustum_planes, expand_frustum_planes, is_sphere_in_frustum};


impl Engine {
    pub(crate) fn render_editor(&mut self) -> Result<()> {
        // Upload editor grid mesh if dirty (groups rendered as ghosts, selected highlighted).
        if self.editor_grid.is_dirty() {
            let (verts, indices) = self.editor_grid.generate_mesh_editor(self.editor_selected_group);
            self.renderer.update_building_mesh(&verts, &indices)?;
            self.editor_grid.clear_dirty();
        }

        let aspect = self.surface_width as f32 / self.surface_height.max(1) as f32;
        let (render_view, render_proj) = self.editor_camera.camera_matrices(aspect);

        self.frame_transforms.clear();
        self.frame_instance_ids.clear();

        // Ground plane -- large flat cube at y = -0.01.
        let ground_transform = Mat4::from_translation(Vec3::new(0.0, -0.01, 0.0))
            * Mat4::from_scale(Vec3::new(200.0, 0.02, 200.0));
        self.frame_transforms.push(ground_transform);
        // Use a muted green for the ground plane.
        // Pack with terrain object ID to reuse an existing mesh slot.
        self.frame_instance_ids.push(pack_instance_id(MESH_CUBE, 0xFFF2));

        // Building mesh (editor grid + preview).
        if (!self.editor_grid.is_empty() || self.editor_grid.has_preview()) && self.renderer.has_building_blas() {
            self.frame_transforms.push(Mat4::IDENTITY);
            self.frame_instance_ids.push(pack_instance_id(self.mesh_building_id, BUILDING_OBJECT_ID));
        }

        // Editor UI.
        self.build_editor_ui();
        self.renderer.wait_for_frame()?;
        self.renderer.upload_ui(
            self.ui.primitives(),
            self.surface_width,
            self.surface_height,
        );

        // Neutral lighting for editor.
        let light_dir = Vec4::new(0.3, 0.8, 0.2, 0.0);
        let light_color = Vec4::new(1.0, 0.98, 0.95, 1.0);
        let sun_moon = Vec4::new(0.3, 0.8, 0.2, 0.8);
        let moon_info = Vec4::new(-0.3, -0.8, -0.2, -0.8);
        let player_vp = render_proj * render_view;
        let debug_info = Vec4::ZERO;
        let debug_info2 = Vec4::ZERO;
        let perf_flag = if self.perf_mode { 1.0 } else { 0.0 };
        let blizzard_info = Vec4::new(0.0, 0.0, WATER_LEVEL, perf_flag);
        let weather_info = Vec4::ZERO;
        let wind_info = Vec4::ZERO;
        self.renderer.upload_point_lights(&[]);

        self.renderer.draw_frame(
            &self.frame_transforms,
            &self.frame_instance_ids,
            render_view,
            render_proj,
            light_dir,
            light_color,
            player_vp,
            false,
            0.0,
            0.0,
            debug_info,
            debug_info2,
            sun_moon,
            moon_info,
            blizzard_info,
            weather_info,
            wind_info,
        )
    }

    pub fn render(&mut self) -> Result<()> {
        if self.editor_mode {
            return self.render_editor();
        }

        // Rebuild building mesh on GPU if grid changed.
        if self.building.is_dirty() {
            let (verts, indices) = self.building.generate_mesh();
            self.renderer.update_building_mesh(&verts, &indices)?;
            self.building.clear_dirty();
        }

        let aspect = self.surface_width as f32 / self.surface_height.max(1) as f32;

        let (cull_view, cull_proj) = if self.ghost.active {
            (self.ghost.frozen_view, self.ghost.frozen_proj)
        } else {
            self.camera.camera_matrices(aspect)
        };

        let (render_view, render_proj) = if self.ghost.active {
            self.ghost.camera_matrices(aspect)
        } else {
            (cull_view, cull_proj)
        };

        self.frame_transforms.clear();
        self.frame_instance_ids.clear();

        // Pre-compute view-projection matrix and frustum planes once for reuse.
        let vp = cull_proj * cull_view;
        let frustum_planes = extract_frustum_planes(vp);

        // Always frustum-cull instances to reduce TLAS size.
        // In normal mode, expand the frustum by a margin so nearby off-screen
        // objects that cast ray-traced shadows are kept.
        // In ghost mode, use the tight frustum for precise debug visualization.
        let cull_planes = if self.ghost.active {
            frustum_planes
        } else {
            expand_frustum_planes(&frustum_planes, 100.0)
        };

        // World entities (skip the player entity -- we render the model instead).
        for entity in &self.world.entities {
            if entity.kind == EntityKind::Player { continue; }

            let pos = self.physics.body_position(entity.body.rigid_body);
            if !is_sphere_in_frustum(&cull_planes, pos, entity.bounding_radius) {
                continue;
            }

            let t = self.physics.body_transform(entity.body.rigid_body)
                * Mat4::from_scale(entity.render_scale);
            self.frame_transforms.push(t);
            self.frame_instance_ids.push(pack_instance_id(entity.mesh_type, entity.id));
        }

        let player_pos = self.physics.body_position(self.player_rb);

        // First-person fists (no body capsule).
        let parts = self.player_model.compute_fp_transforms(
            self.camera.eye,
            self.camera.yaw,
            self.camera.pitch,
        );
        let part_meshes = PlayerModel::fp_mesh_types();
        for (i, (transform, _scale)) in parts.iter().enumerate() {
            self.frame_transforms.push(*transform);
            self.frame_instance_ids.push(pack_instance_id(part_meshes[i], PLAYER_MODEL_OBJECT_ID + i as u32));
        }

        // Player body capsule (shadow-only -- invisible to camera, casts shadow).
        let shadow_parts = self.player_model.compute_transforms(
            player_pos,
            self.player_visual_yaw,
        );
        // Index 0 is the body capsule; skip fists (indices 1,2) as FP fists already cast shadows.
        let (body_transform, _) = shadow_parts[0];
        let body_mesh = PlayerModel::mesh_types()[0];
        self.frame_transforms.push(body_transform);
        self.frame_instance_ids.push(
            pack_instance_id(body_mesh, PLAYER_MODEL_OBJECT_ID + FP_PART_COUNT as u32)
            | SHADOW_ONLY_BIT,
        );

        // Torches.
        for (i, torch) in self.torches.iter().enumerate() {
            if !is_sphere_in_frustum(&cull_planes, torch.position, 2.0) {
                continue;
            }
            let t = Mat4::from_translation(torch.position);
            self.frame_transforms.push(t);
            self.frame_instance_ids.push(pack_instance_id(MESH_TORCH, TORCH_OBJECT_BASE + (i as u32 & 0xFF)));
        }

        // Particles.
        self.particles.render(&mut self.frame_transforms, &mut self.frame_instance_ids);

        // Spell projectiles.
        let projectile_object_base: u32 = 0xFFA0;
        for (i, proj) in self.spells.projectiles.iter().enumerate() {
            if !is_sphere_in_frustum(&cull_planes, proj.position, proj.scale * 2.0) {
                continue;
            }
            let t = Mat4::from_translation(proj.position) * Mat4::from_scale(Vec3::splat(proj.scale));
            self.frame_transforms.push(t);
            self.frame_instance_ids.push(pack_instance_id(proj.mesh_type, projectile_object_base + i as u32));
        }

        // Enemy projectiles (arrows).
        let enemy_proj_object_base: u32 = 0xFF90;
        for (i, proj) in self.enemy_projectiles.iter().enumerate() {
            if !is_sphere_in_frustum(&cull_planes, proj.position, proj.scale * 2.0) {
                continue;
            }
            // Orient arrow along its velocity.
            let dir = proj.velocity.normalize_or_zero();
            let rot = if dir.length_squared() > 0.001 {
                let up = Vec3::Y;
                let right = up.cross(dir).normalize_or_zero();
                let corrected_up = dir.cross(right);
                Mat4::from_cols(
                    Vec4::new(right.x, right.y, right.z, 0.0),
                    Vec4::new(corrected_up.x, corrected_up.y, corrected_up.z, 0.0),
                    Vec4::new(dir.x, dir.y, dir.z, 0.0),
                    Vec4::new(0.0, 0.0, 0.0, 1.0),
                )
            } else {
                Mat4::IDENTITY
            };
            let t = Mat4::from_translation(proj.position) * rot * Mat4::from_scale(Vec3::splat(proj.scale));
            self.frame_transforms.push(t);
            self.frame_instance_ids.push(pack_instance_id(proj.mesh, enemy_proj_object_base + i as u32));
        }

        // Terrain chunks -- distance cull + frustum cull.
        let terrain_cull_dist_sq: f32 = if self.perf_mode { 750.0 * 750.0 } else { 1500.0 * 1500.0 };
        for chunk in &self.terrain_chunks {
            let dx = chunk.center.x - player_pos.x;
            let dz = chunk.center.z - player_pos.z;
            if dx * dx + dz * dz > terrain_cull_dist_sq {
                continue;
            }
            if !is_sphere_in_frustum(&cull_planes, chunk.center, chunk.radius) {
                continue;
            }
            self.frame_transforms.push(Mat4::IDENTITY);
            self.frame_instance_ids.push(pack_instance_id(chunk.mesh_type, self.terrain_object_id));
        }

        // Trees near the player, frustum-culled to the player camera
        // (in ghost mode, use the frozen player frustum like terrain chunks).
        let tree_frustum = frustum_planes;
        let wind_dir = self.weather.wind_dir();
        let wind_strength = self.weather.wind_strength;
        let weather_time = self.weather.weather_time;
        self.structures.render_nearby(
            player_pos, &tree_frustum,
            wind_strength, wind_dir, weather_time,
            &mut self.frame_transforms, &mut self.frame_instance_ids,
        );

        // Leaf particles from tree punches.
        let leaf_object_base: u32 = 0xFF80;
        for (i, leaf) in self.structures.leaf_particles.iter().enumerate() {
            // Fade out: shrink during last 0.5s of life.
            let fade = (leaf.lifetime / 0.5).min(1.0);
            let s = leaf.scale * fade;
            let t = Mat4::from_translation(leaf.position)
                * Mat4::from_rotation_y(leaf.rotation_y)
                * Mat4::from_scale(Vec3::splat(s));
            self.frame_transforms.push(t);
            self.frame_instance_ids.push(pack_instance_id(leaf.mesh_type, leaf_object_base + (i as u32 & 0xFF)));
        }

        // Grass and flowers disabled for performance.

        // Water plane at WATER_LEVEL, drifting slowly for animation.
        const WAVE_PERIOD: f32 = std::f32::consts::TAU / 0.12;
        let water_offset = (self.water_time * 2.0) % WAVE_PERIOD;
        self.frame_transforms.push(Mat4::from_translation(Vec3::new(water_offset, WATER_LEVEL, water_offset * 0.6)));
        self.frame_instance_ids.push(pack_instance_id(MESH_WATER, WATER_OBJECT_ID));

        // Building mesh.
        if !self.building.is_empty() && self.renderer.has_building_blas() {
            self.frame_transforms.push(Mat4::IDENTITY);
            self.frame_instance_ids.push(pack_instance_id(self.mesh_building_id, BUILDING_OBJECT_ID));
        }

        let player_vp = vp;

        let pry_progress = self.interaction.pry_progress();
        let tool_type = match self.interaction.equipped_tool {
            crate::game::interaction::ToolType::Hands => 0.0,
            crate::game::interaction::ToolType::Axe => 1.0,
            crate::game::interaction::ToolType::Pickaxe => 2.0,
            crate::game::interaction::ToolType::Hammer => 3.0,
            crate::game::interaction::ToolType::Chisel => 4.0,
        };

        // Debug overlay data.
        let biome_id = match self.cached_player_biome {
            crate::world::Biome::Plains => 0.0,
            crate::world::Biome::Forest => 1.0,
            crate::world::Biome::Desert => 2.0,
            crate::world::Biome::Mountains => 3.0,
            crate::world::Biome::Dungeon => 4.0,
            crate::world::Biome::Crystal => 5.0,
        };
        let (hp_frac, mana_frac, stam_frac, level) = if let Some(stats) = &self.world.player().stats {
            (
                stats.health / self.player_derived.max_health,
                stats.mana / self.player_derived.max_mana,
                stats.stamina / self.player_derived.max_stamina,
                stats.level as f32,
            )
        } else {
            (1.0, 1.0, 1.0, 1.0)
        };
        let debug_info = Vec4::new(
            if self.show_debug_ui { 1.0 } else { 0.0 },
            biome_id,
            hp_frac,
            mana_frac,
        );
        let debug_info2 = Vec4::new(
            level,
            stam_frac,
            player_pos.x,
            player_pos.z,
        );

        // Compute sun/moon positions from time of day.
        // -cos gives: midnight(0)=-1, sunrise(0.25)=0, noon(0.5)=+1, sunset(0.75)=0
        let phase = self.time_of_day * std::f32::consts::TAU;
        let sun_altitude = -(phase.cos()); // -1 at midnight, +1 at noon
        // Sun east-west sweep: sin gives +1 at sunrise, 0 at noon, -1 at sunset
        let sun_x = phase.sin();
        // Sun direction in world space -- allow going below horizon so it visually sets.
        let sun_dir = Vec3::new(sun_x, sun_altitude, 0.3).normalize();

        // Moon is opposite the sun.
        let moon_altitude = -sun_altitude; // +1 at midnight, -1 at noon
        let moon_dir = Vec3::new(-sun_x, moon_altitude, -0.3).normalize();

        // Light: blend between sun and moon based on altitude.
        // Use smoothstep matching the GPU shader curve for consistent transitions.
        fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
            let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        }
        let sun_fade = smoothstep(-0.35, 0.45, sun_altitude);
        let moon_fade = smoothstep(-0.35, 0.45, moon_altitude);

        let sun_height = sun_altitude.max(0.0);
        let sunset_factor = (1.0 - sun_height * 2.0).max(0.0);
        let r = 1.0;
        let g = 0.95 - sunset_factor * 0.35;
        let b = 0.9 - sunset_factor * 0.6;
        let sun_intensity = (0.3 + sun_height.min(1.0) * 0.7) * sun_fade;
        let sun_color = Vec4::new(r, g, b, sun_intensity);

        let moon_height = moon_altitude.max(0.0);
        let moon_intensity = (0.08 + moon_height * 0.12) * moon_fade;
        let moon_color = Vec4::new(0.6, 0.7, 1.0, moon_intensity);

        // Blend light direction and color smoothly between sun and moon.
        // A constant fill light ensures the total never hits zero, eliminating
        // the hard cutoff that caused a visible pop at horizon transitions.
        let fill_intensity = 0.02_f32;
        let fill_dir = Vec3::new(0.0, 1.0, 0.0);
        let fill_color = Vec4::new(0.5, 0.5, 0.7, fill_intensity);

        let total = sun_intensity + moon_intensity + fill_intensity;
        let sun_weight = sun_intensity / total;
        let moon_weight = moon_intensity / total;
        let fill_weight = fill_intensity / total;

        let blended_dir =
            (sun_dir * sun_weight + moon_dir * moon_weight + fill_dir * fill_weight).normalize();
        let light_dir = blended_dir;
        let light_color = Vec4::new(
            sun_color.x * sun_weight + moon_color.x * moon_weight + fill_color.x * fill_weight,
            sun_color.y * sun_weight + moon_color.y * moon_weight + fill_color.y * fill_weight,
            sun_color.z * sun_weight + moon_color.z * moon_weight + fill_color.z * fill_weight,
            total,
        );

        // Pack sun and moon directions for shader disc rendering.
        // sunMoon.xyz = sun direction, sunMoon.w = sun altitude for sky color.
        let sun_moon = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, sun_altitude);
        // Repurpose ghostMode.w (was unused 0.0) for moon direction packing won't work cleanly.
        // Instead pass moon dir via lightDir.w (was 0.0) -- but lightDir is already used.
        // Simplest: add a second vec4 for moon.
        let moon_info = Vec4::new(moon_dir.x, moon_dir.y, moon_dir.z, moon_altitude);

        let perf_flag = if self.perf_mode { 1.0 } else { 0.0 };
        let blizzard_info = Vec4::new(self.snow_intensity, self.snow_time, WATER_LEVEL, perf_flag);

        let (wd_x, wd_z) = self.weather.wind_dir();
        let weather_info = Vec4::new(
            self.weather.rain_intensity,
            self.weather.fog_density,
            self.weather.lightning_flash,
            self.weather.cloud_coverage,
        );
        let wind_info = Vec4::new(
            self.weather.wind_strength,
            wd_x,
            wd_z,
            self.weather.weather_time,
        );

        // Build UI for this frame.
        if self.game_state == GameState::MainMenu {
            self.build_menu_ui();
        } else {
            self.build_ui(hp_frac, mana_frac, stam_frac, level, biome_id, player_pos);
        }
        // Ensure the GPU finished the previous use of this frame's buffers
        // before writing UI / point-light data into the mapped memory.
        self.renderer.wait_for_frame()?;
        self.renderer.upload_ui(
            self.ui.primitives(),
            self.surface_width,
            self.surface_height,
        );

        // Collect point lights from nearest torches (cap at 8 for performance).
        self.frame_point_lights.clear();
        let time = self.water_time;
        let cam_pos = if self.ghost.active { self.ghost.eye } else { self.camera.eye };
        // Collect distances for sorting (reuse buffer to avoid per-frame allocation).
        self.frame_torch_dists.clear();
        self.frame_torch_dists.extend(
            self.torches.iter().enumerate()
                .map(|(i, t)| (i, t.flame_pos.distance_squared(cam_pos)))
                .filter(|(_, d2)| *d2 < 60.0 * 60.0) // within 60m
        );
        self.frame_torch_dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for &(idx, _) in self.frame_torch_dists.iter().take(8) {
            let torch = &self.torches[idx];
            let flicker = 0.9 + 0.1 * (time * 8.0 + torch.position.x * 1.7).sin();
            self.frame_point_lights.push(GpuPointLight {
                position: [torch.flame_pos.x, torch.flame_pos.y, torch.flame_pos.z],
                radius: 15.0,
                color: [1.0, 0.6, 0.2],
                intensity: 2.5 * flicker,
            });
        }
        self.renderer.upload_point_lights(&self.frame_point_lights);

        self.renderer.draw_frame(
            &self.frame_transforms,
            &self.frame_instance_ids,
            render_view,
            render_proj,
            Vec4::from((light_dir, 0.0)),
            light_color,
            player_vp,
            self.ghost.active,
            pry_progress,
            tool_type,
            debug_info,
            debug_info2,
            sun_moon,
            moon_info,
            blizzard_info,
            weather_info,
            wind_info,
        )
    }
}
