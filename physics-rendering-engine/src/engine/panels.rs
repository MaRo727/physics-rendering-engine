use super::*;

use crate::renderer::mesh;
use crate::world::{TERRAIN_HALF, CHUNKS_PER_SIDE};

// ---------------------------------------------------------------------------
// Ocean panel grid -- each panel is TERRAIN_HALF*2 wide and holds one island.
// ---------------------------------------------------------------------------

pub(super) struct PanelDef {
    pub(super) grid_x: i32,
    pub(super) grid_z: i32,
    pub(super) name: &'static str,
    pub(super) island: IslandDef,
}

pub(super) fn world_panels() -> &'static [PanelDef] {
    use std::sync::OnceLock;
    static PANELS: OnceLock<Vec<PanelDef>> = OnceLock::new();
    PANELS.get_or_init(|| vec![
        PanelDef {
            grid_x: 0, grid_z: 0,
            name: "Starter Island",
            island: IslandDef {
                radius: 1400.0, noise_amp: 350.0, falloff: 300.0,
                forced_biome: None, seed: 42,
            },
        },
        PanelDef {
            grid_x: 1, grid_z: 0,
            name: "Crystal Island",
            island: IslandDef {
                radius: 750.0, noise_amp: 180.0, falloff: 250.0,
                forced_biome: Some(Biome::Crystal), seed: 137,
            },
        },
    ])
}

pub(super) fn find_panel(gx: i32, gz: i32) -> Option<&'static PanelDef> {
    world_panels().iter().find(|p| p.grid_x == gx && p.grid_z == gz)
}

/// Distance from panel edge at which we start preloading the adjacent panel.
pub(super) const PRELOAD_DISTANCE: f32 = 300.0;

/// CPU-side data pre-generated on a background thread for an adjacent panel.
pub(super) struct PreloadedPanel {
    pub(super) gx: i32,
    pub(super) gz: i32,
    pub(super) terrain: TerrainGrid,
    pub(super) chunk_meshes: Vec<(Vec<mesh::Vertex>, Vec<u32>)>,
    pub(super) terrain_chunks: Vec<TerrainChunkInfo>,
    pub(super) structures: StructureGrid,
    pub(super) grass: GrassGrid,
    pub(super) trunk_colliders: Vec<(usize, Vec<(Vec3, Vec3)>)>,
}

impl Engine {
    /// Rebuild terrain chunk meshes, BLASes, and physics heightfield for dirty chunks.
    pub(crate) fn rebuild_dirty_terrain(&mut self) {
        self.terrain.drain_dirty_chunks(&mut self.dirty_chunk_buf);
        if self.dirty_chunk_buf.is_empty() {
            return;
        }

        // Regenerate chunk meshes.
        let updates: Vec<(usize, Vec<mesh::Vertex>, Vec<u32>)> = self.dirty_chunk_buf
            .iter()
            .map(|&idx| {
                let (verts, indices) = self.terrain.regenerate_chunk(idx);
                (idx, verts, indices)
            })
            .collect();

        // Update renderer (GPU mesh + BLASes).
        if let Err(e) = self.renderer.update_terrain_chunks(updates) {
            log::error!("Failed to update terrain chunks: {}", e);
        }

        // Update only the dirty chunks' physics heightfields.
        let chunk_world_size = (TERRAIN_HALF * 2) as f32 / CHUNKS_PER_SIDE as f32;
        let chunk_scale = Vec3::new(chunk_world_size, 1.0, chunk_world_size);
        for &idx in &self.dirty_chunk_buf {
            let (heights, nrows, ncols, _cx, _cz) = self.terrain.chunk_heightfield_data(idx);
            self.physics.update_heightfield_chunk(
                self.terrain_chunk_cols[idx],
                &heights,
                nrows,
                ncols,
                chunk_scale,
            );
        }
    }

    /// Spawn a background thread to pre-generate CPU-side data for an adjacent panel.
    pub(crate) fn start_panel_preload(&mut self, gx: i32, gz: i32) {
        let panel = match find_panel(gx, gz) {
            Some(p) => p,
            None => return,
        };
        // Already preloading or preloaded this panel -- skip.
        if self.preloading_panel == Some((gx, gz)) {
            return;
        }
        if let Some(ref p) = self.panel_preload {
            if p.gx == gx && p.gz == gz {
                return;
            }
        }

        log::info!("Preloading panel ({}, {}) in background...", gx, gz);
        let island = panel.island.clone();
        let (tx, rx) = mpsc::channel();
        self.panel_preload_rx = Some(rx);
        self.preloading_panel = Some((gx, gz));
        // Discard any previously completed preload for a different panel.
        self.panel_preload = None;

        std::thread::spawn(move || {
            let terrain = TerrainGrid::generate_or_load(&island);
            let (chunk_meshes, terrain_chunks, _) =
                terrain.generate_chunks(MESH_TERRAIN_BASE);
            let structures = StructureGrid::generate(island.seed, &terrain);
            let grass = GrassGrid::generate(island.seed, &terrain);
            let trunk_colliders = structures.trunk_colliders();
            let _ = tx.send(PreloadedPanel {
                gx, gz,
                terrain,
                chunk_meshes,
                terrain_chunks,
                structures,
                grass,
                trunk_colliders,
            });
        });
    }

    /// Poll the preload channel and trigger preloads for nearby panels.
    pub(crate) fn poll_panel_preload(&mut self) {
        // Check if background thread has finished.
        if let Some(ref rx) = self.panel_preload_rx {
            if let Ok(data) = rx.try_recv() {
                log::info!("Panel ({}, {}) preloaded and ready", data.gx, data.gz);
                self.panel_preload = Some(data);
                self.panel_preload_rx = None;
                self.preloading_panel = None;
            }
        }

        // Determine which boundary the player is closest to and whether to
        // start preloading an adjacent panel.
        let pos = self.physics.body_position(self.player_rb);
        let half = TERRAIN_HALF as f32;
        let threshold = half - PRELOAD_DISTANCE;

        // Pick the axis where the player is closest to the edge.
        let candidates = [
            (pos.x > threshold,  self.panel_x + 1, self.panel_z),
            (pos.x < -threshold, self.panel_x - 1, self.panel_z),
            (pos.z > threshold,  self.panel_x, self.panel_z + 1),
            (pos.z < -threshold, self.panel_x, self.panel_z - 1),
        ];
        for &(near, gx, gz) in &candidates {
            if near {
                self.start_panel_preload(gx, gz);
                return; // one preload at a time
            }
        }
    }

    /// Switch to a different ocean panel, regenerating terrain, physics, and
    /// structures. `new_x`/`new_z` are the player's position in the new panel.
    pub(crate) fn load_panel(&mut self, gx: i32, gz: i32, new_x: f32, new_z: f32) {
        let panel = match find_panel(gx, gz) {
            Some(p) => p,
            None => {
                log::warn!("No panel at ({}, {}), staying put", gx, gz);
                return;
            }
        };
        log::info!("Loading panel ({}, {})...", gx, gz);
        self.panel_x = gx;
        self.panel_z = gz;

        // Try to use preloaded data from the background thread.
        let preloaded = self.panel_preload.take().filter(|p| p.gx == gx && p.gz == gz)
            .or_else(|| {
                // If the background thread is still running for this panel, block on it.
                if self.preloading_panel == Some((gx, gz)) {
                    if let Some(rx) = self.panel_preload_rx.take() {
                        self.preloading_panel = None;
                        rx.recv().ok().filter(|p| p.gx == gx && p.gz == gz)
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

        if let Some(data) = preloaded {
            log::info!("Using preloaded data for panel ({}, {})", gx, gz);
            self.terrain = data.terrain;
            self.terrain_chunks = data.terrain_chunks;

            // Upload chunk meshes to GPU.
            let updates: Vec<(usize, Vec<mesh::Vertex>, Vec<u32>)> =
                data.chunk_meshes.into_iter().enumerate().map(|(i, (v, idx))| (i, v, idx)).collect();
            if let Err(e) = self.renderer.update_terrain_chunks(updates) {
                log::error!("Failed to update terrain chunks on panel swap: {}", e);
            }

            // Remove old tree physics, apply new structures + grass.
            for &rb in &self.tree_rbs {
                self.physics.remove_body_with_colliders(rb);
            }
            self.tree_rbs.clear();
            self.structures = data.structures;
            self.grass = data.grass;
            for (_, trunks) in &data.trunk_colliders {
                if !trunks.is_empty() {
                    let rb = self.physics.add_compound_static(trunks, crate::physics::world::cg_static());
                    self.tree_rbs.insert(rb);
                }
            }
        } else {
            log::info!("No preloaded data, generating synchronously");
            // 1. Generate new terrain.
            self.terrain = TerrainGrid::generate_or_load(&panel.island);

            // 2. Generate chunk meshes and update renderer.
            let (chunk_meshes, terrain_chunks, _) =
                self.terrain.generate_chunks(MESH_TERRAIN_BASE);
            self.terrain_chunks = terrain_chunks;

            let updates: Vec<(usize, Vec<mesh::Vertex>, Vec<u32>)> =
                chunk_meshes.into_iter().enumerate().map(|(i, (v, idx))| (i, v, idx)).collect();
            if let Err(e) = self.renderer.update_terrain_chunks(updates) {
                log::error!("Failed to update terrain chunks on panel swap: {}", e);
            }

            // 4. Regenerate structures.
            for &rb in &self.tree_rbs {
                self.physics.remove_body_with_colliders(rb);
            }
            self.tree_rbs.clear();
            self.structures = StructureGrid::generate(panel.island.seed, &self.terrain);
            self.grass = GrassGrid::generate(panel.island.seed, &self.terrain);
            for (_, trunks) in self.structures.trunk_colliders() {
                if !trunks.is_empty() {
                    let rb = self.physics.add_compound_static(&trunks, crate::physics::world::cg_static());
                    self.tree_rbs.insert(rb);
                }
            }
        }

        // 3. Update physics heightfields in-place (always needed).
        let chunk_world_size = (TERRAIN_HALF * 2) as f32 / CHUNKS_PER_SIDE as f32;
        let chunk_scale = Vec3::new(chunk_world_size, 1.0, chunk_world_size);
        for i in 0..self.terrain.chunk_count() {
            let (heights, nrows, ncols, _cx, _cz) = self.terrain.chunk_heightfield_data(i);
            self.physics.update_heightfield_chunk(
                self.terrain_chunk_cols[i],
                &heights,
                nrows,
                ncols,
                chunk_scale,
            );
        }

        // 5. Despawn all enemies.
        let enemy_ids: Vec<u32> = self.enemy_ais.keys().copied().collect();
        for id in enemy_ids {
            if let Some(e) = self.world.remove_by_id(id) {
                self.physics.remove_body(e.body.rigid_body, e.body.collider);
            }
            self.enemy_ais.remove(&id);
        }
        self.enemy_projectiles.clear();

        // 6. Reposition player.
        let spawn_y = self.terrain.height_at_world(new_x, new_z).max(6.0) + 2.0;
        self.physics.set_body_position(self.player_rb, Vec3::new(new_x, spawn_y, new_z));
        // Zero velocity so the player doesn't keep sliding.
        self.physics.set_body_linvel(self.player_rb, Vec3::ZERO);

        // Clear any in-flight preload state (we just switched panels).
        self.panel_preload = None;
        self.panel_preload_rx = None;
        self.preloading_panel = None;
    }

    /// Check if the player has crossed a panel boundary and swap if needed.
    pub(crate) fn check_panel_boundary(&mut self) {
        // Poll background preloads and trigger new ones based on proximity.
        self.poll_panel_preload();

        let pos = self.physics.body_position(self.player_rb);
        let half = TERRAIN_HALF as f32;
        let margin = 20.0; // spawn this far from the edge of the new panel

        if pos.x > half {
            self.load_panel(self.panel_x + 1, self.panel_z, -half + margin, pos.z);
        } else if pos.x < -half {
            self.load_panel(self.panel_x - 1, self.panel_z, half - margin, pos.z);
        } else if pos.z > half {
            self.load_panel(self.panel_x, self.panel_z + 1, pos.x, -half + margin);
        } else if pos.z < -half {
            self.load_panel(self.panel_x, self.panel_z - 1, pos.x, half - margin);
        }
    }
}
