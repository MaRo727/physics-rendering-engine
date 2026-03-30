use super::*;

use crate::game::quest;

impl Engine {
    pub(crate) fn do_save(&mut self) {
        let player_pos = self.physics.body_position(self.player_rb);
        let stats = match self.world.player().stats.clone() {
            Some(s) => s,
            None => return,
        };
        let inventory = self.world.player().inventory.clone().unwrap_or_default();
        let equipment = self.world.player().equipment.clone().unwrap_or_default();

        // Save building grid.
        let buildings: Vec<crate::persistence::save::BuildingSave> = self.building.occupied_cells()
            .filter_map(|&(x, y, z)| {
                self.building.cell_info(x, y, z).map(|(bt, rot, subs, col)| {
                    crate::persistence::save::BuildingSave {
                        x, y, z,
                        block_type: bt as u8,
                        rotation: rot,
                        sub_blocks: subs,
                        color: [col.x, col.y, col.z],
                    }
                })
            })
            .collect();

        let torches: Vec<crate::persistence::save::TorchSave> = self.torches.iter()
            .map(|t| crate::persistence::save::TorchSave { x: t.position.x, y: t.position.y, z: t.position.z })
            .collect();

        let data = crate::persistence::save::SaveData {
            player_x: player_pos.x,
            player_y: player_pos.y,
            player_z: player_pos.z,
            camera_yaw: self.camera.yaw,
            camera_pitch: self.camera.pitch,
            stats,
            inventory,
            equipment,
            quest_states: crate::persistence::save::quests_to_save(&self.quests),
            time_of_day: self.time_of_day,
            buildings,
            torches,
        };
        match crate::persistence::save::save(&data) {
            Ok(()) => { self.has_save_file = true; }
            Err(_) => {}
        }
    }

    pub(crate) fn do_load(&mut self) {
        let data = match crate::persistence::save::load() {
            Ok(d) => d,
            Err(e) => { println!("Load failed: {}", e); return; }
        };

        // Restore player position.
        let pos = Vec3::new(data.player_x, data.player_y, data.player_z);
        self.physics.set_body_position(self.player_rb, pos);
        self.camera.yaw = data.camera_yaw;
        self.camera.pitch = data.camera_pitch;

        // Restore stats.
        if let Some(ref mut stats) = self.world.player_mut().stats {
            *stats = data.stats;
        }
        self.world.player_mut().inventory = Some(data.inventory);
        self.world.player_mut().equipment = Some(data.equipment);

        // Restore quests.
        self.quests = quest::create_quests();
        crate::persistence::save::apply_quest_saves(&mut self.quests, &data.quest_states);

        self.time_of_day = data.time_of_day;

        // Restore buildings — clear existing and load from save.
        let old_cells: Vec<_> = self.building.occupied_cells().copied().collect();
        for (x, y, z) in old_cells {
            self.building.remove(&mut self.physics, x, y, z);
        }
        for b in &data.buildings {
            let bt = building::BlockType::from_u8(b.block_type);
            let color = Vec3::new(b.color[0], b.color[1], b.color[2]);
            self.building.load_cell(&mut self.physics, b.x, b.y, b.z, bt, b.rotation, b.sub_blocks, color);
        }

        // Restore torches.
        self.torches.clear();
        for t in &data.torches {
            let pos = Vec3::new(t.x, t.y, t.z);
            self.torches.push(TorchInstance {
                position: pos,
                flame_pos: pos + Vec3::new(0.0, TORCH_FLAME_HEIGHT, 0.0),
            });
        }

        println!("Game loaded.");
    }
}
