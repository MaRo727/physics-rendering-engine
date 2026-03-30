pub(crate) mod panels;
pub(crate) mod editor;
pub(crate) mod rendering;
pub(crate) mod ui_building;
pub(crate) mod save_load;

use panels::{PreloadedPanel, find_panel, world_panels};
use editor::DragPlane;

use std::collections::HashMap;
use std::sync::mpsc;

use anyhow::Result;
use glam::{Mat4, Vec3, Vec4};

use crate::audio::AudioManager;
use crate::persistence::blueprint;
use crate::building::{self, BlockType, BuildingGrid};
use crate::game::camera::FirstPersonCamera;
use crate::game::combat::CombatSystem;
use crate::game::enemy_ai::{self, EnemyAi, EnemyAttackHit, EnemyProjectile};
use crate::game::entity::{Entity, EntityId, EntityKind};
use crate::game::items::ITEM_IRON_SWORD;
use crate::game::player_model::{PlayerModel, FP_PART_COUNT};
use crate::game::progression;
use crate::game::spells::{SpellSystem, SpellHit, CastResult};
use crate::game::stats::{DerivedStats, StatBlock, StatBonuses};
use crate::game::world::World;
use crate::input::InputState;
use crate::game::interaction::{Interaction, ToolType, PickaxeHit, HammerHit, ChiselHit};
use crate::mining::MiningSystem;
use crate::physics::body::{PhysicsBody, WeightClass};
use crate::physics::world::PhysicsWorld;
use crate::game::player::GhostCamera;
use crate::renderer::{Renderer, GpuPointLight, MESH_CUBE, MESH_TERRAIN_BASE};
use crate::renderer::context::VulkanContext;
use crate::renderer::swapchain::Swapchain;
use crate::game::entity::UNIT_BOUNDING_RADIUS;
use crate::world::{StructureGrid, GrassGrid};
use crate::world::{Biome, IslandDef, TerrainGrid, TerrainChunkInfo, TERRAIN_HALF, CHUNKS_PER_SIDE};
use crate::particles::ParticleSystem;
use crate::ui::{Ui, UiPrimitive};
use crate::world::Weather;
use crate::game::npc::{self, ActiveDialogue};
use crate::game::quest::{self, Quest, QuestState};

/// Build the default scene. Returns (entities, player_entity_id, next_entity_id).
fn build_scene(physics: &mut PhysicsWorld) -> (Vec<Entity>, EntityId, EntityId) {
    use crate::physics::body::PhysicsBody;

    let mut entities: Vec<Entity> = Vec::new();
    let mut next_id: EntityId = 0;
    let mut alloc_id = || { let id = next_id; next_id += 1; id };

    // --- Player ---
    let player_id = alloc_id();
    let player_body = PhysicsBody::new_player_box(physics, Vec3::new(0.0, 0.9, 4.0), Vec3::new(0.4, 0.9, 0.4));
    let player_entity = Entity::player(player_id, player_body);
    entities.push(player_entity);

    (entities, player_id, next_id)
}

pub(crate) const PLACE_RANGE: f32 = 8.0;

pub(crate) const BUILDING_OBJECT_ID: u32 = 0xFFF0;
pub(crate) const PLAYER_MODEL_OBJECT_ID: u32 = 0xFFE0;
pub(crate) const WATER_OBJECT_ID: u32 = 0xFFD0;
pub(crate) const PLAYER_SPEED: f32 = 5.0;
pub(crate) const SPRINT_SPEED: f32 = 9.0;
pub(crate) const FAST_SPEED: f32 = 40.0;
pub(crate) const JUMP_VELOCITY: f32 = 6.0;
pub(crate) const WATER_LEVEL: f32 = 5.0;
pub(crate) const BUOYANCY_FORCE: f32 = 25.0;
pub(crate) const WATER_DRAG: f32 = 12.0;

pub(crate) const TORCH_OBJECT_BASE: u32 = 0xFF70;
pub(crate) const TORCH_FLAME_HEIGHT: f32 = 0.85;

pub struct TorchInstance {
    pub position: Vec3,
    pub flame_pos: Vec3,
}

/// Minimum interval between terrain GPU/physics rebuilds (seconds).
pub(crate) const TERRAIN_REBUILD_INTERVAL: f32 = 0.1;
/// Radius of terrain deformation per pickaxe hit (world units).
pub(crate) const DEFORM_RADIUS: f32 = 3.0;
/// Height lowered at the center of a pickaxe hit.
pub(crate) const DEFORM_AMOUNT: f32 = 0.5;

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct EngineConfig {
    pub window_width: u32,
    pub window_height: u32,
    pub window_title: String,
    pub gravity: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameState {
    MainMenu,
    Playing,
}

pub struct Engine {
    pub config: EngineConfig,
    pub(crate) game_state: GameState,
    pub(crate) menu_selection: u8, // 0=New Game, 1=Continue, 2=Quit
    pub(crate) physics: PhysicsWorld,
    pub(crate) world: World,
    pub(crate) player_rb: rapier3d::prelude::RigidBodyHandle,
    pub(crate) player_col: rapier3d::prelude::ColliderHandle,
    pub(crate) player_on_ground: bool,
    pub(crate) camera: FirstPersonCamera,
    pub(crate) player_model: PlayerModel,
    pub(crate) renderer: Renderer,
    pub(crate) interaction: Interaction,
    pub(crate) ghost: GhostCamera,
    pub(crate) terrain: TerrainGrid,
    pub(crate) terrain_chunks: Vec<TerrainChunkInfo>,
    pub(crate) terrain_object_id: u32,
    pub(crate) terrain_rbs: std::collections::HashSet<rapier3d::prelude::RigidBodyHandle>,
    pub(crate) terrain_chunk_cols: Vec<rapier3d::prelude::ColliderHandle>,
    pub(crate) terrain_rebuild_timer: f32,
    pub(crate) panel_x: i32,
    pub(crate) panel_z: i32,
    pub(crate) mesh_building_id: u32,
    pub(crate) building: BuildingGrid,
    pub(crate) mining: MiningSystem,
    pub(crate) structures: StructureGrid,
    pub(crate) grass: GrassGrid,
    pub(crate) place_prev: bool,
    pub(crate) spawn_prev: bool,
    pub(crate) selected_block_type: BlockType,
    pub(crate) selected_rotation: u8,
    pub(crate) cycle_block_prev: bool,
    pub(crate) rotate_prev: bool,
    pub(crate) debug_stats_prev: bool,
    pub(crate) fast_prev: bool,
    pub(crate) fast_mode: bool,
    pub(crate) god_mode: bool,
    pub(crate) god_prev: bool,
    pub(crate) show_debug_ui: bool,
    pub(crate) debug_ui_prev: bool,
    /// Rolling frame time buffer for FPS display.
    pub(crate) frame_times: [f32; 60],
    pub(crate) frame_time_idx: usize,
    pub(crate) show_inventory: bool,
    pub(crate) inventory_prev: bool,
    pub(crate) show_teleport: bool,
    pub(crate) teleport_prev: bool,
    pub(crate) mute_prev: bool,
    pub(crate) fast_time: bool,
    pub(crate) fast_time_prev: bool,
    pub(crate) water_time: f32,
    pub(crate) time_of_day: f32, // 0..1 where 0=midnight, 0.25=sunrise, 0.5=noon, 0.75=sunset
    pub(crate) surface_width: u32,
    pub(crate) surface_height: u32,
    pub(crate) audio: Option<AudioManager>,
    pub(crate) combat: CombatSystem,
    pub(crate) spells: SpellSystem,
    pub(crate) cast_prev: bool,
    pub(crate) enemy_ais: HashMap<u32, EnemyAi>,
    pub(crate) enemy_projectiles: Vec<EnemyProjectile>,
    pub(crate) spawn_timer: f32,
    pub(crate) spawn_seed: u32,
    pub(crate) player_visual_yaw: f32,
    pub(crate) snow_intensity: f32,
    pub(crate) snow_time: f32,
    pub(crate) weather: Weather,
    pub(crate) weather_debug_active: bool,
    pub(crate) weather_prev: bool,
    pub(crate) perf_mode: bool,
    pub(crate) perf_prev: bool,
    pub(crate) wind_leaf_timer: f32,
    pub(crate) tree_rbs: std::collections::HashSet<rapier3d::prelude::RigidBodyHandle>,
    pub(crate) tree_punch_seed: u32,
    // Cached per-frame player derived stats (computed once in update, reused in render).
    pub(crate) player_derived: DerivedStats,
    // Reusable per-frame buffers (avoid heap allocs every frame).
    pub(crate) frame_transforms: Vec<Mat4>,
    pub(crate) frame_instance_ids: Vec<u32>,
    pub(crate) buoyancy_bodies: Vec<(rapier3d::prelude::RigidBodyHandle, f32)>,
    pub(crate) dead_ids: Vec<(u32, u32)>,
    pub(crate) despawn_ids: Vec<u32>,
    // Reusable dirty-chunk index buffer (avoid per-frame heap allocs).
    pub(crate) dirty_chunk_buf: Vec<usize>,
    // Reusable hit-result buffers (avoid per-frame heap allocs).
    pub(crate) enemy_hit_buf: Vec<EnemyAttackHit>,
    pub(crate) arrow_hit_buf: Vec<EnemyAttackHit>,
    pub(crate) spell_hit_buf: Vec<SpellHit>,
    pub(crate) ui: Ui,
    pub(crate) particles: ParticleSystem,
    pub(crate) quests: Vec<Quest>,
    pub(crate) active_dialogue: Option<ActiveDialogue>,
    pub(crate) npc_interact_prev: bool,
    pub(crate) save_prev: bool,
    pub(crate) load_prev: bool,
    pub(crate) has_save_file: bool,
    // Minimap biome cache: 20x20 grid, recomputed only when player moves a cell.
    pub(crate) minimap_biome_cache: [u8; 400],
    pub(crate) minimap_last_cell: (i32, i32),
    pub(crate) minimap_prims: Vec<UiPrimitive>,
    pub(crate) minimap_screen_w: f32,
    // Reusable string buffer for HUD text (avoids format! heap allocs each frame).
    pub(crate) hud_buf: String,
    // Structure editor state.
    pub(crate) editor_mode: bool,
    pub(crate) editor_prev: bool,
    pub(crate) editor_grid: BuildingGrid,
    pub(crate) editor_physics: PhysicsWorld,
    pub(crate) editor_camera: GhostCamera,
    pub(crate) editor_color_idx: usize,
    pub(crate) editor_save_prev: bool,
    pub(crate) editor_load_prev: bool,
    pub(crate) editor_blueprint_idx: usize,
    pub(crate) editor_status: Option<(String, f32)>, // (message, remaining_seconds)
    pub(crate) editor_ground_inited: bool,
    pub(crate) editor_throw_prev: bool,
    pub(crate) editor_bake_prev: bool,
    pub(crate) editor_selected_group: Option<usize>,
    pub(crate) editor_prev_group_prev: bool,
    pub(crate) editor_next_group_prev: bool,
    pub(crate) editor_unbake_prev: bool,
    // Undo/redo.
    pub(crate) undo_stack: Vec<editor::EditorOp>,
    pub(crate) redo_stack: Vec<editor::EditorOp>,
    pub(crate) editor_undo_prev: bool,
    pub(crate) editor_redo_prev: bool,
    // Copy/paste/mirror.
    pub(crate) editor_clipboard: Option<Vec<crate::persistence::blueprint::BlockEntry>>,
    pub(crate) editor_copy_prev: bool,
    pub(crate) editor_paste_prev: bool,
    pub(crate) editor_mirror_x_prev: bool,
    pub(crate) editor_mirror_z_prev: bool,
    // Replace color.
    pub(crate) editor_replace_color_prev: bool,
    // Vertical extrude / move group (arrow keys).
    pub(crate) editor_last_fill: Option<(Vec<(i32, i32, i32)>, BlockType, u8, Vec3)>,
    pub(crate) editor_extrude_height: i32,
    pub(crate) editor_arrow_up_prev: bool,
    pub(crate) editor_arrow_down_prev: bool,
    pub(crate) editor_arrow_left_prev: bool,
    pub(crate) editor_arrow_right_prev: bool,
    // Drag-to-fill state.
    pub(crate) drag_start: Option<(i32, i32, i32)>,
    pub(crate) drag_plane: Option<DragPlane>,
    pub(crate) drag_end: Option<(i32, i32, i32)>,
    /// Time RMB has been held; drag activates after threshold.
    pub(crate) drag_hold_timer: f32,
    /// Whether the drag has activated (hold exceeded threshold).
    pub(crate) drag_active: bool,
    // Torch instances placed in the world.
    pub(crate) torches: Vec<TorchInstance>,
    pub(crate) torch_prev: bool,
    /// Reusable per-frame point light buffer.
    pub(crate) frame_point_lights: Vec<GpuPointLight>,
    /// Reusable per-frame torch distance buffer (avoids allocation each frame).
    pub(crate) frame_torch_dists: Vec<(usize, f32)>,
    /// Time accumulator for fire particle emission (fixed rate).
    pub(crate) fire_particle_timer: f32,
    /// Cached player biome from update(), reused in render() to avoid redundant lookups.
    pub(crate) cached_player_biome: Biome,
    // Panel preloading: background thread generates CPU data for adjacent panels.
    pub(crate) panel_preload: Option<PreloadedPanel>,
    pub(crate) panel_preload_rx: Option<mpsc::Receiver<PreloadedPanel>>,
    pub(crate) preloading_panel: Option<(i32, i32)>,
}

/// Number of progress steps reported by `init_world`.
pub const INIT_STEPS: u32 = 6;

/// Data produced by the background init thread, consumed by `Engine::from_init_data`.
pub struct InitData {
    pub config: EngineConfig,
    pub physics: PhysicsWorld,
    pub world: World,
    pub player_rb: rapier3d::prelude::RigidBodyHandle,
    pub player_col: rapier3d::prelude::ColliderHandle,
    pub camera: FirstPersonCamera,
    pub player_model: PlayerModel,
    pub terrain: TerrainGrid,
    pub terrain_chunks: Vec<TerrainChunkInfo>,
    pub terrain_rbs: std::collections::HashSet<rapier3d::prelude::RigidBodyHandle>,
    pub terrain_chunk_cols: Vec<rapier3d::prelude::ColliderHandle>,
    pub structures: StructureGrid,
    pub grass: GrassGrid,
    pub tree_rbs: std::collections::HashSet<rapier3d::prelude::RigidBodyHandle>,
    pub chunk_meshes: Vec<(Vec<crate::renderer::mesh::Vertex>, Vec<u32>)>,
    pub num_terrain_chunks: usize,
}

/// Run all heavy CPU-side initialisation on a background thread.
/// Bumps `progress` (0 → INIT_STEPS) as each stage completes.
pub fn init_world(
    config: EngineConfig,
    progress: std::sync::Arc<std::sync::atomic::AtomicU32>,
) -> Result<InitData> {
    use std::sync::atomic::Ordering;

    let mut physics = PhysicsWorld::new(config.gravity);
    let (entities, player_id, next_id) = build_scene(&mut physics);
    progress.store(1, Ordering::Relaxed);

    // Generate terrain for the starter island panel.
    let starter_panel = find_panel(0, 0).expect("starter panel must exist");
    let terrain = TerrainGrid::generate_or_load(&starter_panel.island);
    progress.store(2, Ordering::Relaxed);

    let (chunk_meshes, terrain_chunks, _full_mesh) =
        terrain.generate_chunks(MESH_TERRAIN_BASE);
    let num_terrain_chunks = chunk_meshes.len();
    progress.store(3, Ordering::Relaxed);

    // Add per-chunk heightfield colliders for terrain.
    let chunk_world_size = (TERRAIN_HALF * 2) as f32 / CHUNKS_PER_SIDE as f32;
    let chunk_scale = Vec3::new(chunk_world_size, 1.0, chunk_world_size);
    let mut terrain_rbs = std::collections::HashSet::new();
    let mut terrain_chunk_cols = Vec::with_capacity(terrain.chunk_count());
    for i in 0..terrain.chunk_count() {
        let (heights, nrows, ncols, cx, cz) = terrain.chunk_heightfield_data(i);
        let (rb, col) = physics.add_heightfield_chunk(&heights, nrows, ncols, chunk_scale, cx, cz);
        terrain_rbs.insert(rb);
        terrain_chunk_cols.push(col);
    }

    // Build game world.
    let world = World::new(entities, player_id, next_id);

    // Extract player physics handles.
    let player_rb = world.player().body.rigid_body;
    let player_col = world.player().body.collider;

    // Spawn player on terrain surface.
    let spawn_h = terrain.get_height(0, 4) + 1.0 + 0.9;
    physics.set_body_position(player_rb, Vec3::new(0.0, spawn_h, 4.0));

    let camera = FirstPersonCamera::new();
    let player_model = PlayerModel::new();
    progress.store(4, Ordering::Relaxed);

    // Generate structures (trees, ruins) and grass on terrain.
    let structures = StructureGrid::generate(42, &terrain);
    let grass = GrassGrid::generate(42, &terrain);
    progress.store(5, Ordering::Relaxed);

    // Add tree trunk colliders (compound collider per chunk).
    let mut tree_rbs = std::collections::HashSet::new();
    for (_, trunks) in structures.trunk_colliders() {
        if !trunks.is_empty() {
            let rb = physics.add_compound_static(&trunks, crate::physics::world::cg_static());
            tree_rbs.insert(rb);
        }
    }
    progress.store(6, Ordering::Relaxed);

    Ok(InitData {
        config,
        physics,
        world,
        player_rb,
        player_col,
        camera,
        player_model,
        terrain,
        terrain_chunks,
        terrain_rbs,
        terrain_chunk_cols,
        structures,
        grass,
        tree_rbs,
        chunk_meshes,
        num_terrain_chunks,
    })
}

impl Engine {
    /// Assemble the engine from pre-computed init data and Vulkan resources.
    pub fn from_init_data(
        data: InitData,
        context: VulkanContext,
        swapchain: Swapchain,
    ) -> Result<Self> {
        let surface_width = data.config.window_width;
        let surface_height = data.config.window_height;

        let max_instances = (data.world.entities.len() + data.num_terrain_chunks + 3
            + FP_PART_COUNT + 32768 + 16384 + 512) as u32;
        let renderer = Renderer::new(context, swapchain, max_instances, data.chunk_meshes)?;
        let mesh_building_id = renderer.mesh_building_id();

        let mining = MiningSystem::new();
        let mut engine = Self {
            config: data.config,
            game_state: GameState::MainMenu,
            menu_selection: 0,
            physics: data.physics,
            world: data.world,
            player_rb: data.player_rb,
            player_col: data.player_col,
            player_on_ground: false,
            camera: data.camera,
            player_model: data.player_model,
            renderer,
            interaction: Interaction::default(),
            ghost: GhostCamera::default(),
            terrain: data.terrain,
            terrain_chunks: data.terrain_chunks,
            terrain_object_id: 0xFFF1,
            terrain_rbs: data.terrain_rbs,
            terrain_chunk_cols: data.terrain_chunk_cols,
            terrain_rebuild_timer: 0.0,
            panel_x: 0,
            panel_z: 0,
            mesh_building_id,
            building: BuildingGrid::new(),
            mining,
            structures: data.structures,
            grass: data.grass,
            place_prev: false,
            spawn_prev: false,
            selected_block_type: BlockType::Cube,
            selected_rotation: 0,
            cycle_block_prev: false,
            rotate_prev: false,
            debug_stats_prev: false,
            fast_prev: false,
            fast_mode: false,
            god_mode: false,
            god_prev: false,
            show_debug_ui: false,
            debug_ui_prev: false,
            frame_times: [0.016; 60],
            frame_time_idx: 0,
            show_inventory: false,
            inventory_prev: false,
            show_teleport: false,
            teleport_prev: false,
            mute_prev: false,
            fast_time: false,
            fast_time_prev: false,
            water_time: 0.0,
            time_of_day: 0.35,
            surface_width,
            surface_height,
            audio: AudioManager::new(std::path::Path::new("../assets")),
            combat: CombatSystem::new(),
            spells: SpellSystem::new(),
            cast_prev: false,
            enemy_ais: HashMap::new(),
            enemy_projectiles: Vec::new(),
            spawn_timer: 0.0,
            spawn_seed: 42,
            player_visual_yaw: 0.0,
            snow_intensity: 0.0,
            snow_time: 0.0,
            weather: Weather::new(42),
            weather_debug_active: false,
            weather_prev: false,
            perf_mode: false,
            perf_prev: false,
            wind_leaf_timer: 0.0,
            tree_rbs: data.tree_rbs,
            tree_punch_seed: 12345,
            player_derived: StatBlock::new_player().compute_derived(&StatBonuses::default()),
            frame_transforms: Vec::with_capacity(max_instances as usize),
            frame_instance_ids: Vec::with_capacity(max_instances as usize),
            buoyancy_bodies: Vec::new(),
            dead_ids: Vec::new(),
            despawn_ids: Vec::new(),
            dirty_chunk_buf: Vec::new(),
            enemy_hit_buf: Vec::new(),
            arrow_hit_buf: Vec::new(),
            spell_hit_buf: Vec::new(),
            ui: Ui::new(),
            particles: ParticleSystem::new(),
            quests: quest::create_quests(),
            active_dialogue: None,
            npc_interact_prev: false,
            save_prev: false,
            load_prev: false,
            has_save_file: crate::persistence::save::load().is_ok(),
            minimap_biome_cache: [0; 400],
            minimap_last_cell: (i32::MIN, i32::MIN),
            minimap_prims: Vec::with_capacity(400),
            minimap_screen_w: 0.0,
            hud_buf: String::with_capacity(64),
            editor_mode: false,
            editor_prev: false,
            editor_grid: BuildingGrid::new(),
            editor_physics: PhysicsWorld::new(Vec3::ZERO),
            editor_camera: GhostCamera::default(),
            editor_color_idx: 10, // default building tan
            editor_save_prev: false,
            editor_load_prev: false,
            editor_blueprint_idx: 0,
            editor_status: None,
            editor_ground_inited: false,
            editor_throw_prev: false,
            editor_bake_prev: false,
            editor_selected_group: None,
            editor_prev_group_prev: false,
            editor_next_group_prev: false,
            editor_unbake_prev: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            editor_undo_prev: false,
            editor_redo_prev: false,
            editor_clipboard: None,
            editor_copy_prev: false,
            editor_paste_prev: false,
            editor_mirror_x_prev: false,
            editor_mirror_z_prev: false,
            editor_replace_color_prev: false,
            editor_last_fill: None,
            editor_extrude_height: 0,
            editor_arrow_up_prev: false,
            editor_arrow_down_prev: false,
            editor_arrow_left_prev: false,
            editor_arrow_right_prev: false,
            drag_start: None,
            drag_plane: None,
            drag_end: None,
            drag_hold_timer: 0.0,
            drag_active: false,
            torches: Vec::new(),
            torch_prev: false,
            frame_point_lights: Vec::new(),
            frame_torch_dists: Vec::new(),
            fire_particle_timer: 0.0,
            cached_player_biome: Biome::Plains,
            panel_preload: None,
            panel_preload_rx: None,
            preloading_panel: None,
        };
        engine.spawn_npcs();
        engine.spawn_world_structures();
        engine.stamp_world_blueprints();

        {
            let player = engine.world.player_mut();
            if let Some(ref mut eq) = player.equipment {
                let _ = eq.equip(ITEM_IRON_SWORD);
            }
        }

        Ok(engine)
    }


    fn spawn_npcs(&mut self) {
        use crate::game::entity::Entity;
        use crate::renderer::MESH_CAPSULE;
        use crate::physics::body::{PhysicsBody, WeightClass};
        for def in npc::npc_defs() {
            let y = self.terrain.height_at_world(def.world_x, def.world_z) + 1.0;
            let pos = Vec3::new(def.world_x, y, def.world_z);
            let id = self.world.alloc_id();
            let half = Vec3::new(0.4, 0.6, 0.4);
            let (rb, col) = self.physics.add_static_box(pos, half, crate::physics::world::cg_static());
            let body = PhysicsBody { rigid_body: rb, collider: col, weight_class: WeightClass::Heavy };
            let entity = Entity::npc(id, body, MESH_CAPSULE, def.scale, 1.5, def.kind as u8);
            self.world.add_entity(entity);
        }
    }

    fn spawn_world_structures(&mut self) {
        use crate::game::entity::Entity;
        use crate::renderer::{MESH_CUBE, MESH_ROCK};
        use crate::physics::body::{PhysicsBody, WeightClass};

        // Helper: spawn a prop at world (x, z) with automatic terrain Y.
        let spawn = |physics: &mut PhysicsWorld, world: &mut World, terrain: &TerrainGrid,
                         x: f32, z: f32, y_offset: f32, mesh: u32, scale: Vec3| {
            let y = terrain.height_at_world(x, z) + y_offset + scale.y * 0.5;
            let id = world.alloc_id();
            let half = scale * 0.5;
            let (rb, col) = physics.add_static_box(Vec3::new(x, y, z), half, crate::physics::world::cg_static());
            let body = PhysicsBody { rigid_body: rb, collider: col, weight_class: WeightClass::Heavy };
            let entity = Entity::prop(id, body, mesh, scale, scale.length());
            world.add_entity(entity);
        };

        // Waystone Circle (Plains, near spawn).
        for i in 0..5 {
            let angle = i as f32 * std::f32::consts::TAU / 5.0;
            let cx = 45.0 + angle.cos() * 6.0;
            let cz = 80.0 + angle.sin() * 6.0;
            spawn(&mut self.physics, &mut self.world, &self.terrain,
                cx, cz, 0.0, MESH_ROCK, Vec3::new(0.8, 1.4, 0.8));
        }
        // Altar slab.
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            45.0, 80.0, 0.0, MESH_CUBE, Vec3::new(1.5, 0.25, 1.5));

        // Abandoned Campsite (Plains).
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            -30.0, 120.0, 0.0, MESH_CUBE, Vec3::new(0.6, 0.4, 0.6));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            -28.5, 120.5, 0.0, MESH_CUBE, Vec3::new(0.6, 0.4, 0.6));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            -29.0, 121.0, 0.0, MESH_CUBE, Vec3::new(2.0, 0.1, 2.0));

        // Stone Shrine (Forest, near Aldric).
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            -220.0, -355.0, 0.0, MESH_CUBE, Vec3::new(0.6, 3.0, 0.6));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            -220.0, -355.0, 3.0, MESH_CUBE, Vec3::new(1.2, 0.3, 1.2));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            -218.0, -355.0, 0.0, MESH_ROCK, Vec3::new(1.0, 1.2, 1.0));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            -222.0, -355.0, 0.0, MESH_ROCK, Vec3::new(1.0, 1.2, 1.0));

        // Forge Camp (Mountains, near Smith).
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            375.0, -825.0, 0.0, MESH_CUBE, Vec3::new(2.0, 0.5, 2.0));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            374.0, -825.0, 0.5, MESH_CUBE, Vec3::new(0.5, 1.8, 2.0));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            376.0, -825.0, 0.5, MESH_CUBE, Vec3::new(0.5, 1.8, 2.0));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            375.5, -824.0, 0.0, MESH_ROCK, Vec3::new(0.7, 0.9, 0.7));

        // Giant's Cairn (Mountains).
        for (i, s) in [(2.0, 1.2), (1.6, 1.0), (1.2, 0.8), (0.8, 0.6)].iter().enumerate() {
            let stack_y = if i == 0 { 0.0 } else {
                (0..i).map(|j| [(2.0, 1.2), (1.6, 1.0), (1.2, 0.8), (0.8, 0.6)][j].1).sum::<f32>()
            };
            spawn(&mut self.physics, &mut self.world, &self.terrain,
                500.0, -950.0, stack_y, MESH_ROCK, Vec3::new(s.0, s.1, s.0));
        }

        // Sunken Vault Entrance (Dungeon, near Oracle).
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            645.0, 903.0, 0.0, MESH_CUBE, Vec3::new(0.8, 2.5, 0.8));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            648.0, 903.0, 0.0, MESH_CUBE, Vec3::new(0.8, 2.5, 0.8));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            646.5, 903.0, 2.5, MESH_CUBE, Vec3::new(3.0, 0.6, 0.8));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            646.5, 903.5, 0.0, MESH_ROCK, Vec3::new(1.0, 0.8, 1.0));
        spawn(&mut self.physics, &mut self.world, &self.terrain,
            647.5, 902.0, 0.0, MESH_ROCK, Vec3::new(0.8, 0.6, 0.8));

    }

    /// Stamp saved blueprints into the world as pre-built structures.
    fn stamp_world_blueprints(&mut self) {
        let bp_path = blueprint::blueprints_dir().join("structure_1774363199.json");
        let bp = match blueprint::load_blueprint(&bp_path) {
            Ok(bp) => bp,
            Err(e) => {
                log::warn!("Failed to load world blueprint: {}", e);
                return;
            }
        };

        // Placement positions (world XZ) near spawn — player starts at ~(0, h, 4).
        let placements: &[(f32, f32)] = &[
            (25.0, 35.0),
            (-20.0, 50.0),
        ];

        for &(wx, wz) in placements {
            // Find terrain height at the structure's center to ground it.
            let terrain_y = self.terrain.height_at_world(wx, wz);
            // World grid offset: blueprint block (0,0,0) maps to (ox, oy, oz).
            let ox = wx.floor() as i32;
            let oy = terrain_y.ceil() as i32; // sit on terrain surface
            let oz = wz.floor() as i32;

            // Unbaked blocks — individual cells.
            for b in &bp.blocks {
                let cx = ox + b.x;
                let cy = oy + b.y;
                let cz = oz + b.z;
                let bt = BlockType::from_u8(b.block_type);
                let color = Vec3::new(b.color[0], b.color[1], b.color[2]);
                self.building.load_cell(
                    &mut self.physics,
                    cx, cy, cz,
                    bt, b.rotation, b.sub_blocks, color,
                );
            }

            // Baked groups — single merged objects.
            for group_blocks in &bp.groups {
                self.building.load_group_offset(
                    &mut self.physics,
                    group_blocks,
                    ox, oy, oz,
                );
            }
        }

        self.building.mark_dirty();
        log::info!("Stamped {} blueprint(s) into world", placements.len());
    }

    /// Try to turn in complete quests and accept available quests from an NPC.
    fn try_quest_turnin_accept(&mut self, npc_kind: u8) {
        // Turn in complete quests first.
        for q in self.quests.iter_mut() {
            if q.state == QuestState::Complete && q.giver_npc == npc_kind {
                // Award rewards.
                if let Some(ref mut stats) = self.world.player_mut().stats {
                    let levels = progression::award_xp(stats, q.reward_xp);
                    stats.gold += q.reward_gold;
                    println!("Quest '{}' complete! +{} XP, +{} gold", q.name, q.reward_xp, q.reward_gold);
                    if levels > 0 {
                        println!("Level up! Now level {}", stats.level);
                    }
                }
                for &(item_id, count) in q.reward_items {
                    if let Some(ref mut inv) = self.world.player_mut().inventory {
                        inv.add(item_id, count);
                    }
                }
                q.state = QuestState::TurnedIn;
            }
        }
        // Unlock any quests whose prerequisites are now met.
        quest::unlock_quests(&mut self.quests);
        // Accept available quests from this NPC.
        for q in self.quests.iter_mut() {
            if q.state == QuestState::Available && q.giver_npc == npc_kind {
                q.state = QuestState::Active;
                println!("Quest accepted: '{}'", q.name);
            }
        }
    }

    fn update_menu(&mut self, input: &InputState) {
        // Navigate with W/S (reuse forward/backward).
        if input.forward && !self.save_prev {
            // Reusing save_prev as "up_prev" for edge detection in menu.
            if self.menu_selection > 0 { self.menu_selection -= 1; }
        }
        // Reusing load_prev as "down_prev".
        if input.backward && !self.load_prev {
            let max = if self.has_save_file { 2 } else { 1 }; // New Game, [Continue], Quit
            if self.menu_selection < max { self.menu_selection += 1; }
        }
        self.save_prev = input.forward;
        self.load_prev = input.backward;

        // Confirm with E or Space.
        if input.interact || input.jump {
            match self.menu_selection {
                0 => {
                    // New Game.
                    self.game_state = GameState::Playing;
                }
                1 if self.has_save_file => {
                    // Continue — load save.
                    self.do_load();
                    self.game_state = GameState::Playing;
                }
                s if s == (if self.has_save_file { 2 } else { 1 }) => {
                    // Quit — signal via sentinel.
                    self.menu_selection = 255;
                }
                _ => {}
            }
        }
    }

    /// Update the held block's mesh and rotation to match selected_block_type/selected_rotation.
    fn update_held_block_visual(&mut self) {
        let rb = match self.interaction.held_body {
            Some(rb) => rb,
            None => return,
        };
        if let Some(entity) = self.world.entity_by_rb_mut(rb) {
            entity.mesh_type = self.selected_block_type.mesh_id();
        }
        let rot = if self.selected_block_type == BlockType::Slab && self.selected_rotation == 1 {
            // Flip upside-down for top slab preview
            glam::Quat::from_rotation_x(std::f32::consts::PI)
        } else {
            let angle = self.selected_rotation as f32 * std::f32::consts::FRAC_PI_2;
            glam::Quat::from_rotation_y(-angle)
        };
        self.physics.set_body_rotation(rb, rot);
    }

    /// Try to spawn one enemy at a random position near the player (biome-weighted).
    fn try_spawn_enemy(&mut self, player_pos: Vec3) {
        // Count current enemies.
        let enemy_count = self.world.entities.iter()
            .filter(|e| e.kind == EntityKind::Enemy)
            .count();
        if enemy_count >= enemy_ai::MAX_ENEMIES {
            return;
        }

        // Pick a random angle and distance from the player.
        let angle = enemy_ai::cheap_rand_pub(&mut self.spawn_seed) * std::f32::consts::TAU;
        let min = enemy_ai::MIN_SPAWN_DIST;
        let max = enemy_ai::MAX_SPAWN_DIST;
        let dist = min + enemy_ai::cheap_rand_pub(&mut self.spawn_seed) * (max - min);
        let x = player_pos.x + angle.cos() * dist;
        let z = player_pos.z + angle.sin() * dist;

        // Check terrain height — skip if underwater.
        let terrain_y = self.terrain.height_at_world(x, z);
        if terrain_y < WATER_LEVEL + 0.5 {
            return;
        }

        let biome = self.terrain.biome_at_world(x, z);
        let enemy_type = enemy_ai::pick_enemy_for_biome(biome, &mut self.spawn_seed);
        let params = enemy_type.params();
        let spawn_pos = Vec3::new(x, terrain_y + 1.0, z);

        let id = self.world.alloc_id();
        let body = PhysicsBody::new_enemy_ball(
            &mut self.physics,
            spawn_pos,
            params.physics_radius,
            params.weight_class,
        );
        let stats = StatBlock::new_enemy(
            params.level, params.str_, params.int,
            params.dex, params.vit, params.end,
        );
        let entity = Entity::enemy(
            id, body, params.mesh, params.render_scale,
            params.bounding_radius, stats,
        );
        self.world.add_entity(entity);
        self.enemy_ais.insert(id, EnemyAi::new(enemy_type, spawn_pos, self.spawn_seed));
    }


    /// Returns true if the game should quit.
    pub fn should_quit(&self) -> bool {
        self.game_state == GameState::MainMenu && self.menu_selection == 255
    }

    pub fn update(&mut self, dt: f32, input: &InputState) {
        // Track frame times for FPS display.
        self.frame_times[self.frame_time_idx] = dt;
        self.frame_time_idx = (self.frame_time_idx + 1) % self.frame_times.len();

        // --- Main menu ---
        if self.game_state == GameState::MainMenu {
            self.update_menu(input);
            return;
        }

        // --- Structure editor toggle (F7) ---
        if input.toggle_editor && !self.editor_prev {
            self.editor_mode = !self.editor_mode;
            if self.editor_mode {
                // Enter editor: initialize camera.
                self.editor_camera.active = true;
                self.editor_camera.eye = Vec3::new(5.0, 8.0, 15.0);
                self.editor_camera.yaw = 0.0;
                self.editor_camera.pitch = -0.4;
            } else {
                // Exit editor: restore main building mesh, clear drag state.
                self.editor_camera.active = false;
                self.building.mark_dirty();
                self.editor_grid.clear_preview();
                self.drag_start = None;
                self.drag_plane = None;
                self.drag_end = None;
                self.drag_active = false;
                self.drag_hold_timer = 0.0;
            }
        }
        self.editor_prev = input.toggle_editor;

        if self.editor_mode {
            self.update_editor(dt, input);
            return;
        }

        self.water_time += dt;

        // Check if the player crossed an ocean panel boundary.
        self.check_panel_boundary();

        // Day/night cycle: ~4 minutes per full day, F4 speeds up 10x.
        if input.fast_time && !self.fast_time_prev {
            self.fast_time = !self.fast_time;
        }
        self.fast_time_prev = input.fast_time;
        const DAY_DURATION: f32 = 240.0;
        let time_speed = if self.fast_time { 10.0 } else { 1.0 };
        self.time_of_day = (self.time_of_day + dt * time_speed / DAY_DURATION) % 1.0;
        self.terrain_rebuild_timer += dt;

        // --- Fast mode toggle (F2) ---
        if input.toggle_fast && !self.fast_prev {
            self.fast_mode = !self.fast_mode;
        }
        self.fast_prev = input.toggle_fast;

        // --- God mode toggle (F6) ---
        if input.toggle_god && !self.god_prev {
            self.god_mode = !self.god_mode;
        }
        self.god_prev = input.toggle_god;

        // --- Weather debug toggle (F7) ---
        if input.toggle_weather && !self.weather_prev {
            if self.weather_debug_active {
                // Deactivate: return to clear
                self.weather.force_clear();
                self.weather_debug_active = false;
            } else {
                // Activate: pick a random weather
                self.weather.force_random();
                self.weather_debug_active = true;
            }
            log::info!("Weather debug: {:?} (active={})", self.weather.kind, self.weather_debug_active);
        }
        self.weather_prev = input.toggle_weather;

        // --- Performance mode toggle (F11) ---
        if input.toggle_perf && !self.perf_prev {
            self.perf_mode = !self.perf_mode;
            self.renderer.set_render_scale(if self.perf_mode { 0.5 } else { 1.0 });
            log::info!("Performance mode: {}", self.perf_mode);
        }
        self.perf_prev = input.toggle_perf;

        // --- Debug UI toggle (F3) ---
        if input.toggle_debug_ui && !self.debug_ui_prev {
            self.show_debug_ui = !self.show_debug_ui;
        }
        self.debug_ui_prev = input.toggle_debug_ui;

        // --- Inventory toggle (I) ---
        if input.toggle_inventory && !self.inventory_prev {
            self.show_inventory = !self.show_inventory;
        }
        self.inventory_prev = input.toggle_inventory;

        // --- Teleport menu toggle (P) ---
        if input.toggle_teleport && !self.teleport_prev {
            self.show_teleport = !self.show_teleport;
        }
        self.teleport_prev = input.toggle_teleport;

        // --- Teleport selection (digit keys while menu open) ---
        if self.show_teleport {
            if let Some(slot) = input.editor_color_slot {
                let panels = world_panels();
                // slot 0 = Digit1 = first island, slot 1 = Digit2 = second, etc.
                if let Some(panel) = panels.get(slot as usize) {
                    if panel.grid_x != self.panel_x || panel.grid_z != self.panel_z {
                        self.load_panel(panel.grid_x, panel.grid_z, 0.0, 0.0);
                    }
                    self.show_teleport = false;
                }
            }
        }

        // --- Quick save (F5) ---
        if input.quick_save && !self.save_prev {
            self.do_save();
        }
        self.save_prev = input.quick_save;

        // --- Quick load (F9) ---
        if input.quick_load && !self.load_prev {
            self.do_load();
        }
        self.load_prev = input.quick_load;

        // --- Mute toggle (M) ---
        if input.toggle_mute && !self.mute_prev {
            if let Some(audio) = &mut self.audio {
                audio.toggle_mute();
            }
        }
        self.mute_prev = input.toggle_mute;

        // --- Debug stats (F1) ---
        let debug_edge = input.debug_stats && !self.debug_stats_prev;
        self.debug_stats_prev = input.debug_stats;
        if debug_edge {
            self.world.log_player_stats();
        }

        // --- Ghost mode toggle ---
        let just_entered_ghost = self.ghost.handle_toggle(input);

        if just_entered_ghost {
            let aspect = self.surface_width as f32 / self.surface_height.max(1) as f32;
            let (view, proj) = self.camera.camera_matrices(aspect);
            self.ghost.activate_from_fp_camera(&self.camera, view, proj);
            self.interaction.drop_held(&mut self.physics);
        }

        // --- Tool cycling (Tab) ---
        self.interaction.cycle_tool(input.cycle_tool);

        // --- B: cycle block type ---
        let cycle_block = input.cycle_block_type && !self.cycle_block_prev;
        self.cycle_block_prev = input.cycle_block_type;
        if cycle_block {
            self.selected_block_type = self.selected_block_type.next();
            self.selected_rotation = 0;
            self.update_held_block_visual();
        }

        // --- V: rotate block (slab toggles top/bottom) ---
        let rotate = input.rotate_block && !self.rotate_prev;
        self.rotate_prev = input.rotate_block;
        if rotate {
            if self.selected_block_type == BlockType::Slab {
                self.selected_rotation = if self.selected_rotation == 0 { 1 } else { 0 };
            } else {
                self.selected_rotation = (self.selected_rotation + 1) % 4;
            }
            self.update_held_block_visual();
        }

        if self.ghost.active {
            self.ghost.update(dt, input);
            self.physics.set_body_linvel(self.player_rb, Vec3::ZERO);
            self.apply_buoyancy(dt);
            self.physics.step(dt);

        } else {
            // --- First-person camera ---
            self.camera.look(input);
            let player_pos = self.physics.body_position(self.player_rb);
            self.camera.update(player_pos);

            let cam_eye = self.camera.eye;
            let look_dir = self.camera.look_dir();
            let player_eye = cam_eye;

            // Determine if crosshair is aimed at a building cell (for pry logic).
            // Only raycast when E is pressed — the result is unused otherwise.
            let building_cell_aimed_at = if input.interact {
                self.physics
                    .cast_ray_detailed(cam_eye, look_dir, PLACE_RANGE, self.player_col)
                    .and_then(|(hit_pos, normal)| {
                        let target = hit_pos - normal * 0.01;
                        let coords = building::snap_to_grid(target);
                        if self.building.is_occupied(coords.0, coords.1, coords.2) {
                            Some(coords)
                        } else {
                            None
                        }
                    })
            } else {
                None
            };

            let interaction_result = self.interaction.update(
                &mut self.physics,
                &self.world,
                player_eye,
                look_dir,
                input.interact,
                input.throw,
                self.player_col,
                dt,
                building_cell_aimed_at,
                &self.terrain_rbs,
                &self.tree_rbs,
            );

            // --- Handle item pickup ---
            if let Some(drop_id) = interaction_result.picked_up_item {
                // Get position for particles before removing (O(1) via index map).
                let pickup_pos = self.world.entity(drop_id)
                    .map(|e| self.physics.body_position(e.body.rigid_body));
                if self.world.pickup_item(drop_id) {
                    if let Some(entity) = self.world.remove_by_id(drop_id) {
                        self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                    }
                    if let Some(pos) = pickup_pos {
                        self.particles.emit_pickup(pos);
                    }
                }
            }

            // --- Handle pried building cube ---
            if let Some((cx, cy, cz)) = interaction_result.pried_cell {
                self.building.remove(&mut self.physics, cx, cy, cz);

                let center = building::cell_center(cx, cy, cz);
                let obj_id = self.world.alloc_id();

                let body = PhysicsBody::new_dynamic_box(
                    &mut self.physics,
                    center,
                    Vec3::splat(0.5),
                    WeightClass::Medium,
                );
                self.physics.set_gravity_enabled(body.rigid_body, false);
                self.interaction.held_body = Some(body.rigid_body);

                self.world.add_entity(Entity::prop(
                    obj_id,
                    body,
                    MESH_CUBE,
                    Vec3::ONE,
                    UNIT_BOUNDING_RADIUS,
                ));
            }

            // --- Handle axe split ---
            if let Some(target_body) = interaction_result.axe_hit {
                self.split_cube(target_body, player_eye, look_dir);
            }

            // --- Handle pickaxe hit (terrain only) ---
            if let Some(pickaxe_hit) = interaction_result.pickaxe_hit {
                match pickaxe_hit {
                    PickaxeHit::Terrain(hit_pos) => {
                        self.terrain.deform_ground(hit_pos, DEFORM_RADIUS, DEFORM_AMOUNT);
                    }
                }
            }

            // --- Handle hammer hit (structures only) ---
            if let Some(hammer_hit) = interaction_result.hammer_hit {
                match hammer_hit {
                    HammerHit::Body(rb) => {
                        self.damage_mining_chunk(rb);
                    }
                    HammerHit::Static(rb, hit_pos) => {
                        if let Some(gi) = self.building.group_for_body(rb) {
                            self.building.unbake_group(&mut self.physics, gi);
                            self.building.mine_at(&mut self.physics, hit_pos);
                        } else if self.building.has_body(rb) {
                            self.building.mine_at(&mut self.physics, hit_pos);
                        } else if self.mining.is_mining_chunk(rb) {
                            self.damage_mining_chunk(rb);
                        }
                    }
                }
            }

            // --- Handle chisel hit (single sub-block removal) ---
            if let Some(chisel_hit) = interaction_result.chisel_hit {
                match chisel_hit {
                    ChiselHit::Static(rb, hit_pos) => {
                        if let Some(gi) = self.building.group_for_body(rb) {
                            self.building.unbake_group(&mut self.physics, gi);
                            self.building.chisel_at(&mut self.physics, hit_pos);
                        } else if self.building.has_body(rb) {
                            self.building.chisel_at(&mut self.physics, hit_pos);
                        }
                    }
                }
            }

            // Tree punch is handled below, synced to combat active phase.

            // --- RMB: place held cube into building grid ---
            let place_pressed = input.place && !self.place_prev;
            self.place_prev = input.place;

            if place_pressed {
                if let Some(held_handle) = self.interaction.held_body.take() {
                    let pos = self.physics.body_position(held_handle);
                    let (cx, cy, cz) = building::snap_to_grid(pos);

                    let terrain_h = self.terrain.height_at_world(
                        cx as f32 + 0.5,
                        cz as f32 + 0.5,
                    );
                    if self.building.place(&mut self.physics, cx, cy, cz, terrain_h,
                                           self.selected_block_type, self.selected_rotation) {
                        if let Some(entity) = self.world.remove_by_rb(held_handle) {
                            self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                        }
                    } else {
                        self.interaction.held_body = Some(held_handle);
                    }
                }
            }

            // --- F: spawn a new block ---
            let spawn_pressed = input.spawn && !self.spawn_prev;
            self.spawn_prev = input.spawn;

            if spawn_pressed && self.interaction.held_body.is_none() {
                let spawn_pos = player_pos + Vec3::new(0.0, 1.5, 0.0) + look_dir * 3.0;
                let obj_id = self.world.alloc_id();

                let mesh_id = self.selected_block_type.mesh_id();
                let body = PhysicsBody::new_dynamic_box(
                    &mut self.physics,
                    spawn_pos,
                    Vec3::splat(0.5),
                    WeightClass::Medium,
                );
                self.physics.set_gravity_enabled(body.rigid_body, false);
                // Apply initial rotation so the preview matches selected_rotation.
                if self.selected_rotation != 0 {
                    let angle = self.selected_rotation as f32 * std::f32::consts::FRAC_PI_2;
                    self.physics.set_body_rotation(body.rigid_body, glam::Quat::from_rotation_y(-angle));
                }
                self.interaction.held_body = Some(body.rigid_body);

                self.world.add_entity(Entity::prop(
                    obj_id,
                    body,
                    mesh_id,
                    Vec3::ONE,
                    UNIT_BOUNDING_RADIUS,
                ));
            }

            // --- T: place a torch (grid-snapped) ---
            if input.place_torch && !self.torch_prev {
                if let Some((_rb, hit_point, hit_normal)) = self.physics.cast_ray_full(
                    player_eye,
                    look_dir,
                    PLACE_RANGE,
                    self.player_col,
                ) {
                    // Snap to grid: place on the surface cell adjacent to the hit.
                    let target = hit_point + hit_normal * 0.5;
                    let (gx, gy, gz) = building::snap_to_grid(target);
                    let torch_pos = Vec3::new(gx as f32 + 0.5, gy as f32, gz as f32 + 0.5);
                    // Don't place duplicate at same grid cell.
                    let already = self.torches.iter().any(|t| {
                        let (tx, ty, tz) = building::snap_to_grid(t.position);
                        tx == gx && ty == gy && tz == gz
                    });
                    if !already {
                        let flame_pos = torch_pos + Vec3::new(0.0, TORCH_FLAME_HEIGHT, 0.0);
                        self.torches.push(TorchInstance {
                            position: torch_pos,
                            flame_pos,
                        });
                    }
                }
            }
            self.torch_prev = input.place_torch;

            // --- Compute player derived stats once per frame ---
            {
                let p = self.world.player_mut();
                let bonuses = p.equipment.as_mut()
                    .map(|eq| eq.total_bonuses())
                    .unwrap_or_default();
                self.player_derived = p.stats.as_ref()
                    .map(|s| s.compute_derived(&bonuses))
                    .unwrap_or_else(|| StatBlock::new_player().compute_derived(&StatBonuses::default()));
            }
            // Sync cached_derived on the player entity so game_tick() can use it.
            self.world.player_mut().cached_derived = Some(self.player_derived.clone());
            let player_melee_mult = self.player_derived.melee_damage_mult;

            // --- Combat: melee attack ---
            if input.throw
                && self.interaction.held_body.is_none()
                && self.interaction.equipped_tool == ToolType::Hands
            {
                let weapon_data = self.world.player().equipment.as_ref()
                    .and_then(|eq| eq.weapon_data());
                self.combat.try_attack(weapon_data);
            }
            if let Some(hit) = self.combat.update(
                dt,
                &self.physics,
                &self.world,
                player_eye,
                look_dir,
                self.player_col,
                player_melee_mult,
            ) {
                // Apply damage to hit enemy.
                if let Some(entity) = self.world.entity_mut(hit.entity_id) {
                    if let Some(ref mut stats) = entity.stats {
                        let dmg = if self.god_mode { 99999.0 } else { hit.damage };
                        stats.take_damage(dmg);
                    }
                    let hit_pos = self.physics.body_position(entity.body.rigid_body);
                    self.particles.emit_hit(hit_pos);
                    self.physics.apply_impulse(
                        entity.body.rigid_body,
                        hit.knockback_dir * hit.knockback_force * self.physics.body_mass(entity.body.rigid_body),
                    );
                }
            }

            // --- Bare-fist punch: tree shake + prop knockback (synced to combat active phase) ---
            if self.combat.entered_active_phase()
                && self.interaction.equipped_tool == ToolType::Hands
            {
                if let Some(hit_pos) = self.interaction.punch_env(
                    &mut self.physics,
                    &self.world,
                    player_eye,
                    look_dir,
                    self.player_col,
                    &self.tree_rbs,
                ) {
                    self.tree_punch_seed = self.tree_punch_seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    self.structures.punch_tree_at(hit_pos, self.tree_punch_seed);
                }
            }

            // --- Spell cycling (Q) ---
            self.spells.cycle_spell(input.cycle_spell);

            // --- Spell casting (R) ---
            let cast_edge = input.cast_spell && !self.cast_prev;
            self.cast_prev = input.cast_spell;
            if cast_edge {
                // Read player mana; derived stats already cached for this frame.
                let current_mana = self.world.player().stats.as_ref().map_or(0.0, |s| s.mana);
                let derived = &self.player_derived;

                if let Some((result, mana_cost)) = self.spells.try_cast(
                    current_mana,
                    derived,
                    player_eye,
                    look_dir,
                    &self.physics,
                    &self.world,
                    self.player_col,
                ) {
                    // Deduct mana.
                    if let Some(ref mut stats) = self.world.player_mut().stats {
                        stats.mana -= mana_cost;
                    }

                    match result {
                        CastResult::Hit(spell_hit) => {
                            // Ice Shard direct hit.
                            if let Some(entity) = self.world.entity_mut(spell_hit.entity_id) {
                                if let Some(ref mut stats) = entity.stats {
                                    let dmg = if self.god_mode { 99999.0 } else { spell_hit.damage };
                                    stats.take_damage(dmg);
                                }
                                let hit_pos = self.physics.body_position(entity.body.rigid_body);
                                self.particles.emit_ice(hit_pos);
                                self.physics.apply_impulse(
                                    entity.body.rigid_body,
                                    spell_hit.knockback_dir * 4.0 * self.physics.body_mass(entity.body.rigid_body),
                                );
                            }
                        }
                        CastResult::Heal(amount) => {
                            if let Some(ref mut stats) = self.world.player_mut().stats {
                                stats.health = (stats.health + amount).min(derived.max_health);
                            }
                            self.particles.emit_heal(player_pos);
                        }
                        CastResult::Projectile | CastResult::Miss => {}
                    }
                }
            }

            // --- Update spell projectiles ---
            self.spells.update(
                dt,
                &self.physics,
                &self.world,
                self.player_col,
                &mut self.spell_hit_buf,
            );
            for spell_hit in self.spell_hit_buf.iter() {
                if let Some(entity) = self.world.entity_mut(spell_hit.entity_id) {
                    if let Some(ref mut stats) = entity.stats {
                        let dmg = if self.god_mode { 99999.0 } else { spell_hit.damage };
                        stats.take_damage(dmg);
                    }
                    let hit_pos = self.physics.body_position(entity.body.rigid_body);
                    self.particles.emit_fireball(hit_pos);
                    self.physics.apply_impulse(
                        entity.body.rigid_body,
                        spell_hit.knockback_dir * 4.0 * self.physics.body_mass(entity.body.rigid_body),
                    );
                }
            }

            // --- Remove dead enemies + award XP + loot; daytime despawn (single pass) ---
            let is_daytime = !enemy_ai::is_night(self.time_of_day);
            self.dead_ids.clear();
            self.despawn_ids.clear();
            let despawn_dist_sq = (enemy_ai::MAX_SPAWN_DIST * 1.5) * (enemy_ai::MAX_SPAWN_DIST * 1.5);
            for e in self.world.entities.iter() {
                if e.kind != EntityKind::Enemy { continue; }
                if e.stats.as_ref().map_or(false, |s| s.is_dead()) {
                    let level = e.stats.as_ref().map_or(1, |s| s.level);
                    self.dead_ids.push((e.id, level));
                } else if is_daytime {
                    let epos = self.physics.body_position(e.body.rigid_body);
                    if (epos - player_pos).length_squared() > despawn_dist_sq {
                        self.despawn_ids.push(e.id);
                    }
                }
            }
            for i in 0..self.dead_ids.len() {
                let (dead_id, enemy_level) = self.dead_ids[i];
                // Grab enemy type before removing.
                let enemy_type = self.enemy_ais.get(&dead_id).map(|ai| ai.enemy_type);

                // Get enemy position for death particles before removing (O(1) via index map).
                let death_pos = self.world.entity(dead_id)
                    .map(|e| self.physics.body_position(e.body.rigid_body));

                if let Some(entity) = self.world.remove_by_id(dead_id) {
                    self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                    self.enemy_ais.remove(&dead_id);

                    // Death particles.
                    if let Some(pos) = death_pos {
                        self.particles.emit_death(pos);
                    }

                    // Award XP to player.
                    let xp = progression::xp_for_kill(enemy_level);
                    if let Some(ref mut stats) = self.world.player_mut().stats {
                        let levels = progression::award_xp(stats, xp);
                        if levels > 0 {
                            self.particles.emit_level_up(player_pos);
                        }
                    }

                    // Notify quest system of kill.
                    if let Some(etype) = enemy_type {
                        quest::notify_kill(&mut self.quests, etype as u8);
                    }

                    // Roll and award loot.
                    if let Some(etype) = enemy_type {
                        let drops = enemy_ai::roll_loot(etype, &mut self.spawn_seed);
                        for (item_id, count) in drops {
                            if item_id == enemy_ai::LOOT_GOLD {
                                if let Some(ref mut stats) = self.world.player_mut().stats {
                                    stats.gold += count as u32;
                                }
                            } else if let Some(ref mut inv) = self.world.player_mut().inventory {
                                inv.add(item_id, count);
                            }
                        }
                    }
                }
            }
            for i in 0..self.despawn_ids.len() {
                let id = self.despawn_ids[i];
                if let Some(entity) = self.world.remove_by_id(id) {
                    self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                    self.enemy_ais.remove(&id);
                }
            }

            // --- Night enemy spawning ---
            if !is_daytime {
                self.spawn_timer -= dt;
                if self.spawn_timer <= 0.0 {
                    self.try_spawn_enemy(player_pos);
                    // Spawn interval: 2-4 seconds.
                    self.spawn_timer = 2.0 + enemy_ai::cheap_rand_pub(&mut self.spawn_seed) * 2.0;
                }
            }

            // --- Enemy AI ---
            let player_col_handle = self.world.player().body.collider;
            enemy_ai::update_all(
                &mut self.enemy_ais,
                &mut self.enemy_projectiles,
                &mut self.physics,
                &self.world.entities,
                player_pos,
                player_col_handle,
                dt,
                &mut self.enemy_hit_buf,
                &self.terrain,
            );
            // Tick enemy projectiles (arrows).
            enemy_ai::update_projectiles(
                &mut self.enemy_projectiles,
                &self.physics,
                self.player_rb,
                dt,
                &mut self.arrow_hit_buf,
            );
            // Apply enemy damage to player (melee + projectile).
            for hit in self.enemy_hit_buf.iter().chain(self.arrow_hit_buf.iter()) {
                if self.god_mode { continue; }
                if let Some(ref mut stats) = self.world.player_mut().stats {
                    stats.take_damage(hit.damage);
                }
                let player_rb = self.world.player().body.rigid_body;
                let mass = self.physics.body_mass(player_rb);
                self.physics.apply_impulse(player_rb, hit.knockback_dir * hit.knockback_force * mass);
            }

            // --- Cache ground state once per frame ---
            self.player_on_ground = self.physics.is_on_ground(self.player_col);

            // --- Camera-relative player movement ---
            self.apply_player_movement(input);

            // --- Water buoyancy ---
            self.apply_buoyancy(dt);

            self.physics.step(dt);

            // --- Update player model animation ---
            self.player_model.set_attack_progress(self.combat.animation_progress());
            let vel = self.physics.body_linvel_xz(self.player_rb);
            let horiz_speed = Vec3::new(vel.x, 0.0, vel.z).length();
            self.player_model.update(dt, horiz_speed);

            // --- Smooth player facing yaw ---
            let target_yaw = self.player_facing_yaw();
            let mut delta = target_yaw - self.player_visual_yaw;
            // Wrap to [-PI, PI] for shortest-path rotation.
            if delta > std::f32::consts::PI { delta -= std::f32::consts::TAU; }
            if delta < -std::f32::consts::PI { delta += std::f32::consts::TAU; }
            let turn_speed = 12.0; // radians per second
            let max_step = turn_speed * dt;
            self.player_visual_yaw += delta.clamp(-max_step, max_step);
        }

        // Player position after physics step — used by weather, quests, audio, minimap, etc.
        let player_pos = self.physics.body_position(self.player_rb);
        // Cache biome lookup once per frame (used by weather, audio, and render).
        let player_biome = self.terrain.biome_at_world(player_pos.x, player_pos.z);
        self.cached_player_biome = player_biome;

        // --- Update particles ---
        self.particles.update(dt);

        // Emit fire particles for nearby torches at a fixed rate (~20/sec/torch).
        self.fire_particle_timer += dt;
        let fire_interval = 1.0 / 20.0;
        while self.fire_particle_timer >= fire_interval {
            self.fire_particle_timer -= fire_interval;
            for torch in &self.torches {
                if torch.position.distance_squared(player_pos) < 3600.0 {
                    self.particles.emit_fire(torch.flame_pos, 1);
                }
            }
        }

        // --- Update weather system ---
        self.weather.update(dt, player_biome);

        // --- Update tree shake + leaf particles (wind-affected) ---
        let wind_dir = self.weather.wind_dir();
        let wind_strength = self.weather.wind_strength;
        self.structures.update_effects(dt, wind_strength, wind_dir);

        // --- Wind-blown leaves from nearby trees ---
        self.wind_leaf_timer += dt;
        if self.wind_leaf_timer >= 0.3 {
            self.wind_leaf_timer = 0.0;
            self.structures.emit_wind_leaves(
                player_pos, wind_strength, wind_dir, &mut self.tree_punch_seed,
            );
        }

        // --- Blizzard intensity + snow particles ---
        self.snow_time += dt;
        {
            let in_snow = self.terrain.is_snow_zone(player_pos.x, player_pos.z);

            // Smooth snow intensity transition.
            let target = if in_snow { 1.0 } else { 0.0 };
            let ramp_speed = if in_snow { 0.6 } else { 1.5 }; // fade in slower, fade out faster
            if self.snow_intensity < target {
                self.snow_intensity = (self.snow_intensity + ramp_speed * dt).min(target);
            } else {
                self.snow_intensity = (self.snow_intensity - ramp_speed * dt).max(target);
            }

        }

        // --- Batched terrain rebuild (mesh + physics) ---
        if self.terrain.has_dirty_chunks()
            && self.terrain_rebuild_timer >= TERRAIN_REBUILD_INTERVAL
        {
            self.terrain_rebuild_timer = 0.0;
            self.rebuild_dirty_terrain();
        }

        // --- NPC interaction (E key) ---
        {
            let interact_edge = input.interact && !self.npc_interact_prev;
            self.npc_interact_prev = input.interact;

            if interact_edge {
                if self.active_dialogue.is_some() {
                    // Advance or close dialogue.
                    let finished = self.active_dialogue.as_mut().unwrap().advance();
                    if finished {
                        // Try to turn in / accept quests from this NPC.
                        let npc_kind = self.active_dialogue.as_ref().unwrap().npc_kind as u8;
                        self.try_quest_turnin_accept(npc_kind);
                        self.active_dialogue = None;
                    }
                } else {
                    // Check proximity to NPCs.
                    let npc_defs = npc::npc_defs();
                    let mut closest: Option<(u32, u8, f32)> = None;
                    for e in self.world.entities.iter() {
                        if e.kind != EntityKind::Npc { continue; }
                        let npc_pos = self.physics.body_position(e.body.rigid_body);
                        let dist = (player_pos - npc_pos).length();
                        if dist < 4.0 {
                            if closest.is_none() || dist < closest.unwrap().2 {
                                closest = Some((e.id, e.npc_kind.unwrap_or(0), dist));
                            }
                        }
                    }
                    if let Some((eid, nk, _)) = closest {
                        if let Some(def) = npc_defs.iter().find(|d| d.kind as u8 == nk) {
                            self.active_dialogue = Some(ActiveDialogue {
                                npc_entity_id: eid,
                                npc_kind: def.kind,
                                npc_name: def.name,
                                lines: def.dialogue,
                                current_line: 0,
                            });
                        }
                    }
                }
            }
        }

        // --- Per-frame quest checks ---
        {
            if let Some(ref inv) = self.world.player().inventory {
                quest::check_collect_quests(&mut self.quests, inv);
            }
            quest::check_reach_quests(&mut self.quests, player_pos.x, player_pos.z);
        }

        // --- Game tick: regen, etc. ---
        self.world.game_tick(dt);

        // --- Audio: update music and footsteps based on player biome ---
        if let Some(audio) = &mut self.audio {
            audio.update(dt, player_biome, None);

            let vel = self.physics.body_linvel_xz(self.player_rb);
            let horizontal_speed = (vel.x * vel.x + vel.z * vel.z).sqrt();
            audio.update_footsteps(dt, player_biome, horizontal_speed, self.player_on_ground);
        }
    }

    fn apply_player_movement(&mut self, input: &InputState) {
        let forward = self.camera.forward_flat();
        let right = self.camera.right_flat();

        let mut move_vel = Vec3::ZERO;
        if input.forward  { move_vel += forward; }
        if input.backward { move_vel -= forward; }
        if input.right    { move_vel += right; }
        if input.left     { move_vel -= right; }
        let speed = if self.fast_mode { FAST_SPEED } else if input.sprint { SPRINT_SPEED } else { PLAYER_SPEED };
        if move_vel.length_squared() > 0.0 {
            move_vel = move_vel.normalize() * speed;
        }

        // Slide along walls: remove the velocity component pushing into each wall.
        for wall_n in self.physics.wall_normals(self.player_col) {
            let into_wall = move_vel.dot(wall_n);
            if into_wall < 0.0 {
                move_vel -= wall_n * into_wall;
            }
        }

        let mut vy = self.physics.body_linvel_y(self.player_rb);
        if input.jump && self.player_on_ground {
            vy = JUMP_VELOCITY;
        }
        self.physics.set_body_linvel(
            self.player_rb,
            Vec3::new(move_vel.x, vy, move_vel.z),
        );
    }

    /// Apply buoyancy and drag to a single rigid body using impulses (not forces,
    /// which accumulate across frames in Rapier 0.22).
    fn apply_body_buoyancy(&mut self, rb: rapier3d::prelude::RigidBodyHandle, size: f32, dt: f32) {
        let pos = self.physics.body_position(rb);
        let depth = WATER_LEVEL - pos.y;
        if depth > 0.0 {
            let mass = self.physics.body_mass(rb);
            let submerged = (depth / size).min(1.0);

            // Impulse = mass * acceleration * dt.
            let buoyancy = Vec3::new(0.0, mass * BUOYANCY_FORCE * submerged * dt, 0.0);
            self.physics.apply_impulse(rb, buoyancy);

            // Drag on all axes.
            let vel = self.physics.body_linvel_xz(rb);
            let drag = vel * (-mass * WATER_DRAG * submerged * dt);
            self.physics.apply_impulse(rb, drag);
        }
    }

    /// Apply buoyancy and drag to all dynamic bodies submerged below the water level.
    fn apply_buoyancy(&mut self, dt: f32) {
        self.apply_body_buoyancy(self.player_rb, 1.8, dt);

        self.buoyancy_bodies.clear();
        self.buoyancy_bodies.extend(
            self.world.entities.iter()
                .filter(|e| e.kind != EntityKind::Player && self.physics.is_dynamic(e.body.rigid_body))
                .map(|e| (e.body.rigid_body, e.render_scale.max_element())),
        );

        for i in 0..self.buoyancy_bodies.len() {
            let (rb, size) = self.buoyancy_bodies[i];
            self.apply_body_buoyancy(rb, size, dt);
        }
    }

    /// Damage a mining chunk and handle destruction + stability collapse.
    fn damage_mining_chunk(&mut self, rb: rapier3d::prelude::RigidBodyHandle) {
        if let Some(destroyed_id) = self.mining.damage_chunk(rb, 1) {
            if let Some(entity) = self.world.remove_by_id(destroyed_id) {
                self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
            }
            let impact_pos = self.physics.body_position(rb);
            let collapsed = self.mining.check_stability(&self.physics, impact_pos);
            for eid in collapsed {
                if let Some(entity) = self.world.remove_by_id(eid) {
                    self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                }
            }
        }
    }

    /// Split a cube object into two halves along the axis most aligned with the hit.
    fn split_cube(&mut self, target_body: rapier3d::prelude::RigidBodyHandle, _eye: Vec3, look_dir: Vec3) {
        let target_entity = match self.world.entity_by_rb(target_body) {
            Some(e) => e,
            None => return,
        };

        if target_entity.mesh_type != MESH_CUBE {
            let wc = target_entity.body.weight_class;
            let force = look_dir * 8.0 * wc.punch_knockback();
            self.physics.apply_impulse(target_body, force);
            return;
        }

        let entity = self.world.remove_by_rb(target_body).unwrap();
        let pos = self.physics.body_position(entity.body.rigid_body);
        let transform = self.physics.body_transform(entity.body.rigid_body);
        let scale = entity.render_scale;

        let inv_rot = transform.inverse();
        let local_dir = inv_rot.transform_vector3(look_dir);
        let abs = local_dir.abs();

        let split_axis = if abs.x >= abs.y && abs.x >= abs.z {
            0
        } else if abs.y >= abs.x && abs.y >= abs.z {
            1
        } else {
            2
        };

        self.physics.remove_body(entity.body.rigid_body, entity.body.collider);

        let mut half_scale = scale;
        match split_axis {
            0 => half_scale.x *= 0.5,
            1 => half_scale.y *= 0.5,
            _ => half_scale.z *= 0.5,
        }

        let mut offset = Vec3::ZERO;
        match split_axis {
            0 => offset.x = scale.x * 0.25,
            1 => offset.y = scale.y * 0.25,
            _ => offset.z = scale.z * 0.25,
        }

        let world_offset = transform.transform_vector3(offset);
        let half_extents = half_scale * 0.5;
        let wc = entity.body.weight_class;

        for sign in [-1.0_f32, 1.0_f32] {
            let half_pos = pos + world_offset * sign;
            let obj_id = self.world.alloc_id();

            let body = PhysicsBody::new_dynamic_box(
                &mut self.physics,
                half_pos,
                half_extents,
                wc,
            );

            let separation_impulse = world_offset.normalize_or_zero() * sign * 2.0;
            self.physics.apply_impulse(body.rigid_body, separation_impulse);

            self.world.add_entity(Entity::prop(
                obj_id,
                body,
                MESH_CUBE,
                half_scale,
                half_scale.max_element() * UNIT_BOUNDING_RADIUS,
            ));
        }
    }


    /// Compute the direction the player character should face.
    /// Uses movement direction if moving, otherwise faces camera forward.
    fn player_facing_yaw(&self) -> f32 {
        let vel = self.physics.body_linvel_xz(self.player_rb);
        let horiz = Vec3::new(vel.x, 0.0, vel.z);
        if horiz.length_squared() > 0.5 {
            horiz.x.atan2(horiz.z)
        } else {
            // Keep current facing when stopped.
            self.player_visual_yaw
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_width = width;
        self.surface_height = height;
        self.renderer.resize(width, height);
    }
}
