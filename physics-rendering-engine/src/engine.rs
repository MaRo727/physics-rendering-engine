use std::collections::HashMap;
use std::fmt::Write as _;

use anyhow::Result;
use glam::{Mat4, Vec3, Vec4};

use crate::audio::AudioManager;
use crate::blueprint;
use crate::building::{self, BlockType, BuildingGrid};
use crate::game::camera::FirstPersonCamera;
use crate::game::combat::CombatSystem;
use crate::game::enemy_ai::{self, EnemyAi, EnemyAttackHit, EnemyProjectile};
use crate::game::entity::{Entity, EntityKind};
use crate::game::items::ITEM_IRON_SWORD;
use crate::game::player_model::{PlayerModel, FP_PART_COUNT};
use crate::game::progression;
use crate::game::spells::{SpellSystem, SpellHit, CastResult};
use crate::game::stats::{DerivedStats, StatBlock, StatBonuses};
use crate::game::world::World;
use crate::input::InputState;
use crate::interaction::{Interaction, ToolType, PickaxeHit, HammerHit, ChiselHit};
use crate::mining::MiningSystem;
use crate::physics::body::{PhysicsBody, WeightClass};
use crate::physics::world::PhysicsWorld;
use crate::player::{GhostCamera, extract_frustum_planes, is_sphere_in_frustum};
use crate::renderer::{Renderer, GpuPointLight, pack_instance_id, SHADOW_ONLY_BIT, MESH_CUBE, MESH_WATER, MESH_TORCH, MESH_TERRAIN_BASE};
use crate::renderer::context::VulkanContext;
use crate::renderer::swapchain::Swapchain;
use crate::scene::{self, UNIT_BOUNDING_RADIUS};
use crate::structures::{StructureGrid, GrassGrid};
use crate::terrain::{Biome, TerrainGrid, TerrainChunkInfo, TERRAIN_HALF, CHUNKS_PER_SIDE};
use crate::particles::ParticleSystem;
use crate::ui::{Ui, UiPrimitive};
use crate::weather::Weather;
use crate::game::items::item_by_id;
use crate::game::npc::{self, ActiveDialogue};
use crate::game::quest::{self, Quest, QuestState};

const PLACE_RANGE: f32 = 8.0;

/// Editor color palette (12 colors).
const EDITOR_PALETTE: [Vec3; 12] = [
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
const BUILDING_OBJECT_ID: u32 = 0xFFF0;
const PLAYER_MODEL_OBJECT_ID: u32 = 0xFFE0;
const WATER_OBJECT_ID: u32 = 0xFFD0;
const PLAYER_SPEED: f32 = 5.0;
const SPRINT_SPEED: f32 = 9.0;
const FAST_SPEED: f32 = 40.0;
const JUMP_VELOCITY: f32 = 6.0;
const WATER_LEVEL: f32 = 5.0;
const BUOYANCY_FORCE: f32 = 25.0;
const WATER_DRAG: f32 = 12.0;

const TORCH_OBJECT_BASE: u32 = 0xFF70;
const TORCH_FLAME_HEIGHT: f32 = 0.85;

pub struct TorchInstance {
    pub position: Vec3,
    pub flame_pos: Vec3,
}

/// Minimum interval between terrain GPU/physics rebuilds (seconds).
const TERRAIN_REBUILD_INTERVAL: f32 = 0.1;
/// Radius of terrain deformation per pickaxe hit (world units).
const DEFORM_RADIUS: f32 = 3.0;
/// Height lowered at the center of a pickaxe hit.
const DEFORM_AMOUNT: f32 = 0.5;

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

/// Which plane the drag-to-fill rectangle lives on.
#[derive(Debug, Clone, Copy)]
enum DragPlane {
    /// Floor: fill XZ at fixed Y.
    FloorXZ(i32),
    /// Wall: fill XY at fixed Z.
    WallXY(i32),
    /// Wall: fill YZ at fixed X.
    WallYZ(i32),
}

/// Intersect a ray with the drag plane, returning the snapped grid coordinates
/// on the two free axes (the fixed axis comes from the plane).
fn ray_plane_intersect(origin: Vec3, dir: Vec3, plane: DragPlane) -> Option<(i32, i32, i32)> {
    let (plane_val, axis_component) = match plane {
        DragPlane::FloorXZ(y) => (y as f32 + 0.5, dir.y),
        DragPlane::WallXY(z) => (z as f32 + 0.5, dir.z),
        DragPlane::WallYZ(x) => (x as f32 + 0.5, dir.x),
    };
    // Ray nearly parallel to plane — skip.
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
fn build_fill_region(
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

pub struct Engine {
    pub config: EngineConfig,
    game_state: GameState,
    menu_selection: u8, // 0=New Game, 1=Continue, 2=Quit
    physics: PhysicsWorld,
    world: World,
    player_rb: rapier3d::prelude::RigidBodyHandle,
    player_col: rapier3d::prelude::ColliderHandle,
    player_on_ground: bool,
    camera: FirstPersonCamera,
    player_model: PlayerModel,
    renderer: Renderer,
    interaction: Interaction,
    ghost: GhostCamera,
    terrain: TerrainGrid,
    terrain_chunks: Vec<TerrainChunkInfo>,
    terrain_object_id: u32,
    terrain_rbs: std::collections::HashSet<rapier3d::prelude::RigidBodyHandle>,
    terrain_chunk_cols: Vec<rapier3d::prelude::ColliderHandle>,
    terrain_rebuild_timer: f32,
    mesh_building_id: u32,
    building: BuildingGrid,
    mining: MiningSystem,
    structures: StructureGrid,
    grass: GrassGrid,
    place_prev: bool,
    spawn_prev: bool,
    selected_block_type: BlockType,
    selected_rotation: u8,
    cycle_block_prev: bool,
    rotate_prev: bool,
    debug_stats_prev: bool,
    fast_prev: bool,
    fast_mode: bool,
    god_mode: bool,
    god_prev: bool,
    show_debug_ui: bool,
    debug_ui_prev: bool,
    /// Rolling frame time buffer for FPS display.
    frame_times: [f32; 60],
    frame_time_idx: usize,
    show_inventory: bool,
    inventory_prev: bool,
    mute_prev: bool,
    fast_time: bool,
    fast_time_prev: bool,
    water_time: f32,
    time_of_day: f32, // 0..1 where 0=midnight, 0.25=sunrise, 0.5=noon, 0.75=sunset
    surface_width: u32,
    surface_height: u32,
    audio: Option<AudioManager>,
    combat: CombatSystem,
    spells: SpellSystem,
    cast_prev: bool,
    enemy_ais: HashMap<u32, EnemyAi>,
    enemy_projectiles: Vec<EnemyProjectile>,
    spawn_timer: f32,
    spawn_seed: u32,
    player_visual_yaw: f32,
    snow_intensity: f32,
    snow_time: f32,
    weather: Weather,
    weather_debug_active: bool,
    weather_prev: bool,
    wind_leaf_timer: f32,
    tree_rbs: std::collections::HashSet<rapier3d::prelude::RigidBodyHandle>,
    tree_punch_seed: u32,
    // Cached per-frame player derived stats (computed once in update, reused in render).
    player_derived: DerivedStats,
    // Reusable per-frame buffers (avoid heap allocs every frame).
    frame_transforms: Vec<Mat4>,
    frame_instance_ids: Vec<u32>,
    buoyancy_bodies: Vec<(rapier3d::prelude::RigidBodyHandle, f32)>,
    dead_ids: Vec<(u32, u32)>,
    despawn_ids: Vec<u32>,
    // Reusable dirty-chunk index buffer (avoid per-frame heap allocs).
    dirty_chunk_buf: Vec<usize>,
    // Reusable hit-result buffers (avoid per-frame heap allocs).
    enemy_hit_buf: Vec<EnemyAttackHit>,
    arrow_hit_buf: Vec<EnemyAttackHit>,
    spell_hit_buf: Vec<SpellHit>,
    ui: Ui,
    particles: ParticleSystem,
    quests: Vec<Quest>,
    active_dialogue: Option<ActiveDialogue>,
    npc_interact_prev: bool,
    save_prev: bool,
    load_prev: bool,
    has_save_file: bool,
    // Minimap biome cache: 20x20 grid, recomputed only when player moves a cell.
    minimap_biome_cache: [u8; 400],
    minimap_last_cell: (i32, i32),
    minimap_prims: Vec<UiPrimitive>,
    minimap_screen_w: f32,
    // Reusable string buffer for HUD text (avoids format! heap allocs each frame).
    hud_buf: String,
    // Structure editor state.
    editor_mode: bool,
    editor_prev: bool,
    editor_grid: BuildingGrid,
    editor_physics: PhysicsWorld,
    editor_camera: GhostCamera,
    editor_color_idx: usize,
    editor_save_prev: bool,
    editor_load_prev: bool,
    editor_blueprint_idx: usize,
    editor_status: Option<(String, f32)>, // (message, remaining_seconds)
    editor_ground_inited: bool,
    editor_throw_prev: bool,
    editor_bake_prev: bool,
    editor_selected_group: Option<usize>,
    editor_prev_group_prev: bool,
    editor_next_group_prev: bool,
    editor_unbake_prev: bool,
    // Drag-to-fill state.
    drag_start: Option<(i32, i32, i32)>,
    drag_plane: Option<DragPlane>,
    drag_end: Option<(i32, i32, i32)>,
    /// Time RMB has been held; drag activates after threshold.
    drag_hold_timer: f32,
    /// Whether the drag has activated (hold exceeded threshold).
    drag_active: bool,
    // Torch instances placed in the world.
    torches: Vec<TorchInstance>,
    torch_prev: bool,
    /// Reusable per-frame point light buffer.
    frame_point_lights: Vec<GpuPointLight>,
    /// Reusable per-frame torch distance buffer (avoids allocation each frame).
    frame_torch_dists: Vec<(usize, f32)>,
    /// Time accumulator for fire particle emission (fixed rate).
    fire_particle_timer: f32,
    /// Cached player biome from update(), reused in render() to avoid redundant lookups.
    cached_player_biome: Biome,
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
    let (entities, player_id, next_id) = scene::build_scene(&mut physics);
    progress.store(1, Ordering::Relaxed);

    // Generate terrain chunks.
    let terrain = TerrainGrid::generate_or_load(42);
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
            let rb = physics.add_compound_static(&trunks);
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
            has_save_file: crate::save::load().is_ok(),
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
        };
        engine.spawn_npcs();
        engine.spawn_world_structures();
        engine.stamp_world_blueprints();
        engine.spawn_mining_nodes();

        {
            let player = engine.world.player_mut();
            if let Some(ref mut eq) = player.equipment {
                let _ = eq.equip(ITEM_IRON_SWORD);
            }
        }

        Ok(engine)
    }

    fn spawn_mining_nodes(&mut self) {
        let positions = [
            Vec3::new(15.0, 0.0, 10.0),
            Vec3::new(-20.0, 0.0, 25.0),
            Vec3::new(30.0, 0.0, -15.0),
            Vec3::new(-10.0, 0.0, -30.0),
            Vec3::new(40.0, 0.0, 5.0),
        ];
        let sizes = [
            (3, 2, 3),
            (4, 3, 4),
            (2, 2, 2),
            (3, 3, 3),
            (4, 2, 3),
        ];
        for (&pos, &size) in positions.iter().zip(sizes.iter()) {
            let next_id = &mut self.world.next_entity_id;
            let new_entities = self.mining.spawn_node(
                &mut self.physics,
                &self.terrain,
                pos,
                size,
                &mut || {
                    let id = *next_id;
                    *next_id += 1;
                    id
                },
            );
            self.world.entities.extend(new_entities);
        }
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
            let (rb, col) = self.physics.add_static_box(pos, half);
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
            let (rb, col) = physics.add_static_box(Vec3::new(x, y, z), half);
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

    fn do_save(&mut self) {
        let player_pos = self.physics.body_position(self.player_rb);
        let stats = match self.world.player().stats.clone() {
            Some(s) => s,
            None => return,
        };
        let inventory = self.world.player().inventory.clone().unwrap_or_default();
        let equipment = self.world.player().equipment.clone().unwrap_or_default();

        // Save building grid.
        let buildings: Vec<crate::save::BuildingSave> = self.building.occupied_cells()
            .filter_map(|&(x, y, z)| {
                self.building.cell_info(x, y, z).map(|(bt, rot, subs, col)| {
                    crate::save::BuildingSave {
                        x, y, z,
                        block_type: bt as u8,
                        rotation: rot,
                        sub_blocks: subs,
                        color: [col.x, col.y, col.z],
                    }
                })
            })
            .collect();

        let torches: Vec<crate::save::TorchSave> = self.torches.iter()
            .map(|t| crate::save::TorchSave { x: t.position.x, y: t.position.y, z: t.position.z })
            .collect();

        let data = crate::save::SaveData {
            player_x: player_pos.x,
            player_y: player_pos.y,
            player_z: player_pos.z,
            camera_yaw: self.camera.yaw,
            camera_pitch: self.camera.pitch,
            stats,
            inventory,
            equipment,
            quest_states: crate::save::quests_to_save(&self.quests),
            time_of_day: self.time_of_day,
            buildings,
            torches,
        };
        match crate::save::save(&data) {
            Ok(()) => { self.has_save_file = true; }
            Err(_) => {}
        }
    }

    fn do_load(&mut self) {
        let data = match crate::save::load() {
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
        crate::save::apply_quest_saves(&mut self.quests, &data.quest_states);

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

    // -----------------------------------------------------------------------
    // Structure editor
    // -----------------------------------------------------------------------

    fn update_editor(&mut self, dt: f32, input: &InputState) {
        // Ensure editor physics has a ground plane (lazy init).
        if !self.editor_ground_inited {
            self.editor_ground_inited = true;
            use crate::physics::body::SharedShape;
            let half_ext = Vec3::new(200.0, 0.5, 200.0);
            self.editor_physics.add_static_shape(
                Vec3::new(0.0, -0.5, 0.0),
                SharedShape::cuboid(half_ext.x, half_ext.y, half_ext.z),
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
            // Holding RMB — accumulate hold time, activate drag after threshold.
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

        // Remove block (LMB — edge triggered).
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

        // Bake group (G) — merge current cells into a single object.
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

        // Unbake selected group (U) — restore to editable cells.
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

    fn editor_save_blueprint(&mut self) {
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

    fn editor_load_blueprint(&mut self) {
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
                let p = self.world.player();
                let bonuses = p.equipment.as_ref()
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

            // --- Remove dead enemies + award XP + loot ---
            self.dead_ids.clear();
            self.dead_ids.extend(
                self.world.entities.iter()
                    .filter(|e| e.kind == EntityKind::Enemy)
                    .filter(|e| e.stats.as_ref().map_or(false, |s| s.is_dead()))
                    .map(|e| {
                        let level = e.stats.as_ref().map_or(1, |s| s.level);
                        (e.id, level)
                    }),
            );
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

            // --- Night enemy spawning ---
            if enemy_ai::is_night(self.time_of_day) {
                self.spawn_timer -= dt;
                if self.spawn_timer <= 0.0 {
                    self.try_spawn_enemy(player_pos);
                    // Spawn interval: 2-4 seconds.
                    self.spawn_timer = 2.0 + enemy_ai::cheap_rand_pub(&mut self.spawn_seed) * 2.0;
                }
            } else {
                // Daytime: despawn enemies far from player.
                self.despawn_ids.clear();
                self.despawn_ids.extend(
                    self.world.entities.iter()
                        .filter(|e| e.kind == EntityKind::Enemy)
                        .filter(|e| {
                            let epos = self.physics.body_position(e.body.rigid_body);
                            (epos - player_pos).length() > enemy_ai::MAX_SPAWN_DIST * 1.5
                        })
                        .map(|e| e.id),
                );
                for i in 0..self.despawn_ids.len() {
                    let id = self.despawn_ids[i];
                    if let Some(entity) = self.world.remove_by_id(id) {
                        self.physics.remove_body(entity.body.rigid_body, entity.body.collider);
                        self.enemy_ais.remove(&id);
                    }
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

    /// Rebuild terrain chunk meshes, BLASes, and physics heightfield for dirty chunks.
    fn rebuild_dirty_terrain(&mut self) {
        self.terrain.drain_dirty_chunks(&mut self.dirty_chunk_buf);
        if self.dirty_chunk_buf.is_empty() {
            return;
        }

        // Regenerate chunk meshes.
        let updates: Vec<(usize, Vec<crate::renderer::mesh::Vertex>, Vec<u32>)> = self.dirty_chunk_buf
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

    /// Apply camera-relative movement to the player rigid body.
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

    fn render_editor(&mut self) -> Result<()> {
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

        // Ground plane — large flat cube at y = -0.01.
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
        let blizzard_info = Vec4::new(0.0, 0.0, WATER_LEVEL, 0.0);
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

        // In ghost mode, frustum-cull to the frozen camera so only visible
        // geometry appears.  In normal mode, skip culling so off-screen
        // entities can still cast shadows.
        let ghost_frustum = if self.ghost.active {
            Some(&frustum_planes)
        } else {
            None
        };

        // World entities (skip the player entity — we render the model instead).
        for entity in &self.world.entities {
            if entity.kind == EntityKind::Player { continue; }

            let pos = self.physics.body_position(entity.body.rigid_body);
            if let Some(planes) = ghost_frustum {
                if !is_sphere_in_frustum(planes, pos, entity.bounding_radius) {
                    continue;
                }
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

        // Player body capsule (shadow-only — invisible to camera, casts shadow).
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
            let t = Mat4::from_translation(torch.position);
            self.frame_transforms.push(t);
            self.frame_instance_ids.push(pack_instance_id(MESH_TORCH, TORCH_OBJECT_BASE + (i as u32 & 0xFF)));
        }

        // Particles.
        self.particles.render(&mut self.frame_transforms, &mut self.frame_instance_ids);

        // Spell projectiles.
        let projectile_object_base: u32 = 0xFFA0;
        for (i, proj) in self.spells.projectiles.iter().enumerate() {
            let t = Mat4::from_translation(proj.position) * Mat4::from_scale(Vec3::splat(proj.scale));
            self.frame_transforms.push(t);
            self.frame_instance_ids.push(pack_instance_id(proj.mesh_type, projectile_object_base + i as u32));
        }

        // Enemy projectiles (arrows).
        let enemy_proj_object_base: u32 = 0xFF90;
        for (i, proj) in self.enemy_projectiles.iter().enumerate() {
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

        // Terrain chunks — distance cull + frustum cull in ghost mode.
        const TERRAIN_CULL_DIST_SQ: f32 = 1500.0 * 1500.0;
        for chunk in &self.terrain_chunks {
            let dx = chunk.center.x - player_pos.x;
            let dz = chunk.center.z - player_pos.z;
            if dx * dx + dz * dz > TERRAIN_CULL_DIST_SQ {
                continue;
            }
            if let Some(planes) = ghost_frustum {
                if !is_sphere_in_frustum(planes, chunk.center, chunk.radius) {
                    continue;
                }
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
        // Wave period ~52.36 (2*PI/0.12), so wrap offset to stay seamless.
        let wave_period = std::f32::consts::TAU / 0.12;
        let water_offset = (self.water_time * 2.0) % wave_period;
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
            crate::interaction::ToolType::Hands => 0.0,
            crate::interaction::ToolType::Axe => 1.0,
            crate::interaction::ToolType::Pickaxe => 2.0,
            crate::interaction::ToolType::Hammer => 3.0,
            crate::interaction::ToolType::Chisel => 4.0,
        };

        // Debug overlay data.
        let biome_id = match self.cached_player_biome {
            crate::terrain::Biome::Plains => 0.0,
            crate::terrain::Biome::Forest => 1.0,
            crate::terrain::Biome::Desert => 2.0,
            crate::terrain::Biome::Mountains => 3.0,
            crate::terrain::Biome::Dungeon => 4.0,
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
        // Sun direction in world space — allow going below horizon so it visually sets.
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
        // Instead pass moon dir via lightDir.w (was 0.0) — but lightDir is already used.
        // Simplest: add a second vec4 for moon.
        let moon_info = Vec4::new(moon_dir.x, moon_dir.y, moon_dir.z, moon_altitude);

        let blizzard_info = Vec4::new(self.snow_intensity, self.snow_time, WATER_LEVEL, 0.0);

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

    // -----------------------------------------------------------------------
    // UI building — immediate-mode, runs each frame
    // -----------------------------------------------------------------------

    fn build_editor_ui(&mut self) {
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

        // Help text.
        let help = "1-0: Color  RMB: Place (drag to fill)  LMB: Remove  B: Block  V: Rotate  G: Bake  U: Unbake  </>: Groups  F9: Save  F10: Load  F8: Exit";
        let help_w = help.len() as f32 * cell * 0.5;
        self.ui.text((sw - help_w) * 0.5, sh - 28.0, help, scale * 0.5, [0.7, 0.7, 0.7, 1.0]);

        // Block count and type.
        self.hud_buf.clear();
        let gc = self.editor_grid.group_count();
        if let Some(sel) = self.editor_selected_group {
            let _ = write!(self.hud_buf, "Blocks: {}  Group: {}/{}  [{}] r:{}", self.editor_grid.cell_count(), sel + 1, gc, self.selected_block_type.name(), self.selected_rotation);
        } else {
            let _ = write!(self.hud_buf, "Blocks: {}  Groups: {}  [{}] r:{}", self.editor_grid.cell_count(), gc, self.selected_block_type.name(), self.selected_rotation);
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

    fn build_menu_ui(&mut self) {
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
            &["New Game", "Continue", "Quit"]
        } else {
            &["New Game", "Quit"]
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

    fn build_ui(
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
                _ => "DUNGEON",
            };
            let biome_color = match biome_id as u32 {
                0 => [0.5, 0.8, 0.3, 1.0],
                1 => [0.2, 0.6, 0.2, 1.0],
                2 => [0.9, 0.8, 0.4, 1.0],
                3 => [0.7, 0.7, 0.9, 1.0],
                _ => [0.6, 0.4, 0.7, 1.0],
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
                crate::weather::WeatherKind::Clear => "CLEAR",
                crate::weather::WeatherKind::Cloudy => "CLOUDY",
                crate::weather::WeatherKind::Rain => "RAIN",
                crate::weather::WeatherKind::Thunderstorm => "STORM",
                crate::weather::WeatherKind::Fog => "FOG",
                crate::weather::WeatherKind::Windy => "WINDY",
            };
            let weather_color = match self.weather.kind {
                crate::weather::WeatherKind::Clear => [0.9, 0.9, 0.5, 1.0],
                crate::weather::WeatherKind::Cloudy => [0.7, 0.7, 0.7, 1.0],
                crate::weather::WeatherKind::Rain => [0.4, 0.6, 0.9, 1.0],
                crate::weather::WeatherKind::Thunderstorm => [0.9, 0.4, 0.9, 1.0],
                crate::weather::WeatherKind::Fog => [0.6, 0.65, 0.7, 1.0],
                crate::weather::WeatherKind::Windy => [0.5, 0.9, 0.7, 1.0],
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
                            _ => [0.3, 0.2, 0.35, 0.8],   // Dungeon
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

            // NPC and enemy dots.
            for e in self.world.entities.iter() {
                let (size, color) = match e.kind {
                    EntityKind::Npc => (2.0, [0.2, 0.8, 1.0, 1.0]),
                    EntityKind::Enemy => (1.5, [0.9, 0.2, 0.2, 1.0]),
                    _ => continue,
                };
                let epos = self.physics.body_position(e.body.rigid_body);
                let dx = (epos.x - player_pos.x) / world_scale;
                let dz = (epos.z - player_pos.z) / world_scale;
                if dx.abs() < half && dz.abs() < half {
                    let mx = map_x + (dx + half) * map_pixel;
                    let my = map_y + (dz + half) * map_pixel;
                    self.ui.rect(mx - size, my - size, size * 2.0, size * 2.0, color);
                }
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

        // -- Inventory screen (I key) --
        if self.show_inventory {
            self.build_inventory_ui(scale, cell);
        }
    }

    fn build_inventory_ui(&mut self, scale: f32, cell: f32) {
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
}
