use glam::{Mat4, Vec3};

pub struct MeshId(pub u32);

pub struct RenderObject {
    pub mesh_id: MeshId,
    pub transform: Mat4,
}

pub struct EngineConfig {
    pub window_width: u32,
    pub window_height: u32,
    pub window_title: String,
    pub gravity: Vec3,
}

pub struct Engine {
    pub config: EngineConfig,
    render_objects: Vec<RenderObject>,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            config,
            render_objects: Vec::new(),
        }
    }

    /// Advance simulation by `dt` seconds and write transforms into render_objects.
    /// Phase 1 stub — will drive physics::World and collect transforms.
    pub fn update(&mut self, _dt: f32) {}

    /// Submit render_objects to the renderer for drawing.
    /// Phase 1 stub — will call renderer.draw_frame(&self.render_objects).
    pub fn render(&mut self) {}
}
