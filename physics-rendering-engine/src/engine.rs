use std::sync::Arc;

use anyhow::Result;
use glam::{Mat4, Vec3};
use winit::window::Window;

use crate::renderer::Renderer;

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
    renderer: Renderer,
    render_objects: Vec<RenderObject>,
}

impl Engine {
    pub fn new(config: EngineConfig, window: &Arc<Window>) -> Result<Self> {
        let renderer = Renderer::new(window)?;
        Ok(Self {
            config,
            renderer,
            render_objects: Vec::new(),
        })
    }

    /// Advance simulation by `dt` seconds.
    /// Phase 6 — will drive physics::World and write transforms into render_objects.
    pub fn update(&mut self, _dt: f32) {}

    pub fn render(&mut self) -> Result<()> {
        self.renderer.draw_frame()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }
}
