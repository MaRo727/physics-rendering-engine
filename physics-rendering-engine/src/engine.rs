use std::sync::Arc;

use anyhow::Result;
use glam::{Mat4, Quat, Vec3};
use winit::window::Window;

use crate::physics::body::PhysicsBody;
use crate::physics::world::PhysicsWorld;
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
    physics: PhysicsWorld,
    cube: PhysicsBody,
    renderer: Renderer,
    render_objects: Vec<RenderObject>,
}

impl Engine {
    pub fn new(config: EngineConfig, window: &Arc<Window>) -> Result<Self> {
        let renderer = Renderer::new(window)?;

        let mut physics = PhysicsWorld::new(config.gravity);

        // Falling cube — 1×1×1 box, starts 4 units above the floor.
        let cube = PhysicsBody::new_dynamic_box(
            &mut physics,
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
        );

        // Static floor — thin wide slab at y = -0.5 (top surface at y = 0).
        PhysicsBody::new_static_box(
            &mut physics,
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(5.0, 0.5, 5.0),
        );

        // Two render objects: the cube (dynamic) and the floor (static visual).
        let floor_transform = Mat4::from_scale_rotation_translation(
            Vec3::new(10.0, 1.0, 10.0),
            Quat::IDENTITY,
            Vec3::new(0.0, -0.5, 0.0),
        );

        let render_objects = vec![
            RenderObject { mesh_id: MeshId(0), transform: Mat4::IDENTITY },
            RenderObject { mesh_id: MeshId(0), transform: floor_transform },
        ];

        Ok(Self { config, physics, cube, renderer, render_objects })
    }

    pub fn update(&mut self, dt: f32) {
        self.physics.step(dt);
        self.render_objects[0].transform =
            self.physics.body_transform(self.cube.rigid_body);
    }

    pub fn render(&mut self) -> Result<()> {
        let transforms: Vec<Mat4> =
            self.render_objects.iter().map(|o| o.transform).collect();
        self.renderer.draw_frame(&transforms)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }
}
