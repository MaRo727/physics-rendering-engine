use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::engine::{Engine, EngineConfig};

pub struct App {
    engine: Option<Engine>,
    window: Option<Arc<Window>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            engine: None,
            window: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Physics Rendering Engine")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .unwrap(),
        );

        let config = EngineConfig {
            window_width: 1280,
            window_height: 720,
            window_title: "Physics Rendering Engine".to_string(),
            gravity: glam::Vec3::new(0.0, -9.81, 0.0),
        };

        self.engine = Some(Engine::new(config, &window).expect("Failed to initialize engine"));
        self.window = Some(window.clone());

        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.physical_key
                    == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape)
                {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.update(1.0 / 60.0);
                    engine.render().expect("Render error");
                }
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
