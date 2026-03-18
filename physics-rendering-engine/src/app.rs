use std::num::NonZeroU32;
use std::sync::Arc;

use softbuffer::{Context, Surface};
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
    // Softbuffer placeholder — removed in Phase 3 when Vulkan renderer is ready
    sb_context: Option<Context<Arc<Window>>>,
    sb_surface: Option<Surface<Arc<Window>, Arc<Window>>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            engine: None,
            window: None,
            sb_context: None,
            sb_surface: None,
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

        let sb_context = Context::new(window.clone()).unwrap();
        let sb_surface = Surface::new(&sb_context, window.clone()).unwrap();

        let config = EngineConfig {
            window_width: 1280,
            window_height: 720,
            window_title: "Physics Rendering Engine".to_string(),
            gravity: glam::Vec3::new(0.0, -9.81, 0.0),
        };

        self.engine = Some(Engine::new(config, &window).expect("Failed to initialize engine"));
        self.sb_context = Some(sb_context);
        self.sb_surface = Some(sb_surface);
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
            WindowEvent::RedrawRequested => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.update(1.0 / 60.0);
                }

                // Softbuffer clear-color (placeholder until Vulkan renderer, Phase 3)
                if let (Some(surface), Some(window)) =
                    (self.sb_surface.as_mut(), self.window.as_ref())
                {
                    let size = window.inner_size();
                    if let (Some(w), Some(h)) =
                        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                    {
                        surface.resize(w, h).unwrap();
                        let mut buf = surface.buffer_mut().unwrap();
                        buf.fill(0xFF1E1E2E);
                        buf.present().unwrap();
                    }
                }

                if let Some(engine) = self.engine.as_mut() {
                    engine.render();
                }

                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
