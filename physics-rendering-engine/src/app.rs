use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::engine::{Engine, EngineConfig};
use crate::input::InputState;

pub struct App {
    engine: Option<Engine>,
    window: Option<Arc<Window>>,
    input: InputState,
    last_update: Option<Instant>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            engine: None,
            window: None,
            input: InputState::default(),
            last_update: None,
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

        // Grab and hide the cursor for FPS-style mouse look.
        window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
            .ok();
        window.set_cursor_visible(false);

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

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.input.mouse_dx += dx as f32;
            self.input.mouse_dy += dy as f32;
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { physical_key, state, .. },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    PhysicalKey::Code(KeyCode::KeyW) => self.input.forward  = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => self.input.backward = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => self.input.left     = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => self.input.right    = pressed,
                    PhysicalKey::Code(KeyCode::Space) => self.input.jump   = pressed,
                    PhysicalKey::Code(KeyCode::ShiftLeft) => self.input.descend = pressed,
                    PhysicalKey::Code(KeyCode::KeyE) => self.input.interact = pressed,
                    PhysicalKey::Code(KeyCode::KeyG) => self.input.toggle_ghost = pressed,
                    PhysicalKey::Code(KeyCode::KeyF) => self.input.spawn = pressed,
                    PhysicalKey::Code(KeyCode::Tab) => self.input.cycle_tool = pressed,
                    PhysicalKey::Code(KeyCode::F1) => self.input.debug_stats = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.input.throw = state == ElementState::Pressed;
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } => {
                self.input.place = state == ElementState::Pressed;
            }
            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = self.last_update
                    .map(|t| now.duration_since(t).as_secs_f32().min(0.05))
                    .unwrap_or(1.0 / 60.0);
                self.last_update = Some(now);

                if let Some(engine) = self.engine.as_mut() {
                    engine.update(dt, &self.input);
                    engine.render().expect("Render error");
                }

                // Clear mouse delta — it has been consumed for this frame.
                self.input.mouse_dx = 0.0;
                self.input.mouse_dy = 0.0;

                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
