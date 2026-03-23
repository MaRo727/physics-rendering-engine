use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::engine::{self, Engine, EngineConfig, InitData, INIT_STEPS};
use crate::input::InputState;
use crate::renderer::loading::LoadingScreen;

struct LoadingState {
    loading_screen: LoadingScreen,
    thread: Option<thread::JoinHandle<anyhow::Result<InitData>>>,
    progress: Arc<AtomicU32>,
}

pub struct App {
    engine: Option<Engine>,
    loading: Option<LoadingState>,
    window: Option<Arc<Window>>,
    input: InputState,
    last_update: Option<Instant>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            engine: None,
            loading: None,
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
                        .with_title("Loading…")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
                )
                .unwrap(),
        );

        // Show loading screen immediately — don't grab cursor yet.
        let loading_screen =
            LoadingScreen::new(&window).expect("Failed to create loading screen");
        loading_screen.present_progress(0.0).ok();

        let progress = Arc::new(AtomicU32::new(0));
        let progress_clone = progress.clone();

        let config = EngineConfig {
            window_width: 1280,
            window_height: 720,
            window_title: "Physics Rendering Engine".to_string(),
            gravity: glam::Vec3::new(0.0, -9.81, 0.0),
        };

        let thread = thread::spawn(move || engine::init_world(config, progress_clone));

        self.loading = Some(LoadingState {
            loading_screen,
            thread: Some(thread),
            progress,
        });
        self.window = Some(window.clone());
        window.request_redraw();
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.input.mouse_dx += dx as f32;
            self.input.mouse_dy += dy as f32;
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    PhysicalKey::Code(KeyCode::KeyW) => self.input.forward = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => self.input.backward = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => self.input.left = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => self.input.right = pressed,
                    PhysicalKey::Code(KeyCode::Space) => self.input.jump = pressed,
                    PhysicalKey::Code(KeyCode::ShiftLeft) => {
                        self.input.descend = pressed;
                        self.input.sprint = pressed;
                    }
                    PhysicalKey::Code(KeyCode::KeyE) => self.input.interact = pressed,
                    PhysicalKey::Code(KeyCode::KeyG) => self.input.toggle_ghost = pressed,
                    PhysicalKey::Code(KeyCode::KeyF) => self.input.spawn = pressed,
                    PhysicalKey::Code(KeyCode::Tab) => self.input.cycle_tool = pressed,
                    PhysicalKey::Code(KeyCode::F1) => self.input.debug_stats = pressed,
                    PhysicalKey::Code(KeyCode::F2) => self.input.toggle_fast = pressed,
                    PhysicalKey::Code(KeyCode::F3) => self.input.toggle_debug_ui = pressed,
                    PhysicalKey::Code(KeyCode::KeyM) => self.input.toggle_mute = pressed,
                    PhysicalKey::Code(KeyCode::F4) => self.input.fast_time = pressed,
                    PhysicalKey::Code(KeyCode::KeyQ) => self.input.cycle_spell = pressed,
                    PhysicalKey::Code(KeyCode::KeyR) => self.input.cast_spell = pressed,
                    PhysicalKey::Code(KeyCode::KeyI) => self.input.toggle_inventory = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.input.throw = state == ElementState::Pressed;
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                self.input.place = state == ElementState::Pressed;
            }
            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(loading) = &mut self.loading {
                    // Update progress bar.
                    let current = loading.progress.load(Ordering::Relaxed);
                    let frac = current as f32 / INIT_STEPS as f32;
                    loading.loading_screen.present_progress(frac).ok();

                    // Check if the init thread has finished.
                    let finished = loading
                        .thread
                        .as_ref()
                        .map_or(false, |t| t.is_finished());

                    if finished {
                        let loading = self.loading.take().unwrap();
                        let thread = loading.thread.unwrap();
                        let init_data = thread
                            .join()
                            .expect("Init thread panicked")
                            .expect("Init failed");

                        let title = init_data.config.window_title.clone();
                        let (context, swapchain) = loading
                            .loading_screen
                            .into_parts()
                            .expect("Failed to hand off Vulkan resources");

                        self.engine = Some(
                            Engine::from_init_data(init_data, context, swapchain)
                                .expect("Failed to create engine"),
                        );

                        // Now grab cursor for FPS controls.
                        if let Some(window) = self.window.as_ref() {
                            window
                                .set_cursor_grab(CursorGrabMode::Locked)
                                .or_else(|_| {
                                    window.set_cursor_grab(CursorGrabMode::Confined)
                                })
                                .ok();
                            window.set_cursor_visible(false);
                            window.set_title(&title);
                        }
                    }

                    // Keep requesting redraws during loading.
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                } else {
                    // Normal game loop.
                    let now = Instant::now();
                    let dt = self
                        .last_update
                        .map(|t| now.duration_since(t).as_secs_f32().min(0.05))
                        .unwrap_or(1.0 / 60.0);
                    self.last_update = Some(now);

                    if let Some(engine) = self.engine.as_mut() {
                        engine.update(dt, &self.input);
                        engine.render().expect("Render error");
                    }

                    self.input.mouse_dx = 0.0;
                    self.input.mouse_dy = 0.0;

                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                }
            }
            _ => {}
        }
    }
}
