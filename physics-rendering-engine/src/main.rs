mod app;
mod engine;
mod input;
mod interaction;
mod physics;
mod player;
mod renderer;
mod scene;

use app::App;
use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
