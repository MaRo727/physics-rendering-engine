mod app;
mod audio;
mod building;
mod engine;
mod game;
mod input;
mod mining;
mod physics;
mod renderer;
mod particles;
mod persistence;
mod ui;
mod world;

use app::App;
use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
