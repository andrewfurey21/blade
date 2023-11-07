use crate::app::vulkan::App;

mod app;
mod file;
mod math;

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let app = App::new(&event_loop, cfg!(debug_assertions)).unwrap();
    app.run(event_loop);
}
