use crate::app::vulkan::App;

mod app;
mod math;

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let app = App::new(&event_loop).unwrap();
    app.run(event_loop);
}
