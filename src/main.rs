use crate::app::App;

mod app;
mod constants;

fn main() {
    env_logger::init();
    // add checking for the shaders in main file
    let event_loop = winit::event_loop::EventLoop::new();
    let app = App::new(&event_loop).unwrap();
    app.run(event_loop);
}
