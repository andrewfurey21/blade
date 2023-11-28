//use crate::app::vulkan::App;
use vulkan::App;
//mod app;
//mod file;
//mod math;
//
mod shader;
mod utility;
mod validation;
mod vulkan;
//pub(crate) mod vulkan;

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();
    let app = App::new(&event_loop, cfg!(debug_assertions)).unwrap();
    app.run(event_loop);
}
