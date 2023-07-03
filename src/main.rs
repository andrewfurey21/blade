use crate::app::App;
use log::error;

mod app;
mod constants;

fn main() {
    env_logger::init();
    // add checking for the shaders in main file
    let event_loop = winit::event_loop::EventLoop::new();
    let app = App::new(&event_loop);

    match app {
        Ok(app) => app.run(event_loop),
        Err(desc) => error!("{}", desc),
    }
}
