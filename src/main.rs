use crate::app::App;
use log::{error, info};

mod app;
mod constants;

fn main() {
    env_logger::init();

    info!("Starting up application...");
    let app = App::new();
    if let Err(desc) = app {
        error!("{}", desc);
    }

    // let run = app.run();
    // if let Err(desc) = run {
    //  error!("{}", desc);
    //}
    //
    info!("Exiting application.");
}
