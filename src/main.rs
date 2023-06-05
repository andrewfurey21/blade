
fn main() {
    env_logger::init();

    info!("Starting up application...");

    let output = run();
    if let Err(error_description) = output {
        error!("Error: {}", error_description);
    }
    info!("Exiting application.");
}
