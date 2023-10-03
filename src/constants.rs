pub const TITLE: &str = "cad-rs";

pub const WIDTH: u32 = 800;
pub const HEIGHT: u32 = 600;

//change to false if in release mode, else true in debug mode, build.rs
pub const VALIDATION_ENABLED: bool = true;
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub const TEXTURE_PATH: &str = "assets/models/viking_room.png";
pub const MODEL_PATH: &str = "assets/models/viking_room.obj";
