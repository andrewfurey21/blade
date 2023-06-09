pub const TITLE: &str = "cad-rs";

pub const WIDTH: u32 = 800;
pub const HEIGHT: u32 = 600;

#[cfg(not(debug_assertions))]
pub const VALIDATION_ENABLED: bool = false;
#[cfg(debug_assertions)]
pub const VALIDATION_ENABLED: bool = true;
