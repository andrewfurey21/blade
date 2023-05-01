
#![allow(unused)]
use anyhow::Result;
use ash::{self, vk};


fn main() -> Result<()> {
    let entry = unsafe { ash::Entry::load() }?;
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 0, 0),
        ..Default::default()
    };
    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        ..Default::default()
    };
    let instance = unsafe { entry.create_instance(&create_info, None) }?;
    return Ok(());
}
