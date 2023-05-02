#![allow(unused)]
use anyhow::{Context, Result};
use ash::{self, vk};

fn main() -> Result<()> {
    let entry = unsafe { ash::Entry::load() }?;
    //let entry = Entry::linked();
    //let entry = unsafe { ash::Entry::load_from("/home/andrewfurey21/dev/vulkan-sdk/1.3.243.0/x86_64/lib/libvulkan.so.1") }?;
    let app_info = vk::ApplicationInfo {
        api_version: vk::API_VERSION_1_3,
        ..Default::default()
    };
    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        ..Default::default()
    };
    let instance = unsafe { entry.create_instance(&create_info, None) }?;

    let physical_device = unsafe { instance.enumerate_physical_devices() }?
        .into_iter()
        .next()
        .context("No physical devices found")?;

    let queue_priorities = 1.0;
    let queue_create_infos = [
        vk::DeviceQueueCreateInfo {
            queue_family_index: 0,
            p_queue_priorities: &queue_priorities,
            ..Default::default()
        }
    ];

    let create_info = vk::DeviceCreateInfo {
        p_queue_create_infos: &queue_create_infos[0],
        ..Default::default()
    };

    let device = unsafe { instance.create_device(physical_device, &create_info, None)? };
    let queue = unsafe { device.get_device_queue(0, 0) };

    let command_pool = {
        let pool_create_info = vk::CommandPoolCreateInfo::builder().queue_family_index(0);
        unsafe { device.create_command_pool(&pool_create_info, None) }?
    };

    let command_buffer = {
        let buffer_create_info = vk::CommandBufferAllocateInfo {
            command_pool,
            command_buffer_count: 1,
            ..Default::default()
        };
        unsafe { device.allocate_command_buffers(&buffer_create_info) }?
            .into_iter()
            .next()
            .context("No command buffer found.")?
    };


    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    return Ok(());
}
