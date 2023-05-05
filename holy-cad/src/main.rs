#![allow(unused)]
use anyhow::{Context, Result};
use ash::{self, vk};
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;

fn main() -> Result<()> {
    let entry = unsafe { ash::Entry::load() }?;

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


    let queue_family_index = {
        let mut queue_families_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
                .into_iter()
                .enumerate()
                .filter(|queue_family_properties| {
                    queue_family_properties.1.queue_flags.intersects(
                        vk::QueueFlags::TRANSFER
                            | vk::QueueFlags::GRAPHICS
                            | vk::QueueFlags::COMPUTE,
                    )
                })
                .collect::<Vec<_>>();
        queue_families_properties.sort_by_key(|queue_family_properties| {
            (
                queue_family_properties.1.queue_flags.as_raw().count_ones(),
                queue_family_properties.1.queue_count,
            )
        });
        queue_families_properties
            .first()
            .context("No suitable queue family")?
            .0 as u32
    };

    let queue_priorities = 1.0;
    let queue_create_infos = [vk::DeviceQueueCreateInfo {
        queue_family_index,
        p_queue_priorities: &queue_priorities,
        ..Default::default()
    }];

    let create_info = vk::DeviceCreateInfo {
        p_queue_create_infos: &queue_create_infos[0],
        ..Default::default()
    };
    let device = unsafe { instance.create_device(physical_device, &create_info, None)? };

    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let value_count = 16;

    let buffer = {
        let buffer_create_info = vk::BufferCreateInfo {
            size: (value_count as vk::DeviceSize) * (std::mem::size_of::<i32>() as vk::DeviceSize),
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            ..Default::default()
        };

        unsafe { device.create_buffer(&buffer_create_info, None) }?
    };


    let mut allocator = Option::Some({
        let allocator_create_description = AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
        };
        Allocator::new(&allocator_create_description)?
    });

    let mut allocation = Option::Some({
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation_create_desc = AllocationCreateDesc {
            name: "Buffer Allocation",
            requirements: memory_requirements,
            location: MemoryLocation::GpuToCpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator
            .as_mut()
            .unwrap()
            .allocate(&allocation_create_desc)?;

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }?;
        allocation
    });


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

    allocator
        .as_mut()
        .unwrap()
        .free(allocation.take().unwrap())
        .unwrap();
    drop(allocator.take().unwrap());

    unsafe {
        device.destroy_buffer(buffer, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    return Ok(());
}
