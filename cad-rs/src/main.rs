#![allow(unused)]
use ash::{self, vk};
use bytemuck;
use gpu_allocator::{vulkan::*, MemoryLocation};
use log::*;
use softbuffer::GraphicsContext;
use std::time;
use winit::{dpi::PhysicalSize, event_loop::EventLoop, window::WindowBuilder};

fn create_instance() -> vk::Instance {
    let application_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
    let create_info = vk::InstanceCreateInfo::builder().application_info(&application_info);
    unsafe { entry.create_instance(&create_info, None) }.expect("Could not create instance")
}

fn pick_physical_device() -> vk::PhysicalDevice {
    unsafe { instance.enumerate_physical_devices().unwrap() }
        .into_iter()
        .next()
        .expect("No physical devices found")
}

fn choose_queue_family_index() -> u32 {
    unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
        .into_iter()
        .enumerate()
        .filter(|queue_family_properties| {
            queue_family_properties
                .1
                .queue_flags
                .intersects(vk::QueueFlags::GRAPHICS)
        })
        .collect::<Vec<_>>();

    queue_family_properties.sort_by_key(|queue_family_property| {
        (
            queue_family_property.1.queue_flags.as_raw().count_ones(),
            queue_family_property.1.queue_count,
        )
    });

    queue_family_properties
        .first()
        .expect("No available queue families")
        .0 as u32
}

fn create_logical_device() -> vk::Device {
    let priorities = [1.0]; // TODO: add length as arg
    let queue_create_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(index)
        .queue_priorities(&priorities);

    let create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(std::slice::from_ref(&queue_create_info));

    unsafe {
        instance
            .create_device(physical_device, &create_info, None)
            .unwrap()
    }
}

fn create_allocator() -> Option<Allocator> {
    let i = 1;
    let allocator_create_description = AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device,
        debug_settings: Default::default(),
        buffer_device_address: false,
    };

    Allocator::new(&allocator_create_description).unwrap()
}

fn run<E>() -> Result<(), E>
where
    E: std::error::Error,
{
    let width: u32 = 400;
    let height: u32 = 400;

    let application_title = "cad-rs";
    let resizable_window = true;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(application_title)
        .with_inner_size(PhysicalSize::new(width, height))
        .with_resizable(resizable_window)
        .build(&event_loop)?;

    let mut graphics_context = unsafe { GraphicsContext::new(&window, &window) }?;

    let entry = unsafe { ash::Entry::load() }?;

    let instance = create_instance();
    let physical_device = pick_physical_device();

    let queue_family_index = choose_family_queue_index();
    let device = create_logical_device();

    let queue = unsafe { device.get_device_queue(index, 0) };
    return Ok(());
}

fn main() {
    env_logger::init();
    info!("Starting up application...");

    let device = {};

    let mut allocator = Option::Some({});

    let buffer = {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size((width * height) as vk::DeviceSize * std::mem::size_of::<u32>() as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);

        unsafe { device.create_buffer(&buffer_create_info, None).unwrap() }
    };

    let mut allocation = Option::Some({
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation_create_description = AllocationCreateDesc {
            name: "Buffer Allocation",
            requirements: memory_requirements,
            location: MemoryLocation::GpuToCpu,
            linear: true, // buffers are always linear
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator
            .as_mut()
            .unwrap()
            .allocate(&allocation_create_description)
            .unwrap();

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap()
        };
        allocation
    });

    let command_pool = {
        let create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        unsafe { device.create_command_pool(&create_info, None) }.unwrap()
    };

    let command_buffer = {
        let create_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(1);

        unsafe { device.allocate_command_buffers(&create_info).unwrap() }
            .into_iter()
            .next()
            .expect("no command buffer found")
    };

    let fence = {
        let create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED)
            .build();

        unsafe { device.create_fence(&create_info, None).unwrap() }
    };

    let mut t: f64 = 0.0;
    let mut red = 0;
    let blue = 125;
    let green = 50;
    event_loop.run(move |event, _, control_flow| match event {
        winit::event::Event::WindowEvent { window_id, event } => {
            if window_id == window.id() {
                if let winit::event::WindowEvent::CloseRequested = event {
                    control_flow.set_exit();
                }
            }
        }
        winit::event::Event::MainEventsCleared => {
            let start = time::Instant::now();
            t += 0.001;
            red = ((t.sin() * 0.5 + 0.5) * 255.0) as u32;

            unsafe { device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX) }
                .unwrap();

            unsafe { device.reset_fences(std::slice::from_ref(&fence)).unwrap() };

            let command_begin_info = vk::CommandBufferBeginInfo::builder();
            unsafe {
                device
                    .begin_command_buffer(command_buffer, &command_begin_info)
                    .unwrap()
            };

            let pixel_value = blue | green << 8 | red << 16;

            unsafe {
                device.cmd_fill_buffer(
                    command_buffer,
                    buffer,
                    allocation.as_ref().unwrap().offset(),
                    allocation.as_ref().unwrap().size(),
                    pixel_value,
                )
            };

            unsafe { device.end_command_buffer(command_buffer).unwrap() };

            let submit_info =
                vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));

            unsafe {
                device
                    .queue_submit(queue, std::slice::from_ref(&submit_info), fence)
                    .unwrap()
            };

            let data = bytemuck::cast_slice(allocation.as_ref().unwrap().mapped_slice().unwrap());

            graphics_context.set_buffer(data, width as u16, height as u16);
        }
        winit::event::Event::LoopDestroyed => {
            unsafe { device.queue_wait_idle(queue) }.unwrap();

            unsafe { device.destroy_fence(fence, None) }
            unsafe { device.destroy_command_pool(command_pool, None) }

            allocator
                .as_mut()
                .unwrap()
                .free(allocation.take().unwrap())
                .unwrap();
            drop(allocator.take().unwrap());
            unsafe { device.destroy_buffer(buffer, None) }

            unsafe { device.destroy_device(None) }
            unsafe { instance.destroy_instance(None) }
        }
        _ => {}
    });
}
