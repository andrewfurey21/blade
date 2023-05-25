#![allow(unused)]
use ash::vk;
use gpu_allocator::{vulkan::*, MemoryLocation};
use log::*;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::ffi::c_char;
use std::time;
use winit::{dpi::PhysicalSize, event_loop::EventLoop, window::Window, window::WindowBuilder};

struct SurfaceDetails {
    surface_fn: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
}

fn get_surface_extensions(
    event_loop: &EventLoop<()>,
) -> Result<&'static [*const c_char], &'static str> {
    ash_window::enumerate_required_extensions(event_loop.raw_display_handle())
        .map_err(|_| "Couldn't enumerate required extensions")
}

fn create_instance(
    entry: &ash::Entry,
    extension_names: Option<&'static [*const c_char]>,
) -> Result<ash::Instance, &'static str> {
    let application_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
    let create_info = if let Some(ext_names) = extension_names {
        vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_extension_names(ext_names)
    } else {
        vk::InstanceCreateInfo::builder().application_info(&application_info)
    };
    unsafe { entry.create_instance(&create_info, None) }.map_err(|_| "Couldn't create instance")
}

// TODO: add physical device rating for selecting gpus, add checking for graphics bit
fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice, &'static str> {
    unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|_| "Couldn't enumerate physical devices.")
    }?
    .into_iter()
    .filter(|physical_device| {
        //TODO: fix checking for swapchain extension
        //let extension_properties =
        //    unsafe { instance.enumerate_device_extension_properties(*physical_device) }
        //.expect("Couldn't get extension properties.").into_iter().filter(|property| {
        //    CStr::from_ptr(property.extension_name).to_str() == "VK_KHR_swapchain"
        //});
        let current_features = unsafe { instance.get_physical_device_features(*physical_device) };
        let current_properties =
            unsafe { instance.get_physical_device_properties(*physical_device) };
        current_features.sample_rate_shading != 0 // && current_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
    })
    .next()
    .ok_or_else(|| "No physical devices available.")
}

fn choose_queue_family_index(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_details: &SurfaceDetails,
) -> Result<usize, &'static str> {
    let swapchain_support_details = SwapchainSupportDetails::new(physical_device, surface_details)?;
    if swapchain_support_details.formats.len() == 0
        || swapchain_support_details.present_modes.len() == 0
    {
        return Err("Not enough formats and/or present modes");
    }
    let mut queue_family_properties =
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

    // TODO: use filter_map instead
    queue_family_properties
        .into_iter()
        .filter(|enumerated_queue| unsafe {
            surface_details
                .surface_fn
                .get_physical_device_surface_support(
                    physical_device,
                    enumerated_queue.0 as u32,
                    surface_details.surface,
                )
                .unwrap()
        })
        .map(|enumerated_queue| enumerated_queue.0)
        .next()
        .ok_or_else(|| "Couldn't return queue family index")
}

fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<ash::Device, &'static str> {
    let priorities = [1.0];
    let queue_create_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);

    let create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(std::slice::from_ref(&queue_create_info));

    let instance = unsafe {
        instance
            .create_device(physical_device, &create_info, None)
            .map_err(|_| "Couldn't create logical device.")
    }?;

    Ok(instance)
}

fn get_queue_at_index(device: &ash::Device, index: u32) -> vk::Queue {
    unsafe { device.get_device_queue(index, 0) }
}

struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    fn new(
        physical_device: vk::PhysicalDevice,
        surface_details: &SurfaceDetails,
    ) -> Result<Self, &'static str> {
        unsafe {
            let capabilities = surface_details
                .surface_fn
                .get_physical_device_surface_capabilities(physical_device, surface_details.surface)
                .map_err(|_| "Couldn't get physical device surface capabilities")?;
            let present_modes = surface_details
                .surface_fn
                .get_physical_device_surface_present_modes(physical_device, surface_details.surface)
                .map_err(|_| "Couldn't get physical device surface present modes")?;
            let formats = surface_details
                .surface_fn
                .get_physical_device_surface_formats(physical_device, surface_details.surface)
                .map_err(|_| "Couldn't get physical device surface formats")?;

            Ok(SwapchainSupportDetails {
                capabilities,
                formats,
                present_modes,
            })
        }
    }

    fn choose_surface_format(&self) -> Result<vk::SurfaceFormatKHR, &'static str> {
        if (self.formats.len() == 0) {
            return Err("Not enough formats");
        }

        for format in &self.formats {
            if format.format == vk::Format::R8G8B8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT
            {
                return Ok(*format);
            }
        }
        Ok(self.formats[0])
    }

    fn choose_surface_present_mode(&self) -> Result<vk::PresentModeKHR, &'static str> {
        // mailbox more energy consumption, fifo better on small devices
        for present_mode in &self.present_modes {
            if present_mode == vk::PresentModeKHR::MAILBOX {
                return Ok(*present_mode);
            }
        }
        Ok(vk::PresentModeKHR::FIFO)
    }
}

fn create_allocator(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
) -> Result<Allocator, &'static str> {
    let allocator_create_description = AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device,
        debug_settings: Default::default(),
        buffer_device_address: false,
    };

    Allocator::new(&allocator_create_description).map_err(|_| "Couldn't create allocator.")
}

fn create_allocation(
    device: &ash::Device,
    allocator: &mut Allocator,
    buffer: ash::vk::Buffer,
) -> Result<Allocation, &'static str> {
    let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let allocation_create_description = AllocationCreateDesc {
        name: "Buffer Allocation",
        requirements: memory_requirements,
        location: MemoryLocation::GpuOnly,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    };

    let allocation = allocator
        .allocate(&allocation_create_description)
        .map_err(|_| "Couldn't create allocation.")?;

    unsafe {
        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .map_err(|_| "Couldn't bind buffer to memory.")?;
    }
    Ok(allocation)
}

fn create_buffer(device: &ash::Device, size: u32) -> Result<vk::Buffer, &'static str> {
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size((size as usize * std::mem::size_of::<u32>()) as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::TRANSFER_DST);

    unsafe {
        device
            .create_buffer(&buffer_create_info, None)
            .map_err(|_| "Couldn't create buffer.")
    }
}

fn create_command_pool(
    device: &ash::Device,
    queue_index: u32,
) -> Result<vk::CommandPool, &'static str> {
    let create_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    unsafe { device.create_command_pool(&create_info, None) }
        .map_err(|_| "Couldn't create command pool")
}

fn create_command_buffer(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, &'static str> {
    let create_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);

    unsafe { device.allocate_command_buffers(&create_info) }
        .map_err(|_| "Couldn't create command buffers.")?
        .into_iter()
        .next()
        .ok_or_else(|| "No command buffer found.")
}

fn create_fence(device: &ash::Device) -> Result<vk::Fence, &'static str> {
    let create_info = vk::FenceCreateInfo::builder()
        .flags(vk::FenceCreateFlags::SIGNALED)
        .build();

    unsafe { device.create_fence(&create_info, None) }.map_err(|_| "Couldn't create fence.")
}

fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR, &'static str> {
    unsafe {
        ash_window::create_surface(
            entry,
            instance,
            window.raw_display_handle(),
            window.raw_window_handle(),
            None,
        )
        .map_err(|_| "Couldn't create surface")
    }
}

fn run() -> Result<(), &'static str> {
    let width: u32 = 400;
    let height: u32 = 400;

    let application_title = "cad-rs";
    let resizable_window = true;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(application_title)
        .with_inner_size(PhysicalSize::<u32>::from((width, height)))
        .with_resizable(resizable_window)
        .build(&event_loop)
        .map_err(|_| "Couldn't create window.")?;

    let entry = unsafe { ash::Entry::load() }.map_err(|_| "Couldn't create Vulkan entry.")?;

    let surface_extensions = get_surface_extensions(&event_loop)?;
    let instance = create_instance(&entry, Some(surface_extensions))?;

    // TODO:    Refactor into SurfaceDetails::new()
    let surface = create_surface(&entry, &instance, &window)?;
    let surface_fn = ash::extensions::khr::Surface::new(&entry, &instance);
    let surface_details = SurfaceDetails {
        surface_fn,
        surface,
    };

    let physical_device = pick_physical_device(&instance)?;

    let queue_family_index =
        choose_queue_family_index(&instance, physical_device, &surface_details)? as u32;
    let device = create_logical_device(&instance, physical_device, queue_family_index)?;

    let queue = get_queue_at_index(&device, queue_family_index);

    let mut allocator = Some(create_allocator(&instance, &device, physical_device)?);

    let size_of_buffer = width * height;
    let buffer = create_buffer(&device, size_of_buffer)?;
    let mut allocation = Some(create_allocation(
        &device,
        allocator.as_mut().unwrap(),
        buffer,
    )?);

    let command_pool = create_command_pool(&device, queue_family_index)?;
    let command_buffer = create_command_buffer(&device, command_pool)?;

    let fence = create_fence(&device)?;

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
        }
        winit::event::Event::LoopDestroyed => {
            unsafe {
                surface_details.surface_fn.destroy_surface(surface, None);
                device.queue_wait_idle(queue).unwrap();
                device.destroy_fence(fence, None);
                device.destroy_command_pool(command_pool, None);

                device.destroy_buffer(buffer, None);
                device.destroy_device(None);
                instance.destroy_instance(None);
            }

            allocator
                .as_mut()
                .unwrap()
                .free(allocation.take().unwrap())
                .unwrap();
            drop(allocator.take().unwrap());
        }
        _ => {}
    });

    Ok(())
}

fn main() {
    env_logger::init();
    info!("Starting up application...");

    let output = run();
    if let Err(error_description) = output {
        error!("Error: {}", error_description);
    }
    info!("Exiting application.");
}
