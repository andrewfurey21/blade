//use gpu_allocator::{vulkan::*, MemoryLocation};
//use std::path::Path;
//use std::time;

use crate::constants::*;
use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::Surface;
use ash::extensions::khr::WaylandSurface;
use ash::vk;
use raw_window_handle::HasRawDisplayHandle;
use std::collections::HashSet;
use std::ffi::{c_char, CStr, CString};
use std::io::prelude::*;
use std::os::raw::c_void;
use winit::{
    dpi::PhysicalSize, event, event::Event, event_loop::EventLoop, window::Window,
    window::WindowBuilder,
};
//
//fn create_allocator(
//    instance: &ash::Instance,
//    device: &ash::Device,
//    physical_device: vk::PhysicalDevice,
//) -> Result<Allocator, &'static str> {
//    let allocator_create_description = AllocatorCreateDesc {
//        instance: instance.clone(),
//        device: device.clone(),
//        physical_device,
//        debug_settings: Default::default(),
//        buffer_device_address: false,
//    };
//
//    Allocator::new(&allocator_create_description).map_err(|_| "Couldn't create allocator.")
//}
//
//fn create_allocation(
//    device: &ash::Device,
//    allocator: &mut Allocator,
//    buffer: ash::vk::Buffer,
//) -> Result<Allocation, &'static str> {
//    let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
//
//    let allocation_create_description = AllocationCreateDesc {
//        name: "Buffer Allocation",
//        requirements: memory_requirements,
//        location: MemoryLocation::GpuOnly,
//        linear: true,
//        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
//    };
//
//    let allocation = allocator
//        .allocate(&allocation_create_description)
//        .map_err(|_| "Couldn't create allocation.")?;
//
//    unsafe {
//        device
//            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
//            .map_err(|_| "Couldn't bind buffer to memory.")?;
//    }
//    Ok(allocation)
//}
//
//fn create_buffer(device: &ash::Device, size: u32) -> Result<vk::Buffer, &'static str> {
//    let buffer_create_info = vk::BufferCreateInfo::builder()
//        .size((size as usize * std::mem::size_of::<u32>()) as vk::DeviceSize)
//        .usage(vk::BufferUsageFlags::TRANSFER_DST);
//
//    unsafe {
//        device
//            .create_buffer(&buffer_create_info, None)
//            .map_err(|_| "Couldn't create buffer.")
//    }
//}
//
//fn create_fence(device: &ash::Device) -> Result<vk::Fence, &'static str> {
//    let create_info = vk::FenceCreateInfo::builder()
//        .flags(vk::FenceCreateFlags::SIGNALED)
//        .build();
//
//    unsafe { device.create_fence(&create_info, None) }.map_err(|_| "Couldn't create fence.")
//}
//
//fn create_surface(
//    entry: &ash::Entry,
//    instance: &ash::Instance,
//    window: &Window,
//) -> Result<vk::SurfaceKHR, &'static str> {
//    unsafe {
//        ash_window::create_surface(
//            entry,
//            instance,
//            window.raw_display_handle(),
//            window.raw_window_handle(),
//            None,
//        )
//        .map_err(|_| "Couldn't create surface")
//    }
//}
//fn get_image_count(physical_device: vk::PhysicalDevice, surface_details: &SurfaceDetails) -> u32 {
//    let swapchain_support_details =
//        SwapchainSupportDetails::new(physical_device, surface_details).unwrap();
//    let image_count = swapchain_support_details.capabilities.min_image_count + 1;
//
//    if swapchain_support_details.capabilities.max_image_count > 0 {
//        return std::cmp::min(
//            swapchain_support_details.capabilities.max_image_count,
//            image_count,
//        );
//    } else {
//        return image_count;
//    }
//}
//fn get_swapchain_images(
//    instance: &ash::Instance,
//    device: &ash::Device,
//    swapchain: vk::SwapchainKHR,
//) -> Result<Vec<vk::Image>, &'static str> {
//    let swapchain_fn = ash::extensions::khr::Swapchain::new(instance, device);
//    unsafe {
//        swapchain_fn
//            .get_swapchain_images(swapchain)
//            .map_err(|_| "Couldn't get swapchain images")
//    }
//}
//
//
//
//
//
//fn create_render_pass(
//    device: &ash::Device,
//    swapchain_image_format: &vk::Format,
//) -> Result<vk::RenderPass, &'static str> {
//    let attachment_description = vk::AttachmentDescription::builder()
//        .format(*swapchain_image_format)
//        .samples(vk::SampleCountFlags::TYPE_1)
//        .load_op(vk::AttachmentLoadOp::CLEAR)
//        .store_op(vk::AttachmentStoreOp::STORE)
//        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
//        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
//        .initial_layout(vk::ImageLayout::UNDEFINED)
//        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
//        .build();
//
//    let color_attachment_ref = vk::AttachmentReference::builder()
//        .attachment(0)
//        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
//        .build();
//
//    let dependency = vk::SubpassDependency::builder()
//        .src_subpass(vk::SUBPASS_EXTERNAL)
//        .dst_subpass(0)
//        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
//        .src_access_mask(vk::AccessFlags::empty())
//        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
//        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
//        .dependency_flags(vk::DependencyFlags::empty());
//
//    let subpass = vk::SubpassDescription::builder()
//        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
//        .color_attachments(&[color_attachment_ref])
//        .build();
//
//    let render_pass_info = vk::RenderPassCreateInfo::builder()
//        .attachments(&[attachment_description])
//        .subpasses(&[subpass])
//        .dependencies(&[*dependency])
//        .build();
//
//    unsafe { device.create_render_pass(&render_pass_info, None) }
//        .map_err(|_| "Couldn't create render pass")
//}
//
//fn create_framebuffers(
//    device: &ash::Device,
//    image_views: &Vec<vk::ImageView>,
//    render_pass: &vk::RenderPass,
//    swapchain_extent: &vk::Extent2D,
//) -> Result<Vec<vk::Framebuffer>, &'static str> {
//    let mut swapchain_buffers: Vec<vk::Framebuffer> = vec![];
//    for image_view in image_views {
//        let attachments = [*image_view];
//
//        let framebuffer_info = vk::FramebufferCreateInfo::builder()
//            .render_pass(*render_pass)
//            .attachments(&attachments)
//            .width(swapchain_extent.width)
//            .height(swapchain_extent.height)
//            .layers(1)
//            .build();
//
//        let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None) }
//            .map_err(|_| "Couldn't create framebuffer.")?;
//        swapchain_buffers.push(framebuffer);
//    }
//    Ok(swapchain_buffers)
//}
//
//fn create_command_pool(
//    device: &ash::Device,
//    queue_index: u32,
//) -> Result<vk::CommandPool, &'static str> {
//    let create_info = vk::CommandPoolCreateInfo::builder()
//        .queue_family_index(queue_index)
//        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
//
//    unsafe { device.create_command_pool(&create_info, None) }
//        .map_err(|_| "Couldn't create command pool")
//}
//
//fn create_command_buffer(
//    device: &ash::Device,
//    command_pool: vk::CommandPool,
//) -> Result<vk::CommandBuffer, &'static str> {
//    let create_info = vk::CommandBufferAllocateInfo::builder()
//        .level(vk::CommandBufferLevel::PRIMARY)
//        .command_pool(command_pool)
//        .command_buffer_count(1);
//
//    unsafe { device.allocate_command_buffers(&create_info) }
//        .map_err(|_| "Couldn't create command buffers.")?
//        .into_iter()
//        .next()
//        .ok_or_else(|| "No command buffer found.")
//}
//
//fn record_command_buffer(
//    device: &ash::Device,
//    command_buffer: &vk::CommandBuffer,
//    render_pass: &vk::RenderPass,
//    framebuffers: &Vec<vk::Framebuffer>,
//    image_index: usize,
//    swapchain_extent: &vk::Extent2D,
//    graphics_pipeline: &vk::Pipeline,
//) {
//    let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
//
//    unsafe {
//        device
//            .begin_command_buffer(*command_buffer, &command_buffer_begin_info)
//            .unwrap()
//    };
//
//    let render_area = vk::Rect2D {
//        offset: vk::Offset2D { x: 0, y: 0 },
//        extent: *swapchain_extent,
//    };
//
//    let clear_values = [vk::ClearValue {
//        color: vk::ClearColorValue {
//            float32: [0.0, 0.0, 0.0, 1.0],
//        },
//    }];
//
//    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
//        .render_pass(*render_pass)
//        .framebuffer(framebuffers[image_index])
//        .render_area(render_area)
//        .clear_values(&clear_values)
//        .build();
//
//    let viewports = [vk::Viewport::builder()
//        .width(swapchain_extent.width as f32)
//        .height(swapchain_extent.height as f32)
//        .max_depth(1.0)
//        .build()];
//
//    let scissors = [vk::Rect2D::builder().extent(*swapchain_extent).build()];
//
//    unsafe {
//        device.cmd_begin_render_pass(
//            *command_buffer,
//            &render_pass_begin_info,
//            vk::SubpassContents::INLINE,
//        );
//
//        device.cmd_bind_pipeline(
//            *command_buffer,
//            vk::PipelineBindPoint::GRAPHICS,
//            *graphics_pipeline,
//        );
//        device.cmd_set_viewport(*command_buffer, 0, &viewports);
//        device.cmd_set_scissor(*command_buffer, 0, &scissors);
//
//        device.cmd_draw(*command_buffer, 3, 1, 0, 0);
//
//        device.cmd_end_render_pass(*command_buffer);
//    }
//}
//
//fn create_semaphore(device: &ash::Device) -> Result<vk::Semaphore, &'static str> {
//    let create_info = vk::SemaphoreCreateInfo::default();
//    unsafe {
//        device
//            .create_semaphore(&create_info, None)
//            .map_err(|_| "Couldn't create semaphore")
//    }
//}
//
////device: &ash::Device,
////command_buffer: &vk::CommandBuffer,
////render_pass: &vk::RenderPass,
////framebuffers: &Vec<vk::Framebuffer>,
////image_index: usize,
////swapchain_extent: &vk::Extent2D,
////graphics_pipeline: &vk::Pipeline,
//fn draw_frame(
//    device: &ash::Device,
//    fence: &vk::Fence,
//    swapchain_fn: &ash::extensions::khr::Swapchain,
//    image_available: &vk::Semaphore,
//    swapchain: &vk::SwapchainKHR,
//    command_buffer: &vk::CommandBuffer,
//    render_pass: &vk::RenderPass,
//    framebuffers: &Vec<vk::Framebuffer>,
//    swapchain_extent: &vk::Extent2D,
//    graphics_pipeline: &vk::Pipeline,
//    render_finished: &vk::Semaphore,
//) {
//    unsafe {
//        device.wait_for_fences(&[*fence], true, std::u64::MAX);
//        device.reset_fences(&[*fence]);
//    }
//
//    let (image_index, sub_optimal) = unsafe {
//        swapchain_fn
//            .acquire_next_image(*swapchain, std::u64::MAX, *image_available, *fence)
//            .expect("Couldn't acquire next image.")
//    };
//
//    unsafe {
//        device.reset_command_buffer(*command_buffer, vk::CommandBufferResetFlags::from_raw(0));
//    }
//
//    record_command_buffer(
//        device,
//        command_buffer,
//        render_pass,
//        framebuffers,
//        image_index as usize,
//        swapchain_extent,
//        graphics_pipeline,
//    );
//
//    let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
//
//    //let submit_infos = [vk::SubmitInfo {
//    //            s_type: vk::StructureType::SUBMIT_INFO,
//    //            p_next: ptr::null(),
//    //            wait_semaphore_count: wait_semaphores.len() as u32,
//    //            p_wait_semaphores: wait_semaphores.as_ptr(),
//    //            p_wait_dst_stage_mask: wait_stages.as_ptr(),
//    //            command_buffer_count: 1,
//    //            p_command_buffers: &self.command_buffers[image_index as usize],
//    //            signal_semaphore_count: signal_semaphores.len() as u32,
//    //            p_signal_semaphores: signal_semaphores.as_ptr(),
//    //        }];
//    let submit_info = vk::SubmitInfo {
//        s_type: vk::StructureType::SUBMIT_INFO,
//        p_next: ptr::null(),
//        p_wait_semaphores: std::slice::from_ref(image_available).as_ptr(),
//        wait_semaphore_count: 1,
//        p_wait_dst_stage_mask: wait_stages.as_ptr(),
//        command_buffer_count: 1,
//        p_command_buffers: std::slice::from_ref(command_buffer).as_ptr(),
//        signal_semaphore_count: 1,
//        p_signal_semaphores: std::slice::from_ref(render_finished).as_ptr(),
//    };
//
//    unsafe { device.queue_submit() }
//
//    let present_info = vk::PresentInfoKHR::builder()
//        .wait_semaphores(std::slice::from_ref(render_finished))
//        .swapchains(std::slice::from_ref(swapchain))
//        .image_indices(std::slice::from_ref(&image_index))
//        .build();
//}
//
//fn run() -> Result<(), &'static str> {
//    let width: u32 = 400;
//    let height: u32 = 400;
//
//    let application_title = "cad-rs";
//    let resizable_window = true;
//
//    let event_loop = EventLoop::new();
//    let window = WindowBuilder::new()
//        .with_title(application_title)
//        .with_inner_size(PhysicalSize::<u32>::from((width, height)))
//        .with_resizable(resizable_window)
//        .build(&event_loop)
//        .map_err(|_| "Couldn't create window.")?;
//
//    let entry = unsafe { ash::Entry::load() }.map_err(|_| "Couldn't create Vulkan entry.")?;
//
//    let surface_extensions = get_surface_extensions(&event_loop)?;
//    let instance = create_instance(&entry, Some(surface_extensions))?;
//
//    // TODO:    Refactor into SurfaceDetails::new()
//    let surface = create_surface(&entry, &instance, &window)?;
//    let surface_fn = ash::extensions::khr::Surface::new(&entry, &instance);
//    let surface_details = SurfaceDetails {
//        surface_fn,
//        surface,
//    };
//
//    let physical_device = pick_physical_device(&instance)?;
//
//    let queue_family_index =
//        choose_queue_family_index(&instance, physical_device, &surface_details)? as u32;
//    let device = create_logical_device(&instance, physical_device, queue_family_index)?;
//
//    let queue = get_queue_at_index(&device, queue_family_index);
//
//    let swapchain_fn = ash::extensions::khr::Swapchain::new(&instance, &device);
//
//    //TODO: idea, create_swapchain maybe return tuple of swapchain and image count?
//    let swapchain = create_swapchain(
//        &instance,
//        &device,
//        physical_device,
//        &surface_details,
//        width,
//        height,
//    )?;
//
//    let swapchain_images = get_swapchain_images(&instance, &device, swapchain)?;
//
//    let swapchain_support_details =
//        SwapchainSupportDetails::new(physical_device, &surface_details)?;
//    let swapchain_extent =
//        choose_swapchain_extent(&swapchain_support_details.capabilities, width, height)?;
//    let swapchain_image_format = swapchain_support_details.choose_surface_format()?;
//
//    let image_views =
//        create_image_views(&device, &swapchain_images, swapchain_image_format.format)?;
//
//    //TODO: change from create info to info
//    //
//
//    let render_pass = create_render_pass(&device, &swapchain_image_format.format)?;
//    let graphics_pipelines = create_graphics_pipelines(
//        &device,
//        &swapchain_extent,
//        &swapchain_image_format,
//        &render_pass,
//    )?;
//
//    let framebuffers = create_framebuffers(&device, &image_views, &render_pass, &swapchain_extent)?;
//
//    let command_pool = create_command_pool(&device, queue_family_index)?;
//    let command_buffer = create_command_buffer(&device, command_pool)?;
//
//    let fence = create_fence(&device)?;
//    let image_available = create_semaphore(&device)?;
//    let render_finished = create_semaphore(&device)?;
//
//    //let mut allocator = Some(create_allocator(&instance, &device, physical_device)?);
//    //
//    //let size_of_buffer = width * height;
//    //let buffer = create_buffer(&device, size_of_buffer)?;
//    //let mut allocation = Some(create_allocation(
//    //&device,
//    //allocator.as_mut().unwrap(),
//    //buffer,
//    //)?);
//    let mut i = 0;
//    event_loop.run(move |event, _, control_flow| match event {
//        winit::event::Event::WindowEvent { window_id, event } => {
//            if window_id == window.id() {
//                if let winit::event::WindowEvent::CloseRequested = event {
//                    control_flow.set_exit();
//                }
//            }
//        }
//        winit::event::Event::MainEventsCleared => {
//            i += 1;
//            println!("{}", i);
//            draw_frame(
//                &device,
//                &fence,
//                &swapchain_fn,
//                &image_available,
//                &swapchain,
//                &command_buffer,
//                &render_pass,
//                &framebuffers,
//                &swapchain_extent,
//                &graphics_pipelines.0[0],
//                &render_finished,
//            );
//            //let command_begin_info = vk::CommandBufferBeginInfo::builder();
//            //unsafe {
//            //    device
//            //        .begin_command_buffer(command_buffer, &command_begin_info)
//            //        .unwrap()
//            //};
//
//            //let pixel_value = blue | green << 8 | red << 16;
//
//            //unsafe {
//            //    device.cmd_fill_buffer(
//            //        command_buffer,
//            //        buffer,
//            //        allocation.as_ref().unwrap().offset(),
//            //        allocation.as_ref().unwrap().size(),
//            //        pixel_value,
//            //    )
//            //};
//
//            //unsafe { device.end_command_buffer(command_buffer).unwrap() };
//
//            //let submit_info =
//            //    vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));
//
//            //unsafe {
//            //    device
//            //        .queue_submit(queue, std::slice::from_ref(&submit_info), fence)
//            //        .unwrap()
//            //};
//        }
//        //TODO: do an impl Drop instead
//        winit::event::Event::LoopDestroyed => {
//            unsafe {
//                swapchain_fn.destroy_swapchain(swapchain, None);
//
//                surface_details.surface_fn.destroy_surface(surface, None);
//                device.queue_wait_idle(queue).unwrap();
//                device.destroy_semaphore(image_available, None);
//                device.destroy_semaphore(render_finished, None);
//                device.destroy_fence(fence, None);
//                device.destroy_command_pool(command_pool, None);
//
//                device.destroy_device(None);
//                instance.destroy_instance(None);
//            }
//
//            //allocator
//            //.as_mut()
//            //.unwrap()
//            //.free(allocation.take().unwrap())
//            //.unwrap();
//            //drop(allocator.take().unwrap());
//        }
//        _ => {}
//    });
//
//    Ok(())
//}

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some()
    }
}

struct SurfaceDetails {
    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
}

impl SurfaceDetails {
    fn new(entry: &ash::Entry, instance: &ash::Instance, window: &Window) -> Self {
        let surface = unsafe {
            use winit::platform::wayland::WindowExtWayland;

            let wayland_display = window.wayland_display().unwrap();
            let wayland_surface = window.wayland_surface().unwrap();
            let wayland_create_info = vk::WaylandSurfaceCreateInfoKHR::builder()
                .display(wayland_display)
                .surface(wayland_surface)
                .build();

            let wayland_surface_loader = WaylandSurface::new(entry, instance);
            wayland_surface_loader
                .create_wayland_surface(&wayland_create_info, None)
                .expect("Failed to create surface.")
        };

        let surface_loader = ash::extensions::khr::Surface::new(entry, instance);
        SurfaceDetails {
            surface,
            surface_loader,
        }
    }
}

struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

struct SwapchainDetails {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
}

pub struct App {
    window: Window,
    instance: ash::Instance,

    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,

    surface_details: SurfaceDetails,

    physical_device: vk::PhysicalDevice,
    queue_indices: QueueFamilyIndices,
    device: ash::Device,

    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain_details: SwapchainDetails,
    swapchain_image_views: Vec<vk::ImageView>,
}

impl App {
    const VALIDATION_LAYERS: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];
    const REQUIRED_EXTENSION_NAMES: [*const i8; 3] = [
        Surface::name().as_ptr(),
        WaylandSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ];

    const DEVICE_EXTENSIONS: [&'static str; 1] = ["VK_KHR_swapchain"];

    pub fn new(event_loop: &EventLoop<()>) -> Result<App, &'static str> {
        let window = App::init_window(&event_loop, WIDTH, HEIGHT)?;

        let entry = unsafe { ash::Entry::load() }.map_err(|_| "Coudn't create Vulkan entry")?;
        //let surface_extensions = App::get_surface_extensions(event_loop)?;

        let instance = App::create_instance(&entry)?;

        let surface_details = SurfaceDetails::new(&entry, &instance, &window);

        let (debug_utils_loader, debug_messenger) = App::setup_debug_utils(&entry, &instance);

        let physical_device = App::pick_physical_device(&instance, &surface_details)?;

        let queue_indices = App::find_queue_family(&instance, physical_device, &surface_details);
        let device = App::create_logical_device(&instance, physical_device, &queue_indices)?;

        let graphics_queue =
            unsafe { device.get_device_queue(queue_indices.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_indices.present_family.unwrap(), 0) };

        let swapchain_details = App::create_swapchain(
            &instance,
            &device,
            physical_device,
            &surface_details,
            &queue_indices,
        );

        let swapchain_image_views = App::create_image_views(&device, &swapchain_details);

        Ok(App {
            window,
            instance,
            debug_utils_loader,
            debug_messenger,
            surface_details,
            physical_device,
            queue_indices,
            device,
            graphics_queue,
            present_queue,
            swapchain_details,
            swapchain_image_views,
        })
    }

    pub fn run(self, event_loop: EventLoop<()>) -> ! {
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent { window_id, event } => {
                if window_id == self.window.id() {
                    if let event::WindowEvent::CloseRequested = event {
                        control_flow.set_exit();
                    }
                }
            }
            Event::MainEventsCleared => {
                self.draw_frame();
            }
            _ => {}
        });
    }

    fn init_window(
        event_loop: &EventLoop<()>,
        width: u32,
        height: u32,
    ) -> Result<Window, &'static str> {
        WindowBuilder::new()
            .with_title(TITLE)
            .with_inner_size(PhysicalSize::<u32>::from((width, height)))
            .with_resizable(true)
            .build(event_loop)
            .map_err(|_| "Couldn't create window.")
    }

    fn get_surface_extensions(
        event_loop: &EventLoop<()>,
    ) -> Result<&'static [*const c_char], &'static str> {
        ash_window::enumerate_required_extensions(event_loop.raw_display_handle())
            .map_err(|_| "Couldn't enumerate required extensions.")
    }

    fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, &'static str> {
        if VALIDATION_ENABLED && !App::check_validation_layer_support(entry) {
            panic!("Validation layers requested, but not available!");
        }

        let application_info = vk::ApplicationInfo::builder()
            .api_version(vk::API_VERSION_1_3)
            .build();

        let debug_utils_create_info = populate_debug_messenger_create_info();

        let requred_validation_layer_raw_names: Vec<CString> = App::VALIDATION_LAYERS
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();

        let enable_layer_names: Vec<*const i8> = requred_validation_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let instance_create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: if VALIDATION_ENABLED {
                &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                    as *const c_void
            } else {
                std::ptr::null()
            },
            flags: vk::InstanceCreateFlags::empty(),
            p_application_info: &application_info,
            pp_enabled_layer_names: if VALIDATION_ENABLED {
                enable_layer_names.as_ptr()
            } else {
                std::ptr::null()
            },
            enabled_layer_count: if VALIDATION_ENABLED {
                enable_layer_names.len()
            } else {
                0
            } as u32,
            pp_enabled_extension_names: App::REQUIRED_EXTENSION_NAMES.as_ptr(),
            enabled_extension_count: App::REQUIRED_EXTENSION_NAMES.len() as u32,
        };

        unsafe { entry.create_instance(&instance_create_info, None) }
            .map_err(|_| "Couldn't create instance.")
    }

    fn check_validation_layer_support(entry: &ash::Entry) -> bool {
        let layer_properties = entry
            .enumerate_instance_layer_properties()
            .expect("Couldn't enumerate instance layer properties.");

        if layer_properties.len() <= 0 {
            eprintln!("No available layers.");
            return false;
        }

        for required_layer_name in App::VALIDATION_LAYERS.iter() {
            let mut is_layer_found = false;

            for layer_property in layer_properties.iter() {
                let test_layer_name = array_to_string(&layer_property.layer_name);
                if (*required_layer_name) == test_layer_name {
                    is_layer_found = true;
                    break;
                }
            }

            if is_layer_found == false {
                return false;
            }
        }

        true
    }

    fn setup_debug_utils(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> (ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT) {
        let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, instance);

        if VALIDATION_ENABLED == false {
            (debug_utils_loader, ash::vk::DebugUtilsMessengerEXT::null())
        } else {
            let messenger_create_info = populate_debug_messenger_create_info();

            let utils_messenger = unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&messenger_create_info, None)
                    .expect("Debug Utils Callback")
            };

            (debug_utils_loader, utils_messenger)
        }
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface_details: &SurfaceDetails,
    ) -> Result<vk::PhysicalDevice, &'static str> {
        unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(|_| "Couldn't enumerate physical devices.")
        }?
        .into_iter()
        .filter(|physical_device| {
            App::is_device_suitable(instance, *physical_device, surface_details)
        })
        .next()
        .ok_or_else(|| "No physical devices available.")
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_details: &SurfaceDetails,
    ) -> bool {
        let indices = App::find_queue_family(instance, physical_device, surface_details);

        let is_queue_family_supported = indices.is_complete();
        let is_device_extension_supported =
            App::check_device_extension_support(instance, physical_device);
        let is_swapchain_supported = if is_device_extension_supported {
            let swapchain_support = App::query_swapchain_support(physical_device, surface_details);
            !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
        } else {
            false
        };

        return is_queue_family_supported
            && is_device_extension_supported
            && is_swapchain_supported;
    }

    fn find_queue_family(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_details: &SurfaceDetails,
    ) -> QueueFamilyIndices {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut queue_family_indices = QueueFamilyIndices {
            graphics_family: None,
            present_family: None,
        };

        for (index, queue_family) in queue_families.iter().enumerate() {
            if queue_family.queue_count > 0
                && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                queue_family_indices.graphics_family = Some(index as u32);
            }

            let is_present_support = unsafe {
                surface_details
                    .surface_loader
                    .get_physical_device_surface_support(
                        physical_device,
                        index as u32,
                        surface_details.surface,
                    )
                    .expect("Couldn't get physical device surface support.")
            };

            if queue_family.queue_count > 0 && is_present_support {
                queue_family_indices.present_family = Some(index as u32);
            }

            if queue_family_indices.is_complete() {
                break;
            }
        }

        queue_family_indices
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        indices: &QueueFamilyIndices,
    ) -> Result<ash::Device, &'static str> {
        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(indices.graphics_family.unwrap());
        unique_queue_families.insert(indices.present_family.unwrap());

        let priorities = [1.0];
        let mut queue_create_infos = vec![];

        for &queue_family in unique_queue_families.iter() {
            let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)
                .build();

            queue_create_infos.push(queue_create_info);
        }

        let physical_device_features = vk::PhysicalDeviceFeatures::default();

        let requred_validation_layer_raw_names: Vec<CString> = App::VALIDATION_LAYERS
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();

        let enable_layer_names: Vec<*const c_char> = requred_validation_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let extensions = [ash::extensions::khr::Swapchain::name().as_ptr()];

        let create_info = if VALIDATION_ENABLED {
            vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_create_infos as &[vk::DeviceQueueCreateInfo])
                .enabled_extension_names(&extensions)
                .enabled_layer_names(&enable_layer_names)
                .enabled_features(&physical_device_features)
        } else {
            vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_create_infos as &[vk::DeviceQueueCreateInfo])
                .enabled_extension_names(&extensions)
                .enabled_features(&physical_device_features)
        };

        let device = unsafe {
            instance
                .create_device(physical_device, &create_info, None)
                .map_err(|_| "Couldn't create logical device.")
        }?;

        Ok(device)
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        let extensions = App::DEVICE_EXTENSIONS.clone();

        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .expect("Couldn't to get device extension properties.")
        };

        let mut available_extension_names = vec![];

        for extension in available_extensions.iter() {
            let extension_name = array_to_string(&extension.extension_name);
            available_extension_names.push(extension_name);
        }

        let mut required_extensions = HashSet::new();
        for extension in extensions.iter() {
            required_extensions.insert(extension.to_string());
        }

        for extension_name in available_extension_names.iter() {
            required_extensions.remove(*extension_name);
        }

        return required_extensions.is_empty();
    }

    fn query_swapchain_support(
        physical_device: vk::PhysicalDevice,
        surface_details: &SurfaceDetails,
    ) -> SwapchainSupportDetails {
        unsafe {
            let capabilities = surface_details
                .surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface_details.surface)
                .expect("Failed to query for surface capabilities.");

            let formats = surface_details
                .surface_loader
                .get_physical_device_surface_formats(physical_device, surface_details.surface)
                .expect("Failed to query for surface formats.");

            let present_modes = surface_details
                .surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface_details.surface)
                .expect("Failed to query for surface present mode.");

            SwapchainSupportDetails {
                capabilities,
                formats,
                present_modes,
            }
        }
    }

    fn create_swapchain(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface_details: &SurfaceDetails,
        queue_indices: &QueueFamilyIndices,
    ) -> SwapchainDetails {
        let swapchain_support = App::query_swapchain_support(physical_device, surface_details);

        let surface_format = App::choose_swapchain_format(&swapchain_support.formats);
        let present_mode = App::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = App::choose_swapchain_extent(&swapchain_support.capabilities);

        let image_count = swapchain_support.capabilities.min_image_count + 1;
        let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
        } else {
            image_count
        };

        let (image_sharing_mode, queue_family_index_count, queue_family_indices) =
            if queue_indices.graphics_family != queue_indices.present_family {
                (
                    vk::SharingMode::CONCURRENT,
                    2,
                    vec![
                        queue_indices.graphics_family.unwrap(),
                        queue_indices.present_family.unwrap(),
                    ],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, vec![])
            };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: std::ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: surface_details.surface,
            min_image_count: image_count,
            image_color_space: surface_format.color_space,
            image_format: surface_format.format,
            image_extent: extent,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            queue_family_index_count,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            image_array_layers: 1,
        };

        let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, device);
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Couldn't create swapchain.")
        };

        let swapchain_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Couldn't get swapchain images.")
        };

        SwapchainDetails {
            swapchain_loader,
            swapchain,
            format: surface_format.format,
            extent,
            images: swapchain_images,
        }
    }

    fn choose_swapchain_format(
        available_formats: &Vec<vk::SurfaceFormatKHR>,
    ) -> vk::SurfaceFormatKHR {
        // check if list contains most widely used R8G8B8A8 format with nonlinear color space
        for available_format in available_formats {
            if available_format.format == vk::Format::B8G8R8A8_SRGB
                && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return available_format.clone();
            }
        }

        // return the first format from the list
        return available_formats.first().unwrap().clone();
    }

    fn choose_swapchain_present_mode(
        available_present_modes: &Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        for &available_present_mode in available_present_modes.iter() {
            if available_present_mode == vk::PresentModeKHR::MAILBOX {
                return available_present_mode;
            }
        }

        vk::PresentModeKHR::FIFO
    }

    fn choose_swapchain_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            capabilities.current_extent
        } else {
            use num::clamp;

            vk::Extent2D {
                width: clamp(
                    WIDTH,
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: clamp(
                    HEIGHT,
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn create_image_views(
        device: &ash::Device,
        swapchain_details: &SwapchainDetails,
    ) -> Vec<vk::ImageView> {
        let mut image_views: Vec<vk::ImageView> = vec![];

        for &image in swapchain_details.images.iter() {
            let component_mapping = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY)
                .build();

            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();

            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain_details.format)
                .components(component_mapping)
                .subresource_range(subresource_range)
                .build();
            let image_view = unsafe {
                device
                    .create_image_view(&image_view_create_info, None)
                    .expect("Couldn't create image view.")
            };
            image_views.push(image_view);
        }

        image_views
    }

    fn create_shader_module(device: &ash::Device, bytes: &Vec<u8>) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: bytes.len(),
            p_code: bytes.as_ptr() as *const u32,
        };
        unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Couldn't create shader module.")
        }
    }

    fn create_graphics_pipeline(device: &ash::Device, swapchain_details: &SwapchainDetails) {
        let vert_shader_code = read_shader_code("../shaders/spv/vert.spv");
        let frag_shader_code = read_shader_code("../shaders/spv/frag.spv");

        let vertex_module = App::create_shader_module(device, &vert_shader_code);
        let frag_module = App::create_shader_module(device, &frag_shader_code);

        let main_function_name = CString::new("main").expect("Couldn't make c string.");

        let vert_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .name(main_function_name.as_c_str())
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .build();

        let frag_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .name(main_function_name.as_c_str())
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .build();

        let shader_stages = [vert_shader_stage, frag_shader_stage];

        //     viewport, scissor for now, multiple viewports require setting feature
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let pipeline_dyn_states = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states)
            .build();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            primitive_restart_enable: 0,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let viewports = [vk::Viewport {
            width: swapchain_details.extent.width as f32,
            height: swapchain_details.extent.height as f32,
            max_depth: 1.0,
            ..Default::default()
        }];

        let scissors = [vk::Rect2D {
            extent: swapchain_details.extent,
            ..Default::default()
        }];

        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors)
            .build();

        // most of the other options for each setting requires a gpu feature to be set
        let rasterization_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .build();

        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();

        // Depth/stencil testing goes here

        //TODO: fix for coloring based on opacity
        let color_blend_attachment_state_create_info =
            vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ZERO)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)
                .build();

        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&[color_blend_attachment_state_create_info])
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            .build();

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout!")
        };

        //        let pipeline_info = [vk::GraphicsPipelineCreateInfo::builder()
        //            .stages(&shader_stages)
        //            .vertex_input_state(&vertex_input_info)
        //            .input_assembly_state(&input_assembly_state)
        //            .viewport_state(&viewport_state_create_info)
        //            .rasterization_state(&rasterization_create_info)
        //            .multisample_state(&multisample_state_create_info)
        //            .color_blend_state(&color_blend_state_create_info)
        //            .layout(pipeline_layout)
        //            .render_pass(*render_pass)
        //            .subpass(0)
        //            .base_pipeline_handle(vk::Pipeline::null())
        //            .base_pipeline_index(-1)
        //            .dynamic_state(&pipeline_dyn_states)
        //            .build()];

        //        let graphics_pipelines = unsafe {
        //            device
        //                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
        //                .expect("Couldn't create graphics pipeline.")
        //        };

        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(frag_module, None);
        }
        //    Ok((graphics_pipelines, pipeline_layout))
    }

    fn draw_frame(&self) {}
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            for &imageview in self.swapchain_image_views.iter() {
                self.device.destroy_image_view(imageview, None);
            }
            self.device.destroy_device(None);
            self.surface_details
                .surface_loader
                .destroy_surface(self.surface_details.surface, None);
            if VALIDATION_ENABLED {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

fn populate_debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: std::ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            // vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: std::ptr::null_mut(),
    }
}

fn array_to_string(array: &[c_char]) -> &str {
    let raw_string = unsafe { CStr::from_ptr(array.as_ptr()) };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
}

fn read_shader_code(file_name: &str) -> Vec<u8> {
    let path = std::path::Path::new(file_name);
    let file = std::fs::File::open(path).expect(&format!("Failed to find spv file at {:?}", path));
    file.bytes().flatten().collect::<Vec<u8>>()
}
