//TODO: fix references when refactoring

//#![allow(unused)]
//use gpu_allocator::{vulkan::*, MemoryLocation};
//use std::io::prelude::*;
//use std::path::Path;
//use std::ptr;
//use std::time;

use ash::vk;
use raw_window_handle::HasRawDisplayHandle;
use std::ffi::c_char;
use winit::{
    dpi::PhysicalSize, event, event::Event, event_loop::EventLoop, window::Window,
    window::WindowBuilder,
};

use crate::constants::*;

//
//struct SurfaceDetails {
//    surface_fn: ash::extensions::khr::Surface,
//    surface: vk::SurfaceKHR,
//}
//
//fn get_surface_extensions(
//    event_loop: &EventLoop<()>,
//) -> Result<&'static [*const c_char], &'static str> {
//    ash_window::enumerate_required_extensions(event_loop.raw_display_handle())
//        .map_err(|_| "Couldn't enumerate required extensions")
//}
//
//// TODO: fix extensions vec
//fn create_instance(
//    entry: &ash::Entry,
//    extension_names: Option<&'static [*const c_char]>,
//) -> Result<ash::Instance, &'static str> {
//    let application_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
//    let create_info = if let Some(ext_names) = extension_names {
//        vk::InstanceCreateInfo::builder()
//            .application_info(&application_info)
//            .enabled_extension_names(ext_names)
//    } else {
//        vk::InstanceCreateInfo::builder().application_info(&application_info)
//    };
//    unsafe { entry.create_instance(&create_info, None) }.map_err(|_| "Couldn't create instance")
//}
//
//// TODO: add physical device rating for selecting gpus, add checking for graphics bit
//fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice, &'static str> {
//    unsafe {
//        instance
//            .enumerate_physical_devices()
//            .map_err(|_| "Couldn't enumerate physical devices.")
//    }?
//    .into_iter()
//    .filter(|physical_device| {
//        //TODO: fix checking for swapchain extension
//        // let extension_properties =
//        //     unsafe { instance.enumerate_device_extension_properties(*physical_device) }
//        //         .map_err(|_| "Couldn't enumerate device extension properties")
//        //         .unwrap();
//        // for property in extension_properties {
//        //     println!("{:?}", property);
//        // }
//        let current_features = unsafe { instance.get_physical_device_features(*physical_device) };
//        let current_properties =
//            unsafe { instance.get_physical_device_properties(*physical_device) };
//        current_features.sample_rate_shading != 0 // && current_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
//    })
//    .next()
//    .ok_or_else(|| "No physical devices available.")
//}
//
//fn choose_queue_family_index(
//    instance: &ash::Instance,
//    physical_device: vk::PhysicalDevice,
//    surface_details: &SurfaceDetails,
//) -> Result<usize, &'static str> {
//    let swapchain_support_details = SwapchainSupportDetails::new(physical_device, surface_details)?;
//    if swapchain_support_details.formats.len() == 0
//        || swapchain_support_details.present_modes.len() == 0
//    {
//        return Err("Not enough formats and/or present modes");
//    }
//    let mut queue_family_properties =
//        unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
//            .into_iter()
//            .enumerate()
//            .filter(|queue_family_properties| {
//                queue_family_properties
//                    .1
//                    .queue_flags
//                    .intersects(vk::QueueFlags::GRAPHICS)
//            })
//            .collect::<Vec<_>>();
//
//    // TODO: use filter_map instead
//    queue_family_properties
//        .into_iter()
//        .filter(|enumerated_queue| unsafe {
//            surface_details
//                .surface_fn
//                .get_physical_device_surface_support(
//                    physical_device,
//                    enumerated_queue.0 as u32,
//                    surface_details.surface,
//                )
//                .unwrap()
//        })
//        .map(|enumerated_queue| enumerated_queue.0)
//        .next()
//        .ok_or_else(|| "Couldn't return queue family index")
//}
//
//fn create_logical_device(
//    instance: &ash::Instance,
//    physical_device: vk::PhysicalDevice,
//    queue_family_index: u32,
//) -> Result<ash::Device, &'static str> {
//    let priorities = [1.0];
//    let queue_create_info = vk::DeviceQueueCreateInfo::builder()
//        .queue_family_index(queue_family_index)
//        .queue_priorities(&priorities);
//
//    let extensions = [ash::extensions::khr::Swapchain::name().as_ptr()];
//
//    let create_info = vk::DeviceCreateInfo::builder()
//        .queue_create_infos(std::slice::from_ref(&queue_create_info))
//        .enabled_extension_names(&extensions);
//
//    let instance = unsafe {
//        instance
//            .create_device(physical_device, &create_info, None)
//            .map_err(|_| "Couldn't create logical device.")
//    }?;
//
//    Ok(instance)
//}
//
//fn get_queue_at_index(device: &ash::Device, index: u32) -> vk::Queue {
//    unsafe { device.get_device_queue(index, 0) }
//}
//
//struct SwapchainSupportDetails {
//    capabilities: vk::SurfaceCapabilitiesKHR,
//    formats: Vec<vk::SurfaceFormatKHR>,
//    present_modes: Vec<vk::PresentModeKHR>,
//}
//
//impl SwapchainSupportDetails {
//    fn new(
//        physical_device: vk::PhysicalDevice,
//        surface_details: &SurfaceDetails,
//    ) -> Result<Self, &'static str> {
//        unsafe {
//            let capabilities = surface_details
//                .surface_fn
//                .get_physical_device_surface_capabilities(physical_device, surface_details.surface)
//                .map_err(|_| "Couldn't get physical device surface capabilities")?;
//            let present_modes = surface_details
//                .surface_fn
//                .get_physical_device_surface_present_modes(physical_device, surface_details.surface)
//                .map_err(|_| "Couldn't get physical device surface present modes")?;
//            let formats = surface_details
//                .surface_fn
//                .get_physical_device_surface_formats(physical_device, surface_details.surface)
//                .map_err(|_| "Couldn't get physical device surface formats")?;
//
//            Ok(SwapchainSupportDetails {
//                capabilities,
//                formats,
//                present_modes,
//            })
//        }
//    }
//
//    fn choose_surface_format(&self) -> Result<vk::SurfaceFormatKHR, &'static str> {
//        if (self.formats.len() == 0) {
//            return Err("Not enough formats");
//        }
//
//        for format in &self.formats {
//            if format.format == vk::Format::R8G8B8A8_SRGB
//                && format.color_space == vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT
//            {
//                return Ok(*format);
//            }
//        }
//        Ok(self.formats[0])
//    }
//
//    fn choose_surface_present_mode(&self) -> Result<vk::PresentModeKHR, &'static str> {
//        // mailbox more energy consumption, fifo better on small devices
//        for present_mode in &self.present_modes {
//            if *present_mode == vk::PresentModeKHR::MAILBOX {
//                return Ok(*present_mode);
//            }
//        }
//        Ok(vk::PresentModeKHR::FIFO)
//    }
//}
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
//
//fn choose_swapchain_extent(
//    capabilities: &vk::SurfaceCapabilitiesKHR,
//    old_width: u32,
//    old_height: u32,
//) -> Result<vk::Extent2D, &'static str> {
//    if (capabilities.current_extent.width != u32::MAX) {
//        return Ok(capabilities.current_extent);
//    } else {
//        let width = num::clamp(
//            old_width,
//            capabilities.min_image_extent.width,
//            capabilities.max_image_extent.width,
//        );
//
//        let height = num::clamp(
//            old_height,
//            capabilities.min_image_extent.height,
//            capabilities.max_image_extent.height,
//        );
//        return Ok(vk::Extent2D { width, height });
//    }
//}
//
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
//
//fn create_swapchain(
//    instance: &ash::Instance,
//    device: &ash::Device,
//    physical_device: vk::PhysicalDevice,
//    surface_details: &SurfaceDetails,
//    width: u32,
//    height: u32,
//) -> Result<vk::SwapchainKHR, &'static str> {
//    let swapchain_support_details = SwapchainSupportDetails::new(physical_device, surface_details)?;
//    let present_mode = swapchain_support_details.choose_surface_present_mode()?;
//
//    let format = swapchain_support_details.choose_surface_format()?;
//    let extent = choose_swapchain_extent(&swapchain_support_details.capabilities, width, height)?;
//
//    let image_count = get_image_count(physical_device, surface_details);
//
//    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
//        .surface(surface_details.surface)
//        .min_image_count(image_count)
//        .image_format(format.format)
//        .image_color_space(format.color_space)
//        .image_extent(extent)
//        .image_array_layers(1)
//        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
//        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
//        .pre_transform(swapchain_support_details.capabilities.current_transform)
//        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
//        .present_mode(present_mode)
//        .clipped(true)
//        .old_swapchain(vk::SwapchainKHR::null());
//
//    let swapchain_fn = ash::extensions::khr::Swapchain::new(instance, device);
//
//    let swapchain = unsafe {
//        swapchain_fn
//            .create_swapchain(&swapchain_create_info, None)
//            .map_err(|_| "Couldn't create swapchain")
//    };
//    swapchain
//}
//
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
////TODO: fix error handling
//fn create_image_views(
//    device: &ash::Device,
//    swapchain_images: &Vec<vk::Image>,
//    format: vk::Format,
//) -> Result<Vec<vk::ImageView>, &'static str> {
//    let error = format!("Couldn't create image view").as_str();
//    let mut image_views: Vec<vk::ImageView> = vec![];
//    for i in 0..swapchain_images.len() {
//        let component_mapping = vk::ComponentMapping::builder()
//            .r(vk::ComponentSwizzle::IDENTITY)
//            .g(vk::ComponentSwizzle::IDENTITY)
//            .b(vk::ComponentSwizzle::IDENTITY)
//            .a(vk::ComponentSwizzle::IDENTITY)
//            .build();
//
//        let subresource_range = vk::ImageSubresourceRange::builder()
//            .aspect_mask(vk::ImageAspectFlags::COLOR)
//            .base_mip_level(0)
//            .level_count(1)
//            .base_array_layer(0)
//            .layer_count(1)
//            .build();
//
//        let image_view_create_info = vk::ImageViewCreateInfo::builder()
//            .image(swapchain_images[i])
//            .view_type(vk::ImageViewType::TYPE_2D)
//            .format(format)
//            .components(component_mapping)
//            .subresource_range(subresource_range)
//            .build();
//
//        image_views
//            .push(unsafe { device.create_image_view(&image_view_create_info, None) }.unwrap());
//    }
//    Ok(image_views)
//}
//
//fn create_shader_module(
//    device: &ash::Device,
//    bytes: &Vec<u8>,
//) -> Result<vk::ShaderModule, &'static str> {
//    let shader_module_create_info = vk::ShaderModuleCreateInfo {
//        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
//        p_next: ptr::null(),
//        flags: vk::ShaderModuleCreateFlags::empty(),
//        code_size: bytes.len(),
//        p_code: bytes.as_ptr() as *const u32,
//    };
//    unsafe {
//        device
//            .create_shader_module(&shader_module_create_info, None)
//            .map_err(|_| "Couldn't create shader module.")
//    }
//}
//
//fn read_shader_code(file_name: &str) -> Result<Vec<u8>, &'static str> {
//    let path = std::path::Path::new(file_name);
//    let file = std::fs::File::open(path).expect(&format!("Failed to find spv file at {:?}", path));
//    Ok(file.bytes().flatten().collect::<Vec<u8>>())
//}
//
//fn create_graphics_pipelines(
//    device: &ash::Device,
//    swapchain_extent: &vk::Extent2D,
//    swapchain_image_format: &vk::SurfaceFormatKHR,
//    render_pass: &vk::RenderPass,
//) -> Result<(Vec<vk::Pipeline>, vk::PipelineLayout), &'static str> {
//    //let vertex_module = create_shader_module(device, "../shaders/vert.spv").unwrap();
//    //let frag_module = create_shader_module(device, "../shaders/vert.spv").unwrap();
//    let vert_shader_code = read_shader_code("../shaders/spv/vert.spv")?;
//    let frag_shader_code = read_shader_code("../shaders/spv/frag.spv")?;
//
//    let vertex_module = create_shader_module(device, &vert_shader_code)?;
//    let frag_module = create_shader_module(device, &frag_shader_code)?;
//
//    let main_function_name = CString::new("main").expect("Couldn't make c string");
//
//    let vert_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
//        .name(main_function_name.as_c_str())
//        .stage(vk::ShaderStageFlags::VERTEX)
//        .module(vertex_module)
//        .build();
//
//    let frag_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
//        .name(main_function_name.as_c_str())
//        .stage(vk::ShaderStageFlags::FRAGMENT)
//        .module(frag_module)
//        .build();
//
//    let shader_stages = [vert_shader_stage, frag_shader_stage];
//
//    //     viewport, scissor for now, multiple viewports require setting feature
//    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
//    let pipeline_dyn_states = vk::PipelineDynamicStateCreateInfo::builder()
//        .dynamic_states(&dynamic_states)
//        .build();
//
//    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
//    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
//        primitive_restart_enable: 0,
//        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
//        ..Default::default()
//    };
//
//    let viewports = [vk::Viewport {
//        width: swapchain_extent.width as f32,
//        height: swapchain_extent.height as f32,
//        max_depth: 1.0,
//        ..Default::default()
//    }];
//
//    let scissors = [vk::Rect2D {
//        extent: *swapchain_extent,
//        ..Default::default()
//    }];
//
//    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
//        .viewports(&viewports)
//        .scissors(&scissors)
//        .build();
//
//    // most of the other options for each setting requires a gpu feature to be set
//    let rasterization_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
//        .depth_clamp_enable(false)
//        .rasterizer_discard_enable(false)
//        .polygon_mode(vk::PolygonMode::FILL)
//        .line_width(1.0)
//        .cull_mode(vk::CullModeFlags::BACK)
//        .front_face(vk::FrontFace::CLOCKWISE)
//        .depth_bias_enable(false)
//        .build();
//
//    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
//        .sample_shading_enable(false)
//        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
//        .min_sample_shading(1.0)
//        .alpha_to_coverage_enable(false)
//        .alpha_to_one_enable(false)
//        .build();
//
//    // Depth/stencil testing goes here
//
//    //TODO: fix for coloring based on opacity
//    let color_blend_attachment_state_create_info = vk::PipelineColorBlendAttachmentState::builder()
//        .color_write_mask(vk::ColorComponentFlags::RGBA)
//        .blend_enable(false)
//        .src_color_blend_factor(vk::BlendFactor::ONE)
//        .dst_color_blend_factor(vk::BlendFactor::ZERO)
//        .color_blend_op(vk::BlendOp::ADD)
//        .src_alpha_blend_factor(vk::BlendFactor::ONE)
//        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
//        .alpha_blend_op(vk::BlendOp::ADD)
//        .build();
//
//    let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
//        .logic_op_enable(false)
//        .logic_op(vk::LogicOp::COPY)
//        .attachments(&[color_blend_attachment_state_create_info])
//        .blend_constants([0.0, 0.0, 0.0, 0.0])
//        .build();
//
//    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();
//
//    let pipeline_layout = unsafe {
//        device
//            .create_pipeline_layout(&pipeline_layout_create_info, None)
//            .expect("Failed to create pipeline layout!")
//    };
//
//    let pipeline_info = [vk::GraphicsPipelineCreateInfo::builder()
//        .stages(&shader_stages)
//        .vertex_input_state(&vertex_input_info)
//        .input_assembly_state(&input_assembly_state)
//        .viewport_state(&viewport_state_create_info)
//        .rasterization_state(&rasterization_create_info)
//        .multisample_state(&multisample_state_create_info)
//        .color_blend_state(&color_blend_state_create_info)
//        .layout(pipeline_layout)
//        .render_pass(*render_pass)
//        .subpass(0)
//        .base_pipeline_handle(vk::Pipeline::null())
//        .base_pipeline_index(-1)
//        .dynamic_state(&pipeline_dyn_states)
//        .build()];
//
//    let graphics_pipelines = unsafe {
//        device
//            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
//            .expect("Couldn't create graphics pipeline.")
//    };
//
//    unsafe {
//        device.destroy_shader_module(vertex_module, None);
//        device.destroy_shader_module(frag_module, None);
//    }
//    Ok((graphics_pipelines, pipeline_layout))
//}
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

pub struct App {
    window: Window,
    instance: ash::Instance,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>) -> Result<App, &'static str> {
        let window = App::init_window(&event_loop, WIDTH, HEIGHT)?;

        let entry = unsafe { ash::Entry::load() }.map_err(|_| "Coudn't create Vulkan entry")?;
        let surface_extensions = App::get_surface_extensions(event_loop)?;

        let instance = App::create_instance(&entry, surface_extensions)?;

        Ok(App { window, instance })
    }

    pub fn run(mut self, event_loop: EventLoop<()>) -> ! {
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

    fn create_instance(
        entry: &ash::Entry,
        extension_names: &'static [*const c_char],
    ) -> Result<ash::Instance, &'static str> {
        let application_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_extension_names(extension_names);

        unsafe { entry.create_instance(&create_info, None) }.map_err(|_| "Couldn't create instance")
    }

    fn draw_frame(&self) {}
}
