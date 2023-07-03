use crate::constants::*;

use std::collections::HashSet;
use std::ffi::{c_char, CStr, CString};
use std::io::prelude::*;
use std::os::raw::c_void;
use std::path::Path;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::Surface;
use ash::extensions::khr::WaylandSurface;
use ash::vk;

use cgmath;
use image;
use memoffset::offset_of;
use raw_window_handle::HasRawDisplayHandle;
use winit::{event, event::Event, event_loop::EventLoop, window::Window, window::WindowBuilder};

const VERTEX_DATA: [Vertex; 4] = [
    Vertex::new(-0.75, -0.75, 1.0, 0.0, 0.0, 1.0, 0.0),
    Vertex::new(0.75, -0.75, 0.0, 1.0, 0.0, 0.0, 0.0),
    Vertex::new(0.75, 0.75, 0.0, 0.0, 1.0, 0.0, 1.0),
    Vertex::new(-0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0),
];

const INDICES_DATA: [u32; 6] = [2, 0, 1, 0, 2, 3];

#[repr(C)]
struct UniformBufferObject {
    model: cgmath::Matrix4<f32>,
    view: cgmath::Matrix4<f32>,
    proj: cgmath::Matrix4<f32>,
}

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

struct SurfaceDetails {
    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
    width: u32,
    height: u32,
}

impl SurfaceDetails {
    fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &Window,
        width: u32,
        height: u32,
    ) -> Self {
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
            width,
            height,
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

struct SyncObjects {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    inflight_fences: Vec<vk::Fence>,
}

pub struct App {
    _entry: ash::Entry,
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
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    ubo_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    sync_objects: SyncObjects,
    frame: usize,
    //is_framebuffer_resized: bool,
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
        let window = App::init_window(event_loop, WIDTH, HEIGHT)?;

        let entry = unsafe { ash::Entry::load() }.map_err(|_| "Coudn't create Vulkan entry")?;
        //let surface_extensions = App::get_surface_extensions(event_loop)?;

        let instance = App::create_instance(&entry)?;

        let surface_details = SurfaceDetails::new(&entry, &instance, &window, WIDTH, HEIGHT);

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
            &window,
        );

        let swapchain_image_views = App::create_image_views(&device, &swapchain_details);

        let render_pass = App::create_render_pass(&device, &swapchain_details);
        let ubo_layout = App::create_descriptor_set_layout(&device);
        let (graphics_pipeline, pipeline_layout) =
            App::create_graphics_pipeline(&device, &swapchain_details, &render_pass, &ubo_layout);

        let swapchain_framebuffers = App::create_framebuffers(
            &device,
            &swapchain_image_views,
            &render_pass,
            &swapchain_details,
        );

        let command_pool = App::create_command_pool(&device, &queue_indices);
        let (vertex_buffer, vertex_buffer_memory) = App::create_vertex_buffer(
            &instance,
            &device,
            physical_device,
            &command_pool,
            &graphics_queue,
        );
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (texture_image, texture_image_memory) = App::create_texture_image(
            &device,
            command_pool,
            graphics_queue,
            &physical_device_memory_properties,
            &Path::new(TEXTURE_PATH),
        );

        let texture_image_view = App::create_texture_image_view(&device, texture_image);
        let texture_sampler = App::create_texture_sampler(&device);

        let (index_buffer, index_buffer_memory) = App::create_index_buffer(
            &instance,
            &device,
            physical_device,
            command_pool,
            graphics_queue,
        );

        let (uniform_buffers, uniform_buffers_memory) = App::create_uniform_buffers(
            &instance,
            &device,
            physical_device,
            //        &physical_device_memory_properties,
            swapchain_details.images.len(),
        );

        let descriptor_pool = App::create_descriptor_pool(&device, swapchain_details.images.len());
        let descriptor_sets = App::create_descriptor_sets(
            &device,
            descriptor_pool,
            ubo_layout,
            &uniform_buffers,
            swapchain_details.images.len(),
            texture_image_view,
            texture_sampler,
        );

        let command_buffers = App::create_command_buffers(
            &device,
            &command_pool,
            &graphics_pipeline,
            &swapchain_framebuffers,
            &render_pass,
            &swapchain_details,
            &vertex_buffer,
            &index_buffer,
            &pipeline_layout,
            &descriptor_sets,
        );

        let sync_objects = App::create_sync_objects(&device);

        Ok(App {
            _entry: entry,
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
            swapchain_framebuffers,
            render_pass,
            graphics_pipeline,
            pipeline_layout,
            command_pool,
            command_buffers,
            sync_objects,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            ubo_layout,
            uniform_buffers,
            uniform_buffers_memory,
            descriptor_pool,
            descriptor_sets,
            texture_image,
            texture_image_memory,
            texture_image_view,
            texture_sampler,
            frame: 0,
        })
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
                self.window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                self.draw_frame();
            }
            Event::LoopDestroyed => {
                unsafe {
                    self.device
                        .device_wait_idle()
                        .expect("Couldn't wait device idle.")
                };
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
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            //.with_inner_size(PhysicalSize::<u32>::from((width, height)))
            .with_resizable(true)
            .build(event_loop)
            .map_err(|_| "Couldn't create window.")
    }

    #[allow(dead_code)]
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

        if layer_properties.is_empty() {
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

        let physical_device_features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .build();

        let requred_validation_layer_raw_names: Vec<CString> = App::VALIDATION_LAYERS
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();

        let _enable_layer_names: Vec<*const c_char> = requred_validation_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let extensions = [ash::extensions::khr::Swapchain::name().as_ptr()];

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos as &[vk::DeviceQueueCreateInfo])
            .enabled_extension_names(&extensions)
            .enabled_features(&physical_device_features);

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
        window: &winit::window::Window,
    ) -> SwapchainDetails {
        let swapchain_support = App::query_swapchain_support(physical_device, surface_details);

        let surface_format = App::choose_swapchain_format(&swapchain_support.formats);
        let present_mode = App::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = App::choose_swapchain_extent(&swapchain_support.capabilities, window);

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

    fn choose_swapchain_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window: &winit::window::Window,
    ) -> vk::Extent2D {
        //if capabilities.current_extent.width != u32::max_value() {
        //    capabilities.current_extent
        //} else {
        use num::clamp;
        let inner_size = window.outer_size();
        //println!("Window size: {:?}", inner_size);
        vk::Extent2D {
            width: clamp(
                inner_size.width as u32,
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: clamp(
                inner_size.height as u32,
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
        //}
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

    fn create_render_pass(
        device: &ash::Device,
        swapchain_details: &SwapchainDetails,
    ) -> vk::RenderPass {
        let attachment_description = vk::AttachmentDescription::builder()
            .format(swapchain_details.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build();

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();

        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dependency_flags(vk::DependencyFlags::empty());

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&[color_attachment_ref])
            .build();

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&[attachment_description])
            .subpasses(&[subpass])
            .dependencies(&[*dependency])
            .build();

        unsafe { device.create_render_pass(&render_pass_info, None) }
            .expect("Couldn't create render pass")
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        swapchain_details: &SwapchainDetails,
        render_pass: &vk::RenderPass,
        ubo_layout: &vk::DescriptorSetLayout,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let vert_shader_code = read_shader_code("shaders/spv/vert.spv");
        let frag_shader_code = read_shader_code("shaders/spv/frag.spv");

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
        let _dynamic_states = [vk::DynamicState::VIEWPORT]; // vk::DynamicState::SCISSOR];

        let pipeline_dyn_states = vk::PipelineDynamicStateCreateInfo::builder()
            //.dynamic_states(&dynamic_states)
            .build();

        let binding_descriptions = Vertex::get_binding_descriptions();
        let attribute_descriptions = Vertex::get_attribute_descriptions();
        //let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions)
            .build();

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

        let set_layouts = [*ubo_layout];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .build();

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Couldn't create pipeline layout.")
        };

        let pipeline_info = [vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterization_create_info)
            .multisample_state(&multisample_state_create_info)
            .color_blend_state(&color_blend_state_create_info)
            .layout(pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1)
            .dynamic_state(&pipeline_dyn_states)
            .build()];

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                .expect("Couldn't create graphics pipeline.")
        };

        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(frag_module, None);
        }
        (graphics_pipelines[0], pipeline_layout)
    }

    fn create_framebuffers(
        device: &ash::Device,
        image_views: &Vec<vk::ImageView>,
        render_pass: &vk::RenderPass,
        swapchain_details: &SwapchainDetails,
    ) -> Vec<vk::Framebuffer> {
        let mut swapchain_buffers: Vec<vk::Framebuffer> = vec![];
        for image_view in image_views {
            let attachments = [*image_view];

            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(*render_pass)
                .attachments(&attachments)
                .width(swapchain_details.extent.width)
                .height(swapchain_details.extent.height)
                .layers(1)
                .build();

            let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None) }
                .expect("Couldn't create framebuffer.");
            swapchain_buffers.push(framebuffer);
        }
        swapchain_buffers
    }

    fn create_command_pool(
        device: &ash::Device,
        queue_indices: &QueueFamilyIndices,
    ) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_indices.graphics_family.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        unsafe { device.create_command_pool(&create_info, None) }
            .expect("Couldn't create command pool")
    }

    fn create_command_buffers(
        device: &ash::Device,
        command_pool: &vk::CommandPool,
        graphics_pipeline: &vk::Pipeline,
        framebuffers: &Vec<vk::Framebuffer>,
        render_pass: &vk::RenderPass,
        swapchain_details: &SwapchainDetails,
        vertex_buffer: &vk::Buffer,
        index_buffer: &vk::Buffer,
        pipeline_layout: &vk::PipelineLayout,
        descriptor_sets: &Vec<vk::DescriptorSet>,
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(*command_pool)
            .command_buffer_count(framebuffers.len() as u32);

        let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info) }
            .expect("Couldn't create command buffers.");

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: std::ptr::null(),
                p_inheritance_info: std::ptr::null(),
                flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            };

            unsafe {
                device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Couldn't begin recording command buffer at beginning.");
            }

            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.3, 0.0, 0.0, 1.0],
                },
            }];

            let render_pass_begin_info = vk::RenderPassBeginInfo {
                s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                p_next: std::ptr::null(),
                render_pass: *render_pass,
                framebuffer: framebuffers[i],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_details.extent,
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
            };

            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    *graphics_pipeline,
                );

                let vertex_buffers = [*vertex_buffer];
                let offsets = [0_u64];
                let descriptor_sets_to_bind = [descriptor_sets[i]];
                //could use one buffer to store multiple buffers like vertex and index buffer
                // more cache friendly
                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    *index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );

                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    *pipeline_layout,
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                device.cmd_draw_indexed(command_buffer, INDICES_DATA.len() as u32, 1, 0, 0, 0);

                device.cmd_end_render_pass(command_buffer);

                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to record Command Buffer at Ending!");
            }
        }

        command_buffers
    }

    fn create_sync_objects(device: &ash::Device) -> SyncObjects {
        let mut sync_objects = SyncObjects {
            image_available_semaphores: vec![],
            render_finished_semaphores: vec![],
            inflight_fences: vec![],
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
        };

        let fence_create_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::FenceCreateFlags::SIGNALED,
        };

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                let image_available_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");

                let render_finished_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");

                let inflight_fence = device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create Fence Object!");

                sync_objects
                    .image_available_semaphores
                    .push(image_available_semaphore);
                sync_objects
                    .render_finished_semaphores
                    .push(render_finished_semaphore);
                sync_objects.inflight_fences.push(inflight_fence);
            }
        }

        sync_objects
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);

            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.device.destroy_render_pass(self.render_pass, None);
            for &image_view in self.swapchain_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_details
                .swapchain_loader
                .destroy_swapchain(self.swapchain_details.swapchain, None);
        }
    }

    fn recreate_swapchain(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Couldn't wait device idle.")
        };

        self.cleanup_swapchain();

        self.surface_details = SurfaceDetails {
            surface_loader: self.surface_details.surface_loader.clone(),
            surface: self.surface_details.surface,
            width: self.window.inner_size().width,
            height: self.window.inner_size().height,
        };

        let swapchain_details = App::create_swapchain(
            &self.instance,
            &self.device,
            self.physical_device,
            &self.surface_details,
            &self.queue_indices,
            &self.window,
        );

        self.swapchain_details.swapchain_loader = swapchain_details.swapchain_loader;
        self.swapchain_details.swapchain = swapchain_details.swapchain;
        self.swapchain_details.images = swapchain_details.images;
        self.swapchain_details.format = swapchain_details.format;
        self.swapchain_details.extent = swapchain_details.extent;

        self.swapchain_image_views = App::create_image_views(&self.device, &self.swapchain_details);

        self.render_pass = App::create_render_pass(&self.device, &self.swapchain_details);

        let (graphics_pipeline, pipeline_layout) = App::create_graphics_pipeline(
            &self.device,
            &self.swapchain_details,
            &self.render_pass,
            &self.ubo_layout,
        );

        self.graphics_pipeline = graphics_pipeline;
        self.pipeline_layout = pipeline_layout;

        self.swapchain_framebuffers = App::create_framebuffers(
            &self.device,
            &self.swapchain_image_views,
            &self.render_pass,
            &self.swapchain_details,
        );

        self.command_buffers = App::create_command_buffers(
            &self.device,
            &self.command_pool,
            &self.graphics_pipeline,
            &self.swapchain_framebuffers,
            &self.render_pass,
            &self.swapchain_details,
            &self.vertex_buffer,
            &self.index_buffer,
            &self.pipeline_layout,
            &self.descriptor_sets,
        );
    }

    //TODO: maybe use gpu_allocator instead
    fn create_vertex_buffer(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: &vk::CommandPool,
        submit_queue: &vk::Queue,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = std::mem::size_of_val(&VERTEX_DATA) as vk::DeviceSize;
        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let (staging_buffer, staging_buffer_memory) = App::create_buffer(
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            device_memory_properties,
        );

        unsafe {
            //device
            //    .bind_buffer_memory(staging_buffer, staging_buffer_memory, 0)
            //    .expect("Couldn't bind buffer.");

            let data_ptr = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Couldn't map memory.") as *mut Vertex;
            data_ptr.copy_from_nonoverlapping(VERTEX_DATA.as_ptr(), VERTEX_DATA.len());

            device.unmap_memory(staging_buffer_memory);
        }

        let (vertex_buffer, vertex_buffer_memory) = App::create_buffer(
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            device_memory_properties,
        );

        App::copy_buffer(
            device,
            submit_queue,
            command_pool,
            staging_buffer,
            vertex_buffer,
            buffer_size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer, vertex_buffer_memory)
    }

    fn copy_buffer(
        device: &ash::Device,
        submit_queue: &vk::Queue,
        command_pool: &vk::CommandPool,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        let allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_buffer_count: 1,
            command_pool: *command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
        };

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .expect("Couldn't allocate command buffer.")
        };
        let command_buffer = command_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            p_inheritance_info: std::ptr::null(),
        };

        unsafe {
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Couldn't begin command buffer.");

            let copy_regions = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }];

            device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &copy_regions);

            device
                .end_command_buffer(command_buffer)
                .expect("Couldn't end command buffer.");
        }

        let submit_info = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        }];

        unsafe {
            device
                .queue_submit(*submit_queue, &submit_info, vk::Fence::null())
                .expect("Couldn't submit queue.");
            device
                .queue_wait_idle(*submit_queue)
                .expect("Couldn't wait queue idle.");

            device.free_command_buffers(*command_pool, &command_buffers);
        }
    }

    fn find_memory_type(
        type_filter: u32,
        required_properties: vk::MemoryPropertyFlags,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> u32 {
        for (i, memory_type) in mem_properties.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) > 0
                && memory_type.property_flags.contains(required_properties)
            {
                return i as u32;
            }
        }
        panic!("Couldn't find suitable memory type.")
    }

    fn create_buffer(
        device: &ash::Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        required_memory_flags: vk::MemoryPropertyFlags,
        device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let buffer = unsafe {
            device
                .create_buffer(&buffer_create_info, None)
                .expect("Couldn't create vertex buffer.")
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = App::find_memory_type(
            mem_requirements.memory_type_bits,
            required_memory_flags,
            device_memory_properties,
        );

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: memory_type,
        };

        let buffer_memory = unsafe {
            device
                .allocate_memory(&allocate_info, None)
                .expect("Couldn't allocate vertex buffer memory.")
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind buffer.");
        }

        (buffer, buffer_memory)
    }

    fn create_index_buffer(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        submit_queue: vk::Queue,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = std::mem::size_of_val(&INDICES_DATA) as vk::DeviceSize;
        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let (staging_buffer, staging_buffer_memory) = App::create_buffer(
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            device_memory_properties,
        );

        unsafe {
            let data_ptr = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to Map Memory") as *mut u32;

            data_ptr.copy_from_nonoverlapping(INDICES_DATA.as_ptr(), INDICES_DATA.len());

            device.unmap_memory(staging_buffer_memory);
        }

        let (index_buffer, index_buffer_memory) = App::create_buffer(
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            device_memory_properties,
        );

        App::copy_buffer(
            device,
            &submit_queue,
            &command_pool,
            staging_buffer,
            index_buffer,
            buffer_size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (index_buffer, index_buffer_memory)
    }

    fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
        let ubo_layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX,
                p_immutable_samplers: std::ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                // sampler uniform
                binding: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: std::ptr::null(),
            },
        ];

        let ubo_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            binding_count: ubo_layout_bindings.len() as u32,
            p_bindings: ubo_layout_bindings.as_ptr(),
        };

        unsafe {
            device
                .create_descriptor_set_layout(&ubo_layout_create_info, None)
                .expect("Couldn't create descriptor set layout.")
        }
    }

    fn create_uniform_buffers(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        // device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
        swapchain_image_count: usize,
    ) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let buffer_size = std::mem::size_of::<UniformBufferObject>();

        let mut uniform_buffers = vec![];
        let mut uniform_buffers_memory = vec![];

        for _ in 0..swapchain_image_count {
            let (uniform_buffer, uniform_buffer_memory) = App::create_buffer(
                device,
                buffer_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                device_memory_properties,
            );
            uniform_buffers.push(uniform_buffer);
            uniform_buffers_memory.push(uniform_buffer_memory);
        }

        (uniform_buffers, uniform_buffers_memory)
    }

    fn update_uniform_buffer(&self, current_image: usize) {
        let delta_time = 0.1;
        let ubos = [UniformBufferObject {
            model: cgmath::Matrix4::from_angle_z(cgmath::Deg(90.0 * delta_time)),
            view: cgmath::Matrix4::look_at_rh(
                cgmath::Point3::new(2.0, 2.0, 2.0),
                cgmath::Point3::new(0.0, 0.0, 0.0),
                cgmath::Vector3::new(0.0, 0.0, 1.0),
            ),
            proj: cgmath::perspective(
                cgmath::Deg(45.0),
                self.swapchain_details.extent.width as f32
                    / self.swapchain_details.extent.height as f32,
                0.1,
                10.0,
            ),
        }];

        let buffer_size = (std::mem::size_of::<UniformBufferObject>() * ubos.len()) as u64;

        unsafe {
            let data_ptr =
                self.device
                    .map_memory(
                        self.uniform_buffers_memory[current_image],
                        0,
                        buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Couldn't map memory.") as *mut UniformBufferObject;

            data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

            self.device
                .unmap_memory(self.uniform_buffers_memory[current_image]);
        }
    }

    fn create_descriptor_pool(
        device: &ash::Device,
        swapchain_images_size: usize,
    ) -> vk::DescriptorPool {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: swapchain_images_size as u32,
            },
            vk::DescriptorPoolSize {
                // sampler descriptor pool
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: swapchain_images_size as u32,
            },
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            max_sets: swapchain_images_size as u32,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
        };

        unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create Descriptor Pool!")
        }
    }

    fn create_descriptor_sets(
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        uniforms_buffers: &Vec<vk::Buffer>,
        swapchain_images_size: usize,
        texture_image_view: vk::ImageView,
        texture_sampler: vk::Sampler,
    ) -> Vec<vk::DescriptorSet> {
        let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..swapchain_images_size {
            layouts.push(descriptor_set_layout);
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool,
            descriptor_set_count: swapchain_images_size as u32,
            p_set_layouts: layouts.as_ptr(),
        };

        let descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets!")
        };

        for (i, &descritptor_set) in descriptor_sets.iter().enumerate() {
            let descriptor_buffer_info = [vk::DescriptorBufferInfo {
                buffer: uniforms_buffers[i],
                offset: 0,
                range: std::mem::size_of::<UniformBufferObject>() as u64,
            }];

            let descriptor_image_infos = [vk::DescriptorImageInfo {
                sampler: texture_sampler,
                image_view: texture_image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];

            let descriptor_write_sets = [
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descritptor_set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_image_info: std::ptr::null(),
                    p_buffer_info: descriptor_buffer_info.as_ptr(),
                    p_texel_buffer_view: std::ptr::null(),
                },
                vk::WriteDescriptorSet {
                    // sampler uniform
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: std::ptr::null(),
                    dst_set: descritptor_set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: descriptor_image_infos.as_ptr(),
                    p_buffer_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                },
            ];

            unsafe {
                device.update_descriptor_sets(&descriptor_write_sets, &[]);
            }
        }

        descriptor_sets
    }

    fn create_texture_image(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        submit_queue: vk::Queue,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
        image_path: &Path,
    ) -> (vk::Image, vk::DeviceMemory) {
        let mut image_object = image::open(image_path).unwrap();

        image_object = image_object.flipv();
        let (image_width, image_height) = (image_object.width(), image_object.height());
        let image_size =
            (std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;

        let image_data = match &image_object {
            image::DynamicImage::ImageLuma8(_)
            //| image::DynamicImage::ImageBgr8(_)
            | image::DynamicImage::ImageRgb8(_) => image_object.to_rgba8().into_raw(),
            //image::DynamicImage::ImageLumaA8(_)
            //| image::DynamicImage::ImageBgra8(_)
            //| image::DynamicImage::ImageRgba8(_) => image_object.raw_pixels(),
            _ => panic!("other case for image data"),
        };

        if image_size <= 0 {
            panic!("Couldn't load texture image.")
        }

        let (staging_buffer, staging_buffer_memory) = App::create_buffer(
            device,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            *device_memory_properties,
        );

        unsafe {
            let data_ptr = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    image_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Couldn't map memory.") as *mut u8;

            data_ptr.copy_from_nonoverlapping(image_data.as_ptr(), image_data.len());

            device.unmap_memory(staging_buffer_memory);
        }

        let (texture_image, texture_image_memory) = App::create_image(
            device,
            image_width,
            image_height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            device_memory_properties,
        );

        App::transition_image_layout(
            device,
            command_pool,
            submit_queue,
            texture_image,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        App::copy_buffer_to_image(
            device,
            command_pool,
            submit_queue,
            staging_buffer,
            texture_image,
            image_width,
            image_height,
        );

        App::transition_image_layout(
            device,
            command_pool,
            submit_queue,
            texture_image,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (texture_image, texture_image_memory)
    }

    fn create_image(
        device: &ash::Device,
        width: u32,
        height: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        required_memory_properties: vk::MemoryPropertyFlags,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> (vk::Image, vk::DeviceMemory) {
        let image_create_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::ImageCreateFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
            initial_layout: vk::ImageLayout::UNDEFINED,
        };

        let texture_image = unsafe {
            device
                .create_image(&image_create_info, None)
                .expect("Couldn't create texture image.")
        };

        let image_memory_requirement =
            unsafe { device.get_image_memory_requirements(texture_image) };
        let memory_allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            allocation_size: image_memory_requirement.size,
            memory_type_index: App::find_memory_type(
                image_memory_requirement.memory_type_bits,
                required_memory_properties,
                *device_memory_properties,
            ),
        };

        let texture_image_memory = unsafe {
            device
                .allocate_memory(&memory_allocate_info, None)
                .expect("Couldn't allocate texture image memory.")
        };

        unsafe {
            device
                .bind_image_memory(texture_image, texture_image_memory, 0)
                .expect("Couldn't bind image memory.");
        }

        (texture_image, texture_image_memory)
    }

    fn create_texture_image_view(device: &ash::Device, texture_image: vk::Image) -> vk::ImageView {
        let texture_image_view =
            App::create_image_view(device, texture_image, vk::Format::R8G8B8A8_UNORM);
        texture_image_view
    }

    fn create_image_view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
    ) -> vk::ImageView {
        let imageview_create_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::ImageViewCreateFlags::empty(),
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            image,
        };

        unsafe {
            device
                .create_image_view(&imageview_create_info, None)
                .expect("Failed to create Image View!")
        }
    }

    fn create_texture_sampler(device: &ash::Device) -> vk::Sampler {
        let sampler_create_info = vk::SamplerCreateInfo {
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            mip_lod_bias: 0.0,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 16.0,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            min_lod: 0.0,
            max_lod: 0.0,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
        };

        unsafe {
            device
                .create_sampler(&sampler_create_info, None)
                .expect("Failed to create Sampler!")
        }
    }

    fn begin_single_time_command(
        device: &ash::Device,
        command_pool: vk::CommandPool,
    ) -> vk::CommandBuffer {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_buffer_count: 1,
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
        };

        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        }[0];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            p_inheritance_info: std::ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        };

        unsafe {
            device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!");
        }

        command_buffer
    }

    fn end_single_time_command(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        submit_queue: vk::Queue,
        command_buffer: vk::CommandBuffer,
    ) {
        unsafe {
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to record Command Buffer at Ending!");
        }

        let buffers_to_submit = [command_buffer];

        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: buffers_to_submit.as_ptr(),
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        }];

        unsafe {
            device
                .queue_submit(submit_queue, &submit_infos, vk::Fence::null())
                .expect("Failed to Queue Submit!");
            device
                .queue_wait_idle(submit_queue)
                .expect("Failed to wait Queue idle!");
            device.free_command_buffers(command_pool, &buffers_to_submit);
        }
    }

    fn transition_image_layout(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        submit_queue: vk::Queue,
        image: vk::Image,
        _format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let command_buffer = App::begin_single_time_command(device, command_pool);

        let src_access_mask;
        let dst_access_mask;
        let source_stage;
        let destination_stage;

        if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            src_access_mask = vk::AccessFlags::empty();
            dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            dst_access_mask = vk::AccessFlags::SHADER_READ;
            source_stage = vk::PipelineStageFlags::TRANSFER;
            destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        } else {
            panic!("Unsupported layout transition!")
        }

        let image_barriers = [vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: std::ptr::null(),
            src_access_mask,
            dst_access_mask,
            old_layout,
            new_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        }];

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_barriers,
            );
        }

        App::end_single_time_command(device, command_pool, submit_queue, command_buffer);
    }

    fn copy_buffer_to_image(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        submit_queue: vk::Queue,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) {
        let command_buffer = App::begin_single_time_command(device, command_pool);

        let buffer_image_regions = [vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            buffer_offset: 0,
            buffer_image_height: 0,
            buffer_row_length: 0,
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        }];

        unsafe {
            device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &buffer_image_regions,
            );
        }

        App::end_single_time_command(device, command_pool, submit_queue, command_buffer);
    }

    fn draw_frame(&mut self) {
        let wait_fences = [self.sync_objects.inflight_fences[self.frame]];

        let (image_index, _is_sub_optimal) = unsafe {
            self.device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Couldn't wait for fence.");

            let result = self.swapchain_details.swapchain_loader.acquire_next_image(
                self.swapchain_details.swapchain,
                std::u64::MAX,
                self.sync_objects.image_available_semaphores[self.frame],
                vk::Fence::null(),
            );

            match result {
                Ok(image_index) => image_index,
                Err(_) => panic!("Coudln't acquire Swapchain image."),
            }
        };

        self.update_uniform_buffer(image_index as usize);

        let wait_semaphores = [self.sync_objects.image_available_semaphores[self.frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.sync_objects.render_finished_semaphores[self.frame]];

        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[image_index as usize],
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
        }];

        unsafe {
            self.device
                .reset_fences(&wait_fences)
                .expect("Couldn't reset fence.");

            self.device
                .queue_submit(
                    self.graphics_queue,
                    &submit_infos,
                    self.sync_objects.inflight_fences[self.frame],
                )
                .expect("Couldn't execute queue submit.");
        }

        let swapchains = [self.swapchain_details.swapchain];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: std::ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: std::ptr::null_mut(),
        };

        let result = unsafe {
            self.swapchain_details
                .swapchain_loader
                .queue_present(self.present_queue, &present_info)
        };

        if let Err(_) = result {
            panic!("Couldn't present, possibly resize.");
        }

        if self.surface_details.width != self.window.inner_size().width
            || self.surface_details.height != self.window.inner_size().height
        {
            //println!(
            //    "surface width: {} surface height: {} window width: {} window height:{}",
            //    self.surface_details.width,
            //    self.surface_details.height,
            //    self.window.inner_size().width,
            //    self.window.inner_size().height
            //);
            self.recreate_swapchain();
        }
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.sync_objects.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.sync_objects.render_finished_semaphores[i], None);
                self.device
                    .destroy_fence(self.sync_objects.inflight_fences[i], None);
            }

            self.cleanup_swapchain();

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            for i in 0..self.uniform_buffers.len() {
                self.device.destroy_buffer(self.uniform_buffers[i], None);
                self.device
                    .free_memory(self.uniform_buffers_memory[i], None);
            }

            self.device
                .destroy_image_view(self.texture_image_view, None);
            self.device.destroy_sampler(self.texture_sampler, None);

            self.device
                .destroy_descriptor_set_layout(self.ubo_layout, None);
            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);

            self.device.destroy_image(self.texture_image, None);
            self.device.free_memory(self.texture_image_memory, None);

            self.device.destroy_command_pool(self.command_pool, None);

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

#[repr(C)]
#[derive(Clone, Debug, Copy)]
struct Vertex {
    pos: cgmath::Vector2<f32>,
    tex_coord: cgmath::Vector2<f32>,
    color: cgmath::Vector3<f32>,
}

impl Vertex {
    const fn new(x: f32, y: f32, r: f32, g: f32, b: f32, tx: f32, ty: f32) -> Self {
        let pos = cgmath::Vector2::new(x, y);
        let color = cgmath::Vector3::new(r, g, b);
        let tex_coord = cgmath::Vector2::new(tx, ty);
        Vertex {
            pos,
            color,
            tex_coord,
        }
    }

    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Self, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Self, color) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Self, tex_coord) as u32,
            },
        ]
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
    println!("[Debug]{}{}{:?}\n\n", severity, types, message);

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
    let path = Path::new(file_name);
    let file = std::fs::File::open(path).expect(&format!("Failed to find spv file at {:?}", path));
    file.bytes().flatten().collect::<Vec<u8>>()
}
