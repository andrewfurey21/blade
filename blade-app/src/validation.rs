use crate::utility;

use ash::vk;
use colored::Colorize;

use std::ffi::CStr;
use std::os::raw::c_void;

pub const VALIDATION_ENABLED: bool = true;
pub const VALIDATION_LAYERS: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];

pub fn check_validation_layer_support(entry: &ash::Entry) -> bool {
    let layer_properties = entry
        .enumerate_instance_layer_properties()
        .expect("Couldn't enumerate instance layer properties.");

    if layer_properties.is_empty() {
        println!("No available layers.");
        return false;
    }

    for required_layer_name in VALIDATION_LAYERS.iter() {
        let mut is_layer_found = false;

        for layer_property in layer_properties.iter() {
            let test_layer_name = utility::array_to_string(&layer_property.layer_name);
            if (*required_layer_name) == test_layer_name {
                is_layer_found = true;
                break;
            }
        }

        if !is_layer_found {
            return false;
        }
    }
    true
}

pub fn setup_debug_utils(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> (ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT) {
    let debug_utils_loader = ash::extensions::ext::DebugUtils::new(entry, instance);

    if VALIDATION_ENABLED == false {
        (debug_utils_loader, ash::vk::DebugUtilsMessengerEXT::null())
    } else {
        let messenger_create_info = debug_messenger_create_info();

        let utils_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&messenger_create_info, None)
                .expect("Debug Utils Callback")
        };

        (debug_utils_loader, utils_messenger)
    }
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "Verbose".bold().green(),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "Warning".bold().yellow(),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "Error".bold().red(),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "Info".bold().blue(),
        _ => "Unknown".into(),
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "General".underline(),
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "Performance".underline(),
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "Validation".underline(),
        vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING => "Address Binding".underline(),
        _ => "Unknown".underline(),
    };
    let message = CStr::from_ptr((*p_callback_data).p_message).to_str().unwrap();
    let message = message.trim();
    println!("{}: {}\n{}\n", types, severity, message);
    vk::FALSE
}

pub fn debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
    use vk::DebugUtilsMessageTypeFlagsEXT as Type;

    //TODO: arg in build.rs
    let severity = Severity::WARNING | Severity::ERROR | Severity::VERBOSE | Severity::INFO;
    let types = Type::GENERAL | Type::PERFORMANCE | Type::VALIDATION;

    vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(severity)
        .message_type(types)
        .pfn_user_callback(Some(vulkan_debug_utils_callback))
        .build()
}
