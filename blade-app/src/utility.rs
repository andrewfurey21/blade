use std::ffi::{c_char, CStr};

pub fn array_to_string(array: &[c_char]) -> &str {
    let raw_string = unsafe { CStr::from_ptr(array.as_ptr()) };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
}
