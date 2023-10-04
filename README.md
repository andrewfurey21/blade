# cad-rs

A CAD tool written in Rust

### Installed:
* libvulkan-dev
* spirv-tools
* vulkan-validationlayers-dev
* glslc
* lunar vulkan sdk
* vulkan-tools

### project structure:

- image
- application (vulkan, rendering, window, keyboard/mouse io)
- geometry manipulation code (probably includes fonts)
- gui related code
- file io (saving models, outputting models to different formats, loading models, version control)

### Features To implement:
- [ ] Image: Decode bitmap, png, jpg, svg
- [ ] Application: fast vulkan code for rendering 
- [ ] Version control: make branches, save operations instantly, scroll back through time easily
- [ ] Decent gui 

### TODO
- use glam instead of cgmath
- create build.rs instead of build.sh, properly pass in commands for shaders, ie there may be multiple shaders, with different entry points and names. Also change where there are validation layers based on whether in debug mode or release.
- shader details should be more specific about where it checks the file, extension names, maybe use spirq crate.
- put shader module in shader details and use bytemuck for Vec<u8> to Vec<u32>, and use builder struct instead.
- Update create_graphics_pipeline to have more parameters on how it should be set up. Maybe use struct to define what the graphics pipeline should look like.
- fix for coloring based on opacity in color_blend_attachment_state_create_info in create_graphics_pipeline
- fix how you setup command buffers, take a look at single time recording, seperate recording from creation in create_command_buffers
- make device_memory_properties a member of the App
- use gpu_allocator
- image module, sort out textures, fix other cases for image_data
- in find_supported format, use map or other more idiomatic functions.
- learn about and fixup vulkan_debug_utils_callback, add coloring and better formatting

