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
- math/geometry manipulation code (probably includes fonts)
- gui related code
- file io (saving models, outputting models to different formats, loading models, version control)

### Features To implement:
- [ ] Fast!
- [ ] Image: Decode bitmap, png, jpg, svg, edge detection
- [ ] Application: fast vulkan code for rendering 
- [ ] Version control: make branches, save operations instantly, scroll back through time easily
- [ ] Decent gui 

### TODO
- [ ] make device_memory_properties a member of the App
- [ ] in find_supported format, use map or other more idiomatic functions.
- [ ] learn about and fixup vulkan_debug_utils_callback, add coloring and better formatting
- [ ] fix paths for shaders
- [ ] shader details should be more specific about where it checks the file, extension names, maybe use spirq crate.
- [ ] put shader module in shader details and use bytemuck for Vec<u8> to Vec<u32>, and use builder struct instead.
- [ ] create build.rs instead of build.sh, properly pass in commands for shaders, ie there may be multiple shaders, with different entry points and names. Also change where there are validation layers based on whether in debug mode or release.
- [ ] fix window minimization
- [ ] use gpu_allocator
- [ ] use glam instead of cgmath
- [ ] fps
- [ ] fix for coloring based on opacity in color_blend_attachment_state_create_info in create_graphics_pipeline
- [ ] Update create_graphics_pipeline to have more parameters on how it should be set up. Maybe use struct to define what the graphics pipeline should look like.
- [ ] fix how you setup command buffers, take a look at single time recording, seperate recording from creation in create_command_buffers
- [ ] image module, sort out textures, fix other cases for image_data
- [ ] get object spinning and up right
- [ ] keyboard and mouse input
- [ ] manipulate object with moust input
- [ ] mouse appears on other side of the screen like in blender
- [ ] a couple of gui elements that affects state of application

