# cad-rs

A CAD tool written in Rust

### Installed:
* libvulkan-dev
* spirv-tools
* vulkan-validationlayers-dev
* [lunar vulkan sdk for ubuntu](https://vulkan.lunarg.com/doc/view/latest/linux/getting_started_ubuntu.html)
* vulkan-tools

### project structure:

- image
- application (vulkan, rendering, gui, window, keyboard/mouse io)
- math/geometry manipulation code
- file io (saving models, outputting models to different formats, loading models, version control)

### Features To implement:
- [ ] Fast!
- [ ] Image: Decode bitmap, png, jpg, svg, edge detection
- [ ] Application: fast vulkan code for rendering 
- [ ] Version control: make branches, save operations instantly, scroll back through time easily
- [ ] Decent gui 

### TODO
- [ ] improve panics in readme for shaders and stuff
- [ ] fix for coloring based on opacity in color_blend_attachment_state_create_info in create_graphics_pipeline
- [ ] fix how you setup command buffers, take a look at single time recording, seperate recording from creation in create_command_buffers
- [ ] image module, sort out textures, fix other cases for image_data
- [ ] fps
- [ ] Update create_graphics_pipeline to have more parameters on how it should be set up. Maybe use struct to define what the graphics pipeline should look like.
- [ ] use gpu_allocator
- [ ] get object spinning and up right
- [ ] keyboard and mouse input
- [ ] manipulate object with moust input
- [ ] mouse appears on other side of the screen like in blender
- [ ] a couple of gui elements that affects state of application

