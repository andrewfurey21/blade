# blade 

A CAD tool written in Rust

### Project Structure:

- blade-image: Image related functions, decoding images (bitmap, png, jpg, svg), edge detection
- blade-app (vulkan, rendering, gui, window, keyboard/mouse io)
- blade-core: math + geometry manipulation code
- blade-io (saving models, outputting models to different formats, loading models)
- blade-vc: version control

### Goals:
The goal of this project is to make a fast CAD kernel, using the GPU when possible.

### Tools Installed (ubuntu):
* libvulkan-dev
* spirv-tools
* vulkan-validationlayers-dev
* [lunar vulkan sdk for ubuntu](https://vulkan.lunarg.com/doc/view/latest/linux/getting_started_ubuntu.html)
* vulkan-tools

### Current To Do List
- [ ] recursively compile shaders in shaders/
- [ ] fix release mode errors
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

