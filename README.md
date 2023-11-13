# blade 

A CAD tool written in Rust

### Project Structure:

- blade-core: math + geometry manipulation code
- blade-app: rendering, gui, window, keyboard/mouse io
- blade-io: saving models, outputting models to different formats, loading models, version control
- blade-image: Image related functions, encoding/decoding images (bitmap, png, jpg, svg), edge detection, rendering (raytracing?)

### Goals:
The goal of this project is to make a fast CAD kernel, using the GPU when possible, and as much multithreading as possible.

### Tools Installed (ubuntu):
* libvulkan-dev
* spirv-tools
* vulkan-validationlayers-dev
* [lunar vulkan sdk for ubuntu](https://vulkan.lunarg.com/doc/view/latest/linux/getting_started_ubuntu.html)
* vulkan-tools

### Current To Do List
- [ ] fix how you setup command buffers, take a look at single time recording, seperate recording from creation in create_command_buffers
- [ ] fix image module, sort out textures, fix other cases for image_data
- [ ] fps
- [ ] use gpu_allocator
- [ ] get object spinning and up right
- [ ] keyboard and mouse input
- [ ] manipulate object with moust input
- [ ] mouse appears on other side of the screen like in blender
- [ ] a couple of gui elements that affects state of application

