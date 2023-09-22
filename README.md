# cad-rs

A CAD tool written in Rust

Installed:
* libvulkan-dev
* spirv-tools
* vulkan-validationlayers-dev
* glslc
* lunar vulkan sdk
* vulkan-tools

project structure:

- image
- application (vulkan, rendering, window, io)
- geometry manipulation code (probably includes fonts)
- gui related code
- lua macros
- file io (saving models, outputting models to different formats, loading models, version control)

Features To implement:
- [ ] Image: Decode bitmap, png, jpg
- [ ] Application: fast vulkan code for rendering 
- [ ] Lua (maybe python instead?) macros (implement operations on the fly without recompiling the application)
- [ ] Version control: make branches, save operations instantly, scroll back through time easily
- [ ] Decent gui and baked in operations

