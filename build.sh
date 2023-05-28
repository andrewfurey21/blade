echo "Building vertex shader"
glslc shaders/shader.vert -o shaders/vert.spv
echo "Building fragment shader"
glslc shaders/shader.frag -o shaders/frag.spv
cd cad-rs
echo "Building project"
cargo run
