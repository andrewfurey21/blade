
# add check for spv folder
echo "Building vertex shader"
glslc shaders/src/shader.vert -o shaders/spv/vert.spv
echo "Building fragment shader"
glslc shaders/src/shader.frag -o shaders/spv/frag.spv
echo "Building project"
cargo fmt
# cargo clippy
cargo run
