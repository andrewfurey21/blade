
# add check for spv folder
glslc shaders/src/shader.vert -o shaders/spv/vert.spv
glslc shaders/src/shader.frag -o shaders/spv/frag.spv
cargo fmt
cargo run
