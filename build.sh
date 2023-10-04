
# add check for spv folder
glslc shaders/src/shader.vert -o shaders/spv/vert.vert
glslc shaders/src/shader.frag -o shaders/spv/frag.frag
cargo fmt
cargo run
