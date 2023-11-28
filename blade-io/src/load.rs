use blade_core::vertex;
use std::path::Path;
use tobj;

pub fn load_model(model_path: &Path) -> (Vec<vertex::Vertex>, Vec<u32>) {
    let _load_options = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    };
    let model_obj = tobj::load_obj(model_path, &tobj::LoadOptions::default())
        .expect("Failed to load model object!");

    let mut vertices = vec![];
    let mut indices = vec![];

    let (models, _) = model_obj;
    for m in models.iter() {
        let mesh = &m.mesh;

        if mesh.texcoords.len() == 0 {
            panic!("Missing texture coordinate for the model.")
        }

        let total_vertices_count = mesh.positions.len() / 3;
        for i in 0..total_vertices_count {
            let x = mesh.positions[i * 3];
            let y = mesh.positions[i * 3 + 1];
            let z = mesh.positions[i * 3 + 2];
            let vertex = vertex::Vertex::new(
                x,
                y,
                z,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                mesh.texcoords[i * 2],
                mesh.texcoords[i * 2 + 1],
            );
            vertices.push(vertex);
        }

        indices = mesh.indices.clone();
    }
    (vertices, indices)
}
