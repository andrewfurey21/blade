use shaderc;

use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::io::Write;
use std::path::Path;

const SHADER_DIR: &str = "./shaders/src/";
const OUTPUT_DIR: &str = "./shaders/spv/";
const OUTPUT_EXT: &str = ".spv";

fn main() {
    let profile = env::var("PROFILE").unwrap();

    let shader_compiler = shaderc::Compiler::new().expect("Couldn't create spir-v compiler.");
    let mut compiler_options = shaderc::CompileOptions::new().unwrap();
    compiler_options.set_target_spirv(shaderc::SpirvVersion::V1_4);

    let shader_dir = Path::new(SHADER_DIR);
    let output_dir = Path::new(OUTPUT_DIR);

    if !output_dir.exists() {
        println!(
            "Creating output directory for spirv files called {:?}.",
            output_dir
        );
        fs::create_dir(output_dir).expect("Couldn't create output directory.");
    }

    let shader_dir = shader_dir.read_dir().unwrap();
    if !shader_dir.is_dir() {
        panic!("The shaders directory does not exist.");
    }

    //TODO: recursively check directories for shaders
    for shader in shader_dir {
        let shader_path = shader.unwrap().path();

        let shader_type = {
            match shader_path.extension() {
                Some(extension) if extension == "vert" => shaderc::ShaderKind::Vertex,
                Some(extension) if extension == "frag" => shaderc::ShaderKind::Fragment,
                _ => {
                    panic!("Unknown shader type.");
                }
            }
        };

        let source_name = shader_path.file_name().unwrap().to_str().unwrap();
        let mut shader_file = File::open(shader_path.clone()).expect("Couldn't open file.");
        let mut source = String::from("");

        let _bytes = shader_file.read_to_string(&mut source).expect("Couldn't write file to string. Maybe not valid UTF-8?");

        let source_name_stem = shader_path.file_stem().unwrap().to_str().unwrap();

        let output_file_name = match shader_type {
            shaderc::ShaderKind::Vertex => {
                OUTPUT_DIR.to_owned() + &source_name_stem.to_owned() + &OUTPUT_EXT.to_owned()
            }
            shaderc::ShaderKind::Fragment => {
                OUTPUT_DIR.to_owned() + &source_name_stem.to_owned() + &OUTPUT_EXT.to_owned()
            }
            _ => panic!("Unimplemented shader type."),
        };

        println!("Shader source name: {}", source_name);
        let compiler_artifact = shader_compiler
            .compile_into_spirv(
                &source,
                shader_type,
                source_name,
                "main",
                Some(&compiler_options),
            )
            .unwrap();

        let mut binary_file = File::create(&output_file_name).unwrap();
        binary_file.write_all(compiler_artifact.as_binary_u8()).expect("Couldn't write.");
    }

    match profile.as_str() {
        "debug" => (),
        "release" => (),
        _ => (),
    }
}
