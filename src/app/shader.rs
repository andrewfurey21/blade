use anyhow::{bail, Result};
use ash::vk;

use std::io::Read;
use std::path::Path;

pub struct ShaderDetails {
    path: Box<Path>,
    entry_point: String,
    pub shader_stage: vk::ShaderStageFlags,
    code: Vec<u8>,
    pub module: vk::ShaderModule,
}

impl ShaderDetails {
    pub fn new(device: &ash::Device, file_name: &str, entry_point: &str) -> Result<ShaderDetails> {
        let entry_point = String::from(entry_point);
        let path = Path::new(file_name);
        if path.exists() {
            let shader_stage = {
                let stem = path
                    .file_stem()
                    .expect("Shader has no file stem.")
                    .to_str()
                    .unwrap();

                let shader_type = if stem == "frag" {
                    vk::ShaderStageFlags::FRAGMENT
                } else if stem == "vert" {
                    vk::ShaderStageFlags::VERTEX
                } else {
                    bail!(
                        "Shader type {} does not exist or has not been implemented.",
                        stem
                    )
                };
                shader_type
            };
            let file = std::fs::File::open(path)
                .expect(&format!("Failed to open file \"{}\"", path.display()));
            let code = file.bytes().flatten().collect::<Vec<u8>>();
            let module = ShaderDetails::create_shader_module(device, &code);
            return Ok(ShaderDetails {
                path: path.into(),
                entry_point,
                shader_stage,
                code,
                module,
            });
        } else {
            bail!("\"{}\" does not exist.", path.display());
        }
    }

    fn create_shader_module(device: &ash::Device, bytes: &Vec<u8>) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            code_size: bytes.len(),
            p_code: bytes.as_ptr() as *const u32,
            ..Default::default()
        };
        unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Couldn't create shader module.")
        }
    }

    fn read_shader_code(file_name: &str) -> Vec<u8> {
        let path = Path::new(file_name);
        let file =
            std::fs::File::open(path).expect(&format!("Failed to find spv file at {:?}", path));
        file.bytes().flatten().collect::<Vec<u8>>()
    }
}
