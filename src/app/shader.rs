use anyhow::{bail, Result};
use ash::vk;

use std::io::Read;
use std::path::Path;

use bytemuck;

pub struct ShaderDetails {
    pub module: vk::ShaderModule,
    pub shader_stage: vk::ShaderStageFlags,
}

impl ShaderDetails {
    pub fn new(
        device: &ash::Device,
        file_name: &str,
        shader_stage: vk::ShaderStageFlags,
    ) -> Result<ShaderDetails> {
        let path = Path::new(file_name);
        if path.exists() {
            let file = std::fs::File::open(path)
                .expect(&format!("Failed to open file \"{}\"", path.display()));
            let code = file.bytes().flatten().collect::<Vec<u8>>();
            let module = ShaderDetails::create_shader_module(device, &code);

            return Ok(ShaderDetails {
                shader_stage,
                module,
            });
        } else {
            bail!("File \"{}\" does not exist.", path.display());
        }
    }

    fn create_shader_module(device: &ash::Device, bytes: &Vec<u8>) -> vk::ShaderModule {
        let code: &[u32] = bytemuck::try_cast_slice(bytes).unwrap();
        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(code);
        unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Couldn't create shader module.")
        }
    }

    #[allow(dead_code)]
    fn read_shader_code(file_name: &str) -> Vec<u8> {
        let path = Path::new(file_name);
        let file =
            std::fs::File::open(path).expect(&format!("Failed to find spv file at {:?}", path));
        file.bytes().flatten().collect::<Vec<u8>>()
    }
}
