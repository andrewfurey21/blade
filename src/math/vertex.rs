use ash::vk;
use glam;

use memoffset::offset_of;

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct Vertex {
    pub pos: glam::Vec4,
    pub tex_coord: glam::Vec2,
    pub color: glam::Vec4,
}

impl Vertex {
    pub const fn new(
        x: f32,
        y: f32,
        z: f32,
        w: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        tx: f32,
        ty: f32,
    ) -> Self {
        let pos = glam::Vec4::new(x, y, z, w);
        let color = glam::Vec4::new(r, g, b, a);
        let tex_coord = glam::Vec2::new(tx, ty);
        Vertex {
            pos,
            color,
            tex_coord,
        }
    }

    pub fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Self, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Self, color) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Self, tex_coord) as u32,
            },
        ]
    }
}
