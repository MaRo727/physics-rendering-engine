use anyhow::{Context, Result};
use ash::vk;
use glam::Vec3;
use std::f32::consts::PI;

use super::context::VulkanContext;

// ---------------------------------------------------------------------------
// Vertex layout
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
}

impl Vertex {
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(12),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(24),
        ]
    }
}

// ---------------------------------------------------------------------------
// Mesh (GPU buffers)
// ---------------------------------------------------------------------------

pub struct Mesh {
    pub vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    pub index_count: u32,
    pub vertex_count: u32,
}

impl Mesh {
    pub fn new(context: &VulkanContext, vertices: &[Vertex], indices: &[u32]) -> Result<Self> {
        let rt_flags = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        let (vertex_buffer, vertex_memory) = upload_via_staging(
            context,
            vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER | rt_flags,
        )?;

        let (index_buffer, index_memory) = upload_via_staging(
            context,
            indices,
            vk::BufferUsageFlags::INDEX_BUFFER | rt_flags,
        )?;

        Ok(Self {
            vertex_buffer,
            vertex_memory,
            index_buffer,
            index_memory,
            index_count: indices.len() as u32,
            vertex_count: vertices.len() as u32,
        })
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_memory, None);
            device.destroy_buffer(self.index_buffer, None);
            device.free_memory(self.index_memory, None);
        }
    }
}

pub fn get_device_address(device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceAddress {
    unsafe {
        device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::default().buffer(buffer),
        )
    }
}

// ---------------------------------------------------------------------------
// Cube mesh data
// ---------------------------------------------------------------------------

/// Unit cube centered at origin.
/// 24 vertices (4 per face) so each face carries a correct outward normal.
/// All faces are gray; diffuse lighting in the fragment shader provides shading.
pub fn cube() -> (Vec<Vertex>, Vec<u32>) {
    let g = Vec3::new(0.85, 0.65, 0.45);

    #[rustfmt::skip]
    let vertices = vec![
        // Front (+Z)
        Vertex { position: Vec3::new(-0.5, -0.5,  0.5), normal:  Vec3::Z,     color: g },
        Vertex { position: Vec3::new( 0.5, -0.5,  0.5), normal:  Vec3::Z,     color: g },
        Vertex { position: Vec3::new( 0.5,  0.5,  0.5), normal:  Vec3::Z,     color: g },
        Vertex { position: Vec3::new(-0.5,  0.5,  0.5), normal:  Vec3::Z,     color: g },
        // Back (-Z)
        Vertex { position: Vec3::new( 0.5, -0.5, -0.5), normal: Vec3::NEG_Z,  color: g },
        Vertex { position: Vec3::new(-0.5, -0.5, -0.5), normal: Vec3::NEG_Z,  color: g },
        Vertex { position: Vec3::new(-0.5,  0.5, -0.5), normal: Vec3::NEG_Z,  color: g },
        Vertex { position: Vec3::new( 0.5,  0.5, -0.5), normal: Vec3::NEG_Z,  color: g },
        // Left (-X)
        Vertex { position: Vec3::new(-0.5, -0.5, -0.5), normal: Vec3::NEG_X,  color: g },
        Vertex { position: Vec3::new(-0.5, -0.5,  0.5), normal: Vec3::NEG_X,  color: g },
        Vertex { position: Vec3::new(-0.5,  0.5,  0.5), normal: Vec3::NEG_X,  color: g },
        Vertex { position: Vec3::new(-0.5,  0.5, -0.5), normal: Vec3::NEG_X,  color: g },
        // Right (+X)
        Vertex { position: Vec3::new( 0.5, -0.5,  0.5), normal:  Vec3::X,     color: g },
        Vertex { position: Vec3::new( 0.5, -0.5, -0.5), normal:  Vec3::X,     color: g },
        Vertex { position: Vec3::new( 0.5,  0.5, -0.5), normal:  Vec3::X,     color: g },
        Vertex { position: Vec3::new( 0.5,  0.5,  0.5), normal:  Vec3::X,     color: g },
        // Top (+Y)
        Vertex { position: Vec3::new(-0.5,  0.5,  0.5), normal:  Vec3::Y,     color: g },
        Vertex { position: Vec3::new( 0.5,  0.5,  0.5), normal:  Vec3::Y,     color: g },
        Vertex { position: Vec3::new( 0.5,  0.5, -0.5), normal:  Vec3::Y,     color: g },
        Vertex { position: Vec3::new(-0.5,  0.5, -0.5), normal:  Vec3::Y,     color: g },
        // Bottom (-Y)
        Vertex { position: Vec3::new(-0.5, -0.5, -0.5), normal: Vec3::NEG_Y,  color: g },
        Vertex { position: Vec3::new( 0.5, -0.5, -0.5), normal: Vec3::NEG_Y,  color: g },
        Vertex { position: Vec3::new( 0.5, -0.5,  0.5), normal: Vec3::NEG_Y,  color: g },
        Vertex { position: Vec3::new(-0.5, -0.5,  0.5), normal: Vec3::NEG_Y,  color: g },
    ];

    // Each face: two triangles (0,1,2) and (0,2,3) into its 4 vertices.
    let indices: Vec<u32> = (0..6u32)
        .flat_map(|f| {
            let b = f * 4;
            [b, b + 1, b + 2, b, b + 2, b + 3]
        })
        .collect();

    (vertices, indices)
}

// ---------------------------------------------------------------------------
// Ball (UV sphere) mesh data
// ---------------------------------------------------------------------------

/// UV sphere centered at origin with radius 0.5.
pub fn ball(stacks: u32, slices: u32) -> (Vec<Vertex>, Vec<u32>) {
    let color = Vec3::new(0.3, 0.6, 0.85);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=stacks {
        let phi = PI * i as f32 / stacks as f32;
        let (sin_phi, cos_phi) = phi.sin_cos();
        for j in 0..=slices {
            let theta = 2.0 * PI * j as f32 / slices as f32;
            let (sin_theta, cos_theta) = theta.sin_cos();

            let normal = Vec3::new(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
            vertices.push(Vertex {
                position: normal * 0.5,
                normal,
                color,
            });
        }
    }

    for i in 0..stacks {
        for j in 0..slices {
            let row0 = i * (slices + 1) + j;
            let row1 = (i + 1) * (slices + 1) + j;
            indices.extend_from_slice(&[row0, row1, row0 + 1]);
            indices.extend_from_slice(&[row0 + 1, row1, row1 + 1]);
        }
    }

    (vertices, indices)
}

// ---------------------------------------------------------------------------
// Pyramid mesh data
// ---------------------------------------------------------------------------

/// Square-base pyramid centered at origin, height 1 (base at y=-0.5, apex at y=0.5).
pub fn pyramid() -> (Vec<Vertex>, Vec<u32>) {
    let color = Vec3::new(0.85, 0.75, 0.3);
    let apex = Vec3::new(0.0, 0.5, 0.0);
    let bl = Vec3::new(-0.5, -0.5, 0.5);
    let br = Vec3::new(0.5, -0.5, 0.5);
    let fr = Vec3::new(0.5, -0.5, -0.5);
    let fl = Vec3::new(-0.5, -0.5, -0.5);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Helper to add a triangular face with computed normal.
    let mut add_tri = |a: Vec3, b: Vec3, c: Vec3| {
        let normal = (b - a).cross(c - a).normalize();
        let base = vertices.len() as u32;
        vertices.push(Vertex { position: a, normal, color });
        vertices.push(Vertex { position: b, normal, color });
        vertices.push(Vertex { position: c, normal, color });
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    };

    // Four side faces.
    add_tri(apex, bl, br); // front
    add_tri(apex, br, fr); // right
    add_tri(apex, fr, fl); // back
    add_tri(apex, fl, bl); // left

    // Bottom face (two triangles).
    let normal = Vec3::NEG_Y;
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: bl, normal, color });
    vertices.push(Vertex { position: fr, normal, color });
    vertices.push(Vertex { position: br, normal, color });
    vertices.push(Vertex { position: fl, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2]);
    indices.extend_from_slice(&[base, base + 3, base + 1]);

    (vertices, indices)
}

// ---------------------------------------------------------------------------
// Mesh building helpers
// ---------------------------------------------------------------------------

fn push_tri(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    a: Vec3, b: Vec3, c: Vec3,
    color: Vec3,
) {
    let normal = (b - a).cross(c - a).normalize();
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: a, normal, color });
    vertices.push(Vertex { position: b, normal, color });
    vertices.push(Vertex { position: c, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2]);
}

fn push_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    a: Vec3, b: Vec3, c: Vec3, d: Vec3,
    color: Vec3,
) {
    let normal = (b - a).cross(d - a).normalize();
    let base = vertices.len() as u32;
    vertices.push(Vertex { position: a, normal, color });
    vertices.push(Vertex { position: b, normal, color });
    vertices.push(Vertex { position: c, normal, color });
    vertices.push(Vertex { position: d, normal, color });
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

// ---------------------------------------------------------------------------
// Triangular prism (wedge) mesh data
// ---------------------------------------------------------------------------

/// Triangular prism (wedge) centered at origin, depth 1 along Z.
/// The triangular cross-section is in the XY plane.
pub fn triangle_prism() -> (Vec<Vertex>, Vec<u32>) {
    let c = Vec3::new(0.4, 0.8, 0.45);
    let tl = Vec3::new(-0.5, -0.5, 0.0);
    let tr = Vec3::new(0.5, -0.5, 0.0);
    let top = Vec3::new(0.0, 0.5, 0.0);

    let mut v = Vec::new();
    let mut i = Vec::new();

    let f = Vec3::new(0.0, 0.0, 0.5);
    let b = Vec3::new(0.0, 0.0, -0.5);

    push_tri(&mut v, &mut i, tl + f, tr + f, top + f, c);
    push_tri(&mut v, &mut i, tr + b, tl + b, top + b, c);
    push_quad(&mut v, &mut i, tl + b, tr + b, tr + f, tl + f, c);
    push_quad(&mut v, &mut i, tl + f, top + f, top + b, tl + b, c);
    push_quad(&mut v, &mut i, tr + b, top + b, top + f, tr + f, c);

    (v, i)
}

// ---------------------------------------------------------------------------
// Slope (ramp) mesh data
// ---------------------------------------------------------------------------

/// Right-angle slope/ramp centered at origin.
/// Flat bottom, vertical back face, slanted top surface.
pub fn slope() -> (Vec<Vertex>, Vec<u32>) {
    let c = Vec3::new(0.7, 0.4, 0.6);
    let bl_f = Vec3::new(-0.5, -0.5, 0.5);
    let br_f = Vec3::new(0.5, -0.5, 0.5);
    let tl_f = Vec3::new(-0.5, 0.5, 0.5);
    let bl_b = Vec3::new(-0.5, -0.5, -0.5);
    let br_b = Vec3::new(0.5, -0.5, -0.5);
    let tl_b = Vec3::new(-0.5, 0.5, -0.5);

    let mut v = Vec::new();
    let mut i = Vec::new();

    push_quad(&mut v, &mut i, bl_b, br_b, br_f, bl_f, c);
    push_quad(&mut v, &mut i, bl_f, tl_f, tl_b, bl_b, c);
    push_quad(&mut v, &mut i, tl_f, br_f, br_b, tl_b, c);
    push_tri(&mut v, &mut i, br_f, tl_f, bl_f, c);
    push_tri(&mut v, &mut i, bl_b, tl_b, br_b, c);

    (v, i)
}

// ---------------------------------------------------------------------------
// Combined mesh: merges multiple sub-meshes into one buffer
// ---------------------------------------------------------------------------

/// Describes a sub-mesh within the combined vertex/index buffers.
pub struct SubMeshInfo {
    pub vertex_offset: u32, // in vertices
    pub index_offset: u32,  // in indices
    pub vertex_count: u32,
    pub index_count: u32,
}

/// Combines multiple mesh data sets into one vertex and index buffer.
/// Returns per-sub-mesh offset info.
pub fn combine_meshes(
    meshes: &[(Vec<Vertex>, Vec<u32>)],
) -> (Vec<Vertex>, Vec<u32>, Vec<SubMeshInfo>) {
    let mut combined_verts = Vec::new();
    let mut combined_indices = Vec::new();
    let mut infos = Vec::new();

    for (verts, idxs) in meshes {
        let vertex_offset = combined_verts.len() as u32;
        let index_offset = combined_indices.len() as u32;

        combined_verts.extend_from_slice(verts);
        // Indices stay 0-based within each sub-mesh (relative to vertex_offset).
        combined_indices.extend_from_slice(idxs);

        infos.push(SubMeshInfo {
            vertex_offset,
            index_offset,
            vertex_count: verts.len() as u32,
            index_count: idxs.len() as u32,
        });
    }

    (combined_verts, combined_indices, infos)
}

// ---------------------------------------------------------------------------
// Buffer allocation helpers
// ---------------------------------------------------------------------------

/// Upload `data` to a DEVICE_LOCAL buffer via a short-lived staging buffer.
pub fn upload_via_staging<T: Copy>(
    context: &VulkanContext,
    data: &[T],
    dst_usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let size = std::mem::size_of_val(data) as vk::DeviceSize;

    // Host-visible staging buffer.
    let (staging, staging_mem) = create_buffer(
        &context.device,
        &context.memory_properties,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        false,
    )?;

    unsafe {
        let ptr = context
            .device
            .map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())
            .context("Failed to map staging memory")? as *mut T;
        ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        context.device.unmap_memory(staging_mem);
    }

    // Device-local destination buffer — needs DEVICE_ADDRESS flag if dst_usage includes it.
    let needs_device_address = dst_usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
    let (dst, dst_mem) = create_buffer(
        &context.device,
        &context.memory_properties,
        size,
        dst_usage | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        needs_device_address,
    )?;

    // One-time command buffer to copy staging → device-local.
    let pool = unsafe {
        context.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(context.graphics_queue_family)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
            None,
        )
    }
    .context("Failed to create transient command pool")?;

    let cb = unsafe {
        context.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .context("Failed to allocate copy command buffer")?[0];

    unsafe {
        context.device.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        context.device.cmd_copy_buffer(
            cb,
            staging,
            dst,
            &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }],
        );
        context.device.end_command_buffer(cb)?;

        let fence = context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .context("Failed to create upload fence")?;
        context.device.queue_submit(
            context.graphics_queue,
            &[vk::SubmitInfo::default().command_buffers(&[cb])],
            fence,
        )?;
        context.device.wait_for_fences(&[fence], true, u64::MAX)?;
        context.device.destroy_fence(fence, None);
        context.device.destroy_command_pool(pool, None);
    }

    // Staging buffer no longer needed.
    unsafe {
        context.device.destroy_buffer(staging, None);
        context.device.free_memory(staging_mem, None);
    }

    Ok((dst, dst_mem))
}

pub fn create_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    required_flags: vk::MemoryPropertyFlags,
    device_address: bool,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .context("Failed to create buffer")?;

    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type = find_memory_type(mem_props, reqs.memory_type_bits, required_flags)
        .context("No suitable memory type for buffer")?;

    let mut alloc_flags_info = vk::MemoryAllocateFlagsInfo::default()
        .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(memory_type);

    let memory = if device_address {
        unsafe { device.allocate_memory(&alloc_info.push_next(&mut alloc_flags_info), None) }
    } else {
        unsafe { device.allocate_memory(&alloc_info, None) }
    }
    .context("Failed to allocate buffer memory")?;

    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.context("Failed to bind buffer memory")?;

    Ok((buffer, memory))
}

pub fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        type_filter & (1 << i) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(required)
    })
}
