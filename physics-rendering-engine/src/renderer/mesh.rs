use anyhow::{Context, Result};
use ash::vk;
use glam::Vec3;

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
