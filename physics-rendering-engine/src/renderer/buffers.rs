use anyhow::{Context, Result};
use ash::vk;

use super::context::VulkanContext;
use super::mesh;
use super::rt_pipeline::SceneUBO;
use crate::ui;

// ---------------------------------------------------------------------------
// Per-frame UBO buffer
// ---------------------------------------------------------------------------

pub(super) struct SceneUboBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped: *mut SceneUBO,
}

unsafe impl Send for SceneUboBuffer {}

pub(super) fn create_scene_ubo_buffer(context: &VulkanContext) -> Result<SceneUboBuffer> {
    let size = std::mem::size_of::<SceneUBO>() as vk::DeviceSize;
    let (buffer, memory) = mesh::create_buffer(
        &context.device,
        &context.memory_properties,
        size,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        false,
    )?;
    let mapped = unsafe {
        context.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
    }
    .context("Failed to map SceneUBO memory")? as *mut SceneUBO;
    Ok(SceneUboBuffer { buffer, memory, mapped })
}

// ---------------------------------------------------------------------------
// Mesh offsets GPU buffer
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct GpuMeshOffset {
    index_base: u32,
    vertex_base: u32,
}

pub(super) struct MeshOffsetsBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
}

pub(super) fn create_mesh_offsets_buffer(
    context: &VulkanContext,
    infos: &[mesh::SubMeshInfo],
) -> Result<MeshOffsetsBuffer> {
    let offsets: Vec<GpuMeshOffset> = infos
        .iter()
        .map(|info| GpuMeshOffset {
            index_base: info.index_offset,
            vertex_base: info.vertex_offset,
        })
        .collect();

    let (buffer, memory) = mesh::upload_via_staging(
        context,
        &offsets,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    Ok(MeshOffsetsBuffer { buffer, memory })
}

// ---------------------------------------------------------------------------
// Per-frame UI buffer (host-visible, written each frame)
// ---------------------------------------------------------------------------

pub(super) struct UiGpuBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped: *mut u8,
    pub capacity: usize, // total byte capacity
}

unsafe impl Send for UiGpuBuffer {}

pub(super) fn create_ui_buffer(context: &VulkanContext) -> Result<UiGpuBuffer> {
    let capacity = ui::UI_BUFFER_BYTES;
    let (buffer, memory) = mesh::create_buffer(
        &context.device,
        &context.memory_properties,
        capacity as vk::DeviceSize,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        false,
    )?;
    let mapped = unsafe {
        context.device.map_memory(memory, 0, capacity as vk::DeviceSize, vk::MemoryMapFlags::empty())
    }
    .context("Failed to map UI buffer memory")? as *mut u8;

    // Zero-initialise so the shader sees count=0 on the first frame.
    unsafe { std::ptr::write_bytes(mapped, 0, capacity) };

    Ok(UiGpuBuffer { buffer, memory, mapped, capacity })
}

// ---------------------------------------------------------------------------
// Per-frame point light buffer (host-visible, written each frame)
// ---------------------------------------------------------------------------

pub const MAX_POINT_LIGHTS: usize = 64;

/// GPU-side point light (std430 layout: 32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuPointLight {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
}

/// Header: u32 count + 3 padding u32s = 16 bytes, then MAX_POINT_LIGHTS lights.
pub(super) const POINT_LIGHT_BUFFER_BYTES: usize = 16 + MAX_POINT_LIGHTS * std::mem::size_of::<GpuPointLight>();

pub(super) struct PointLightBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped: *mut u8,
}

unsafe impl Send for PointLightBuffer {}

pub(super) fn create_point_light_buffer(context: &VulkanContext) -> Result<PointLightBuffer> {
    let capacity = POINT_LIGHT_BUFFER_BYTES;
    let (buffer, memory) = mesh::create_buffer(
        &context.device,
        &context.memory_properties,
        capacity as vk::DeviceSize,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        false,
    )?;
    let mapped = unsafe {
        context.device.map_memory(memory, 0, capacity as vk::DeviceSize, vk::MemoryMapFlags::empty())
    }
    .context("Failed to map point light buffer memory")? as *mut u8;

    // Zero-initialise so the shader sees count=0 on the first frame.
    unsafe { std::ptr::write_bytes(mapped, 0, capacity) };

    Ok(PointLightBuffer { buffer, memory, mapped })
}

// ---------------------------------------------------------------------------
// Storage image
// ---------------------------------------------------------------------------

pub(super) struct StorageImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
}

pub(super) fn create_storage_image(context: &VulkanContext, extent: vk::Extent2D) -> Result<StorageImage> {
    let image = unsafe {
        context.device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )
    }
    .context("Failed to create storage image")?;

    let reqs = unsafe { context.device.get_image_memory_requirements(image) };
    let memory_type = mesh::find_memory_type(
        &context.memory_properties,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .context("No suitable memory type for storage image")?;

    let memory = unsafe {
        context.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(memory_type),
            None,
        )
    }
    .context("Failed to allocate storage image memory")?;

    unsafe { context.device.bind_image_memory(image, memory, 0) }
        .context("Failed to bind storage image memory")?;

    let view = unsafe {
        context.device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            None,
        )
    }
    .context("Failed to create storage image view")?;

    Ok(StorageImage { image, view, memory })
}
