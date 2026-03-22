pub mod acceleration_structure;
pub mod context;
pub mod frame;
pub mod mesh;
pub mod rt_pipeline;
pub mod shapes;
pub mod swapchain;

use anyhow::{Context, Result};
use ash::vk;
use glam::{Mat4, Vec4};
use std::sync::Arc;
use winit::window::Window;

use acceleration_structure::{Blas, Tlas, mat4_to_transform};
use context::VulkanContext;
use frame::{create_frame, FrameData, MAX_FRAMES_IN_FLIGHT};
use mesh::Mesh;
use rt_pipeline::{RtPipeline, SceneUBO};
use swapchain::Swapchain;

// ---------------------------------------------------------------------------
// Mesh type constants
// ---------------------------------------------------------------------------

pub const MESH_CUBE: u32 = 0;
pub const MESH_BALL: u32 = 1;
pub const MESH_PYRAMID: u32 = 2;
pub const MESH_TRIANGLE: u32 = 3;
pub const MESH_SLOPE: u32 = 4;
pub const MESH_CAPSULE: u32 = 5;
pub const MESH_WATER: u32 = 6;
pub const MESH_ROCK: u32 = 7;
pub const MESH_TREE_OAK: u32 = 8;
pub const MESH_TREE_PINE: u32 = 9;
pub const MESH_TREE_DEAD: u32 = 10;
pub const MESH_TREE_OAK_LOD: u32 = 11;
pub const MESH_TREE_PINE_LOD: u32 = 12;
pub const MESH_TREE_DEAD_LOD: u32 = 13;
pub const MESH_TERRAIN_BASE: u32 = 14;
const SHAPE_MESH_COUNT: usize = 14;

/// Pre-allocated capacity for the building mesh slot in the combined buffer.
const BUILDING_INITIAL_VERTS: u32 = 65536;
const BUILDING_INITIAL_INDICES: u32 = 98304;

/// Pack mesh_type (upper 8 bits) and object_id (lower 16 bits) into 24-bit custom index.
pub fn pack_instance_id(mesh_type: u32, object_id: u32) -> u32 {
    (mesh_type << 16) | (object_id & 0xFFFF)
}

// ---------------------------------------------------------------------------
// Per-frame UBO buffer
// ---------------------------------------------------------------------------

struct SceneUboBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut SceneUBO,
}

unsafe impl Send for SceneUboBuffer {}

fn create_scene_ubo_buffer(context: &VulkanContext) -> Result<SceneUboBuffer> {
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
struct GpuMeshOffset {
    index_base: u32,
    vertex_base: u32,
}

struct MeshOffsetsBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

fn create_mesh_offsets_buffer(
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
// Storage image
// ---------------------------------------------------------------------------

struct StorageImage {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
}

fn create_storage_image(context: &VulkanContext, extent: vk::Extent2D) -> Result<StorageImage> {
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

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

pub struct Renderer {
    context: VulkanContext,
    swapchain: Swapchain,
    frames: [FrameData; MAX_FRAMES_IN_FLIGHT],
    mesh: Mesh,
    base_mesh_data: Vec<(Vec<mesh::Vertex>, Vec<u32>)>,
    base_mesh_count: usize,
    sub_mesh_infos: Vec<mesh::SubMeshInfo>,
    building_data: Option<(Vec<mesh::Vertex>, Vec<u32>)>,
    /// Allocated capacity (verts, indices) for the building slot in the combined buffer.
    building_capacity: Option<(u32, u32)>,
    blas_list: Vec<Blas>,
    tlas: Tlas,
    rt_pipeline: RtPipeline,
    storage_image: StorageImage,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    scene_ubo_buffers: [SceneUboBuffer; MAX_FRAMES_IN_FLIGHT],
    mesh_offsets_buf: MeshOffsetsBuffer,
    current_frame: usize,
    extent: vk::Extent2D,
    surface_width: u32,
    surface_height: u32,
    swapchain_dirty: bool,
}

impl Renderer {
    pub fn new(
        window: &Arc<Window>,
        max_instances: u32,
        terrain_chunks: Vec<(Vec<mesh::Vertex>, Vec<u32>)>,
    ) -> Result<Self> {
        let size = window.inner_size();
        let context = VulkanContext::new(window.as_ref())?;

        let swapchain = Swapchain::new(&context, size.width, size.height)?;
        let extent = swapchain.extent;

        let frames = [
            create_frame(&context.device, context.graphics_queue_family)?,
            create_frame(&context.device, context.graphics_queue_family)?,
        ];

        // Generate all mesh types and combine into single buffers.
        let mut base_mesh_data = vec![
            shapes::cube(),               // MESH_CUBE = 0
            shapes::ball(16, 24),         // MESH_BALL = 1
            shapes::pyramid(),            // MESH_PYRAMID = 2
            shapes::triangle_prism(),     // MESH_TRIANGLE = 3
            shapes::slope(),              // MESH_SLOPE = 4
            shapes::capsule(0.5, 1.0, 12, 16), // MESH_CAPSULE = 5
            shapes::water_plane(),               // MESH_WATER = 6
            shapes::rock_chunk(),                // MESH_ROCK = 7
            shapes::tree_oak(),                  // MESH_TREE_OAK = 8
            shapes::tree_pine(),                 // MESH_TREE_PINE = 9
            shapes::tree_dead(),                 // MESH_TREE_DEAD = 10
            shapes::tree_oak_lod(),              // MESH_TREE_OAK_LOD = 11
            shapes::tree_pine_lod(),             // MESH_TREE_PINE_LOD = 12
            shapes::tree_dead_lod(),             // MESH_TREE_DEAD_LOD = 13
        ];
        // Terrain chunks follow the shape meshes.
        base_mesh_data.extend(terrain_chunks);
        let base_mesh_count = base_mesh_data.len();

        let (mut combined_verts, mut combined_indices, mut sub_mesh_infos) =
            mesh::combine_meshes(&base_mesh_data);

        // Pre-allocate a building slot at the end of the combined buffer.
        // This avoids a full rebuild when the first block is placed.
        let building_vert_offset = combined_verts.len() as u32;
        let building_idx_offset = combined_indices.len() as u32;
        sub_mesh_infos.push(mesh::SubMeshInfo {
            vertex_offset: building_vert_offset,
            index_offset: building_idx_offset,
            vertex_count: 0,
            index_count: 0,
        });
        // Extend the buffer with zeroed placeholder data for the building slot.
        combined_verts.resize(
            combined_verts.len() + BUILDING_INITIAL_VERTS as usize,
            mesh::Vertex { position: glam::Vec3::ZERO, normal: glam::Vec3::ZERO, color: glam::Vec3::ZERO },
        );
        combined_indices.resize(
            combined_indices.len() + BUILDING_INITIAL_INDICES as usize,
            0,
        );

        let combined_mesh = Mesh::new(&context, &combined_verts, &combined_indices)?;

        // Build one BLAS per base mesh type (not the empty building slot).
        let mut blas_list = Vec::with_capacity(base_mesh_count);
        for info in &sub_mesh_infos[..base_mesh_count] {
            blas_list.push(Blas::from_range(&context, &combined_mesh, info)?);
        }

        let tlas = Tlas::new(&context, max_instances)?;

        let rt_pipeline = RtPipeline::new(&context)?;

        let storage_image = create_storage_image(&context, extent)?;

        // Transition storage image to GENERAL layout once.
        acceleration_structure::one_shot(&context, |cb| {
            image_barrier(
                &context.device,
                cb,
                storage_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );
        })?;

        let ubo0 = create_scene_ubo_buffer(&context)?;
        let ubo1 = create_scene_ubo_buffer(&context)?;
        let scene_ubo_buffers = [ubo0, ubo1];

        let mesh_offsets_buf = create_mesh_offsets_buffer(&context, &sub_mesh_infos)?;

        // Descriptor pool.
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * 3),
        ];
        let descriptor_pool = unsafe {
            context.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .context("Failed to create descriptor pool")?;

        let layouts = [rt_pipeline.descriptor_set_layout; MAX_FRAMES_IN_FLIGHT];
        let sets = unsafe {
            context.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
        }
        .context("Failed to allocate descriptor sets")?;
        let descriptor_sets = [sets[0], sets[1]];

        let renderer = Self {
            context,
            swapchain,
            frames,
            mesh: combined_mesh,
            base_mesh_data,
            base_mesh_count,
            sub_mesh_infos,
            building_data: None,
            building_capacity: Some((BUILDING_INITIAL_VERTS, BUILDING_INITIAL_INDICES)),
            blas_list,
            tlas,
            rt_pipeline,
            storage_image,
            descriptor_pool,
            descriptor_sets,
            scene_ubo_buffers,
            mesh_offsets_buf,
            current_frame: 0,
            extent,
            surface_width: size.width,
            surface_height: size.height,
            swapchain_dirty: false,
        };

        // Write initial descriptor sets.
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            renderer.write_descriptor_set(i);
        }

        Ok(renderer)
    }

    fn write_descriptor_set(&self, i: usize) {
        let tlas_handle = self.tlas.handle;
        let mut write_as = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(std::slice::from_ref(&tlas_handle));

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(self.storage_image.view)
            .image_layout(vk::ImageLayout::GENERAL);

        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(self.scene_ubo_buffers[i].buffer)
            .offset(0)
            .range(std::mem::size_of::<SceneUBO>() as vk::DeviceSize);

        let vertex_info = vk::DescriptorBufferInfo::default()
            .buffer(self.mesh.vertex_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);

        let index_info = vk::DescriptorBufferInfo::default()
            .buffer(self.mesh.index_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);

        let mesh_offset_info = vk::DescriptorBufferInfo::default()
            .buffer(self.mesh_offsets_buf.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);

        let set = self.descriptor_sets[i];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .push_next(&mut write_as),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&vertex_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&index_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&mesh_offset_info)),
        ];

        unsafe { self.context.device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_width = width;
        self.surface_height = height;
        self.swapchain_dirty = true;
    }

    fn recreate_swapchain_and_storage(&mut self) -> Result<()> {
        unsafe { self.context.device.device_wait_idle()? };

        self.swapchain.recreate(&self.context, self.surface_width, self.surface_height)?;
        self.extent = self.swapchain.extent;

        // Recreate storage image at new size.
        unsafe {
            self.context.device.destroy_image_view(self.storage_image.view, None);
            self.context.device.destroy_image(self.storage_image.image, None);
            self.context.device.free_memory(self.storage_image.memory, None);
        }
        self.storage_image = create_storage_image(&self.context, self.extent)?;

        // Transition new storage image to GENERAL.
        acceleration_structure::one_shot(&self.context, |cb| {
            image_barrier(
                &self.context.device,
                cb,
                self.storage_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );
        })?;

        // Re-write descriptor sets with the new storage image view.
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            self.write_descriptor_set(i);
        }

        Ok(())
    }

    /// Update the building mesh on the GPU.
    /// Fast path: if the new mesh fits in the existing allocation, only the
    /// building region is copied and its BLAS rebuilt — no full buffer recreate.
    /// Slow path (new buildings or growth): full combined mesh rebuild.
    pub fn update_building_mesh(
        &mut self,
        building_verts: &[mesh::Vertex],
        building_indices: &[u32],
    ) -> Result<()> {
        if building_verts.is_empty() {
            if self.building_data.is_some() {
                // All buildings removed — remove building BLAS but keep the
                // pre-allocated slot so future placements still use the fast path.
                unsafe { self.context.device.device_wait_idle()? };
                if self.blas_list.len() > self.base_mesh_count {
                    self.blas_list.pop().unwrap().destroy(&self.context);
                }
                self.sub_mesh_infos[self.base_mesh_count].vertex_count = 0;
                self.sub_mesh_infos[self.base_mesh_count].index_count = 0;
                self.building_data = None;
            }
            return Ok(());
        }

        let new_vert_count = building_verts.len() as u32;
        let new_idx_count = building_indices.len() as u32;

        // Fast path: fits within existing allocation.
        if let Some((vert_cap, idx_cap)) = self.building_capacity {
            if new_vert_count <= vert_cap && new_idx_count <= idx_cap {
                unsafe { self.context.device.device_wait_idle()? };

                let info = &self.sub_mesh_infos[self.base_mesh_count];
                self.mesh.update_region(&self.context, info, building_verts, building_indices)?;

                // Update the sub_mesh_info counts (offsets stay the same).
                self.sub_mesh_infos[self.base_mesh_count].vertex_count = new_vert_count;
                self.sub_mesh_infos[self.base_mesh_count].index_count = new_idx_count;

                // Rebuild only the building BLAS.
                if self.blas_list.len() > self.base_mesh_count {
                    self.blas_list.pop().unwrap().destroy(&self.context);
                }
                let building_info = &self.sub_mesh_infos[self.base_mesh_count];
                let building_blas =
                    Blas::from_range(&self.context, &self.mesh, building_info)?;
                self.blas_list.push(building_blas);

                self.building_data =
                    Some((building_verts.to_vec(), building_indices.to_vec()));
                return Ok(());
            }
        }

        // Slow path: first building or mesh grew — full rebuild.
        self.building_data = Some((building_verts.to_vec(), building_indices.to_vec()));
        self.rebuild_combined_mesh(&[])
    }

    /// Update terrain chunk mesh data with partial GPU buffer writes.
    /// Batches all chunk copies into a single command buffer, then rebuilds only dirty BLASes.
    pub fn update_terrain_chunks(
        &mut self,
        updates: &[(usize, Vec<mesh::Vertex>, Vec<u32>)],
    ) -> Result<()> {
        unsafe { self.context.device.device_wait_idle()? };

        // Batch all chunk copies into one command submission.
        let batch: Vec<(&mesh::SubMeshInfo, &[mesh::Vertex], &[u32])> = updates
            .iter()
            .map(|(chunk_idx, verts, indices)| {
                let mesh_idx = SHAPE_MESH_COUNT + *chunk_idx;
                (&self.sub_mesh_infos[mesh_idx], verts.as_slice(), indices.as_slice())
            })
            .collect();
        self.mesh.update_regions_batched(&self.context, &batch)?;

        // Rebuild only dirty BLASes and keep CPU data in sync.
        for (chunk_idx, verts, indices) in updates {
            let mesh_idx = SHAPE_MESH_COUNT + *chunk_idx;
            self.blas_list[mesh_idx].destroy(&self.context);
            self.blas_list[mesh_idx] =
                Blas::from_range(&self.context, &self.mesh, &self.sub_mesh_infos[mesh_idx])?;
            self.base_mesh_data[mesh_idx] = (verts.clone(), indices.clone());
        }

        Ok(())
    }

    /// Rebuild the combined vertex/index buffer from base_mesh_data + building_data.
    /// Rebuilds BLASes listed in `rebuild_blas_indices`, plus always rebuilds the
    /// building BLAS if building data exists.
    fn rebuild_combined_mesh(&mut self, rebuild_blas_indices: &[usize]) -> Result<()> {
        unsafe { self.context.device.device_wait_idle()? };

        // Remove old building BLAS if it exists.
        if self.blas_list.len() > self.base_mesh_count {
            self.blas_list.pop().unwrap().destroy(&self.context);
        }

        // Build combined mesh directly from references — no cloning base_mesh_data.
        let total_verts: usize = self.base_mesh_data.iter().map(|(v, _)| v.len()).sum();
        let total_indices: usize = self.base_mesh_data.iter().map(|(_, i)| i.len()).sum();
        let extra_verts = self.building_data.as_ref().map_or(0, |bd| bd.0.len());
        let extra_indices = self.building_data.as_ref().map_or(0, |bd| bd.1.len());
        // Allocate 2x headroom for building so future growth uses the fast path.
        let building_vert_cap = (extra_verts * 2).max(BUILDING_INITIAL_VERTS as usize);
        let building_idx_cap = (extra_indices * 2).max(BUILDING_INITIAL_INDICES as usize);

        let mut combined_verts = Vec::with_capacity(total_verts + building_vert_cap);
        let mut combined_indices = Vec::with_capacity(total_indices + building_idx_cap);
        // Always include the building slot entry.
        let mut sub_mesh_infos = Vec::with_capacity(self.base_mesh_data.len() + 1);

        for (verts, idxs) in self.base_mesh_data.iter() {
            sub_mesh_infos.push(mesh::SubMeshInfo {
                vertex_offset: combined_verts.len() as u32,
                index_offset: combined_indices.len() as u32,
                vertex_count: verts.len() as u32,
                index_count: idxs.len() as u32,
            });
            combined_verts.extend_from_slice(verts);
            combined_indices.extend_from_slice(idxs);
        }

        // Building slot — always present, with pre-allocated capacity.
        let building_vert_offset = combined_verts.len() as u32;
        let building_idx_offset = combined_indices.len() as u32;
        if let Some(ref bd) = self.building_data {
            sub_mesh_infos.push(mesh::SubMeshInfo {
                vertex_offset: building_vert_offset,
                index_offset: building_idx_offset,
                vertex_count: bd.0.len() as u32,
                index_count: bd.1.len() as u32,
            });
            combined_verts.extend_from_slice(&bd.0);
            combined_indices.extend_from_slice(&bd.1);
        } else {
            sub_mesh_infos.push(mesh::SubMeshInfo {
                vertex_offset: building_vert_offset,
                index_offset: building_idx_offset,
                vertex_count: 0,
                index_count: 0,
            });
        }
        // Pad to full capacity.
        let zero_vert = mesh::Vertex { position: glam::Vec3::ZERO, normal: glam::Vec3::ZERO, color: glam::Vec3::ZERO };
        combined_verts.resize(total_verts + building_vert_cap, zero_vert);
        combined_indices.resize(total_indices + building_idx_cap, 0);

        // Destroy old mesh and create new one.
        self.mesh.destroy(&self.context.device);
        self.mesh = Mesh::new(&self.context, &combined_verts, &combined_indices)?;
        self.sub_mesh_infos = sub_mesh_infos;
        self.building_capacity = Some((building_vert_cap as u32, building_idx_cap as u32));

        // Rebuild specified BLASes (terrain chunks that changed).
        for &idx in rebuild_blas_indices {
            if idx < self.blas_list.len() {
                self.blas_list[idx].destroy(&self.context);
                self.blas_list[idx] =
                    Blas::from_range(&self.context, &self.mesh, &self.sub_mesh_infos[idx])?;
            }
        }

        // Add building BLAS if we have building data.
        if self.building_data.is_some() {
            let building_info = &self.sub_mesh_infos[self.base_mesh_count];
            let building_blas = Blas::from_range(&self.context, &self.mesh, building_info)?;
            self.blas_list.push(building_blas);
        }

        // Rebuild mesh offsets buffer.
        unsafe {
            self.context.device.destroy_buffer(self.mesh_offsets_buf.buffer, None);
            self.context.device.free_memory(self.mesh_offsets_buf.memory, None);
        }
        self.mesh_offsets_buf = create_mesh_offsets_buffer(&self.context, &self.sub_mesh_infos)?;

        // Update descriptor sets (new vertex/index/offsets buffers).
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            self.write_descriptor_set(i);
        }

        Ok(())
    }

    pub fn has_building_blas(&self) -> bool {
        self.blas_list.len() > self.base_mesh_count
    }

    pub fn mesh_building_id(&self) -> u32 {
        self.base_mesh_count as u32
    }

    pub fn draw_frame(
        &mut self,
        transforms: &[Mat4],
        instance_ids: &[u32],   // packed: mesh_type << 16 | object_id
        view: Mat4,
        proj: Mat4,
        light_dir: Vec4,
        light_color: Vec4,
        player_vp: Mat4,
        ghost_mode: bool,
        pry_progress: f32,
        tool_type: f32,
        debug_info: Vec4,
        debug_info2: Vec4,
    ) -> Result<()> {
        if self.surface_width == 0 || self.surface_height == 0 {
            return Ok(());
        }

        if self.swapchain_dirty {
            self.recreate_swapchain_and_storage()?;
            self.swapchain_dirty = false;
        }

        let fi = self.current_frame;

        // Write UBO.
        unsafe {
            *self.scene_ubo_buffers[fi].mapped = SceneUBO {
                inv_view: view.inverse(),
                inv_proj: proj.inverse(),
                light_dir,
                light_color,
                player_vp,
                ghost_mode: Vec4::new(if ghost_mode { 1.0 } else { 0.0 }, pry_progress, tool_type, 0.0),
                debug_info,
                debug_info2,
            };
        }

        let frame = &self.frames[fi];

        unsafe {
            self.context.device.wait_for_fences(
                std::slice::from_ref(&frame.in_flight),
                true,
                u64::MAX,
            )?;
        }

        let image_index = match unsafe {
            self.swapchain.loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                frame.image_available,
                vk::Fence::null(),
            )
        } {
            Ok((index, _suboptimal)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.swapchain_dirty = true;
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };

        unsafe {
            self.context.device.reset_fences(std::slice::from_ref(&frame.in_flight))?;
            self.context.device.reset_command_pool(
                frame.command_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;
        }

        let cb = frame.command_buffer;
        unsafe {
            self.context.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        // Build TLAS instances — each references the correct BLAS for its mesh type.
        let instances: Vec<vk::AccelerationStructureInstanceKHR> = transforms
            .iter()
            .zip(instance_ids.iter())
            .map(|(&t, &packed_id)| {
                let mesh_type = (packed_id >> 16) as usize;
                let blas_address = self.blas_list[mesh_type.min(self.blas_list.len() - 1)].device_address;
                vk::AccelerationStructureInstanceKHR {
                    transform: mat4_to_transform(t),
                    instance_custom_index_and_mask: vk::Packed24_8::new(packed_id, 0xFF),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        0,
                        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: blas_address,
                    },
                }
            })
            .collect();

        self.tlas.update(&self.context, cb, &instances);

        // Trace rays.
        unsafe {
            self.context.device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.rt_pipeline.pipeline,
            );
            self.context.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.rt_pipeline.layout,
                0,
                &[self.descriptor_sets[fi]],
                &[],
            );

            self.context.rt_pipeline_loader.cmd_trace_rays(
                cb,
                &self.rt_pipeline.raygen_region,
                &self.rt_pipeline.miss_region,
                &self.rt_pipeline.hit_region,
                &self.rt_pipeline.callable_region,
                self.extent.width,
                self.extent.height,
                1,
            );

            // storage image GENERAL → TRANSFER_SRC
            image_barrier(
                &self.context.device,
                cb,
                self.storage_image.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::TRANSFER,
            );

            // swapchain image UNDEFINED → TRANSFER_DST
            let sc_image = self.swapchain.images[image_index as usize];
            image_barrier(
                &self.context.device,
                cb,
                sc_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            );

            // Blit storage → swapchain.
            let blit = vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.extent.width as i32,
                        y: self.extent.height as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.swapchain.extent.width as i32,
                        y: self.swapchain.extent.height as i32,
                        z: 1,
                    },
                ],
            };
            self.context.device.cmd_blit_image(
                cb,
                self.storage_image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                sc_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&blit),
                vk::Filter::LINEAR,
            );

            // swapchain image TRANSFER_DST → PRESENT_SRC
            image_barrier(
                &self.context.device,
                cb,
                sc_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            );

            // storage image back to GENERAL for next frame
            image_barrier(
                &self.context.device,
                cb,
                self.storage_image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );

            self.context.device.end_command_buffer(cb)?;
        }

        let render_finished = self.swapchain.render_finished[image_index as usize];
        let wait_semaphores = [frame.image_available];
        let wait_stages = [vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR];
        let signal_semaphores = [render_finished];

        unsafe {
            self.context.device.queue_submit(
                self.context.graphics_queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&wait_stages)
                    .command_buffers(&[cb])
                    .signal_semaphores(&signal_semaphores)],
                frame.in_flight,
            )?;
        }

        match unsafe {
            self.swapchain.loader.queue_present(
                self.context.graphics_queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&signal_semaphores)
                    .swapchains(&[self.swapchain.handle])
                    .image_indices(&[image_index]),
            )
        } {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.swapchain_dirty = true,
            Ok(false) => {}
            Err(e) => return Err(e.into()),
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe { self.context.device.device_wait_idle().ok() };

        for frame in &self.frames {
            unsafe {
                self.context.device.destroy_semaphore(frame.image_available, None);
                self.context.device.destroy_fence(frame.in_flight, None);
                self.context.device.destroy_command_pool(frame.command_pool, None);
            }
        }
        for ubo in &self.scene_ubo_buffers {
            unsafe {
                self.context.device.unmap_memory(ubo.memory);
                self.context.device.destroy_buffer(ubo.buffer, None);
                self.context.device.free_memory(ubo.memory, None);
            }
        }
        unsafe {
            self.context.device.destroy_buffer(self.mesh_offsets_buf.buffer, None);
            self.context.device.free_memory(self.mesh_offsets_buf.memory, None);
            self.context.device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
        self.rt_pipeline.destroy(&self.context.device);
        self.tlas.destroy(&self.context);
        for blas in &self.blas_list {
            blas.destroy(&self.context);
        }
        self.mesh.destroy(&self.context.device);
        unsafe {
            self.context.device.destroy_image_view(self.storage_image.view, None);
            self.context.device.destroy_image(self.storage_image.image, None);
            self.context.device.free_memory(self.storage_image.memory, None);
        }
        self.swapchain.destroy(&self.context);
    }
}

// ---------------------------------------------------------------------------
// Image layout transition helper
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn image_barrier(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);

    unsafe {
        device.cmd_pipeline_barrier(
            cb,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    }
}
