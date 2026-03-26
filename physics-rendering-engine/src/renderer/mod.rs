pub mod acceleration_structure;
pub mod context;
pub mod frame;
pub mod loading;
pub mod mesh;
pub mod rt_pipeline;
pub mod shapes;
pub mod swapchain;

use anyhow::{Context, Result};
use ash::vk;
use glam::{Mat4, Vec3, Vec4};
use acceleration_structure::{Blas, Tlas, mat4_to_transform};
use context::VulkanContext;
use frame::{create_frame, FrameData, MAX_FRAMES_IN_FLIGHT};
use mesh::Mesh;
use rt_pipeline::{RtPipeline, SceneUBO};
use swapchain::Swapchain;

use crate::ui;

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
pub const MESH_GRASS_A: u32 = 14;
pub const MESH_GRASS_B: u32 = 15;
pub const MESH_GRASS_C: u32 = 16;
pub const MESH_FLOWER_RED: u32 = 17;
pub const MESH_FLOWER_YELLOW: u32 = 18;
pub const MESH_FLOWER_BLUE: u32 = 19;
pub const MESH_FLOWER_WHITE: u32 = 20;
pub const MESH_FLOWER_PURPLE: u32 = 21;
pub const MESH_SLIME: u32 = 22;
pub const MESH_FIST: u32 = 23;
pub const MESH_FIREBALL: u32 = 24;
pub const MESH_ICESHARD: u32 = 25;
pub const MESH_SKELETON: u32 = 26;
pub const MESH_GOBLIN: u32 = 27;
pub const MESH_GOLEM: u32 = 28;
pub const MESH_ARROW: u32 = 29;
pub const MESH_BLOCK_SLAB: u32 = 30;
pub const MESH_BLOCK_VSLAB: u32 = 31;
pub const MESH_BLOCK_SLOPE: u32 = 32;
pub const MESH_BLOCK_INNER_CORNER: u32 = 33;
pub const MESH_BLOCK_STAIRS: u32 = 34;
pub const MESH_BLOCK_FENCE: u32 = 35;
pub const MESH_CACTUS: u32 = 36;
pub const MESH_CACTUS_SMALL: u32 = 37;
pub const MESH_CACTUS_LOD: u32 = 38;
pub const MESH_CACTUS_SMALL_LOD: u32 = 39;
pub const MESH_LEAF_PARTICLE: u32 = 40;
pub const MESH_BARK_CHIP: u32 = 41;
pub const MESH_TORCH: u32 = 42;
pub const MESH_TERRAIN_BASE: u32 = 43;
const SHAPE_MESH_COUNT: usize = 43;

/// Pre-allocated capacity for the building mesh slot in the combined buffer.
const BUILDING_INITIAL_VERTS: u32 = 65536;
const BUILDING_INITIAL_INDICES: u32 = 98304;

/// Pack mesh_type (upper 8 bits) and object_id (lower 16 bits) into 24-bit custom index.
pub fn pack_instance_id(mesh_type: u32, object_id: u32) -> u32 {
    (mesh_type << 16) | (object_id & 0xFFFF)
}

/// Flag bit (in upper byte of packed u32) marking an instance as shadow-only.
/// Shadow-only instances use TLAS mask 0x02 so primary rays (mask 0xFD) skip them
/// while shadow rays (mask 0xFF) still hit them.
pub const SHADOW_ONLY_BIT: u32 = 0x8000_0000;

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
// Per-frame UI buffer (host-visible, written each frame)
// ---------------------------------------------------------------------------

struct UiGpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    capacity: usize, // total byte capacity
}

unsafe impl Send for UiGpuBuffer {}

fn create_ui_buffer(context: &VulkanContext) -> Result<UiGpuBuffer> {
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
const POINT_LIGHT_BUFFER_BYTES: usize = 16 + MAX_POINT_LIGHTS * std::mem::size_of::<GpuPointLight>();

struct PointLightBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
}

unsafe impl Send for PointLightBuffer {}

fn create_point_light_buffer(context: &VulkanContext) -> Result<PointLightBuffer> {
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
    ui_buffers: [UiGpuBuffer; MAX_FRAMES_IN_FLIGHT],
    point_light_buffers: [PointLightBuffer; MAX_FRAMES_IN_FLIGHT],
    mesh_offsets_buf: MeshOffsetsBuffer,
    current_frame: usize,
    extent: vk::Extent2D,
    surface_width: u32,
    surface_height: u32,
    swapchain_dirty: bool,
    render_scale: f32,
}

impl Renderer {
    pub fn new(
        context: VulkanContext,
        swapchain: Swapchain,
        max_instances: u32,
        terrain_chunks: Vec<(Vec<mesh::Vertex>, Vec<u32>)>,
    ) -> Result<Self> {
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
            shapes::grass_patch_a(),             // MESH_GRASS_A = 14
            shapes::grass_patch_b(),             // MESH_GRASS_B = 15
            shapes::grass_patch_c(),             // MESH_GRASS_C = 16
            shapes::flower(Vec3::new(0.85, 0.15, 0.12)),  // MESH_FLOWER_RED = 17
            shapes::flower(Vec3::new(0.90, 0.80, 0.15)),  // MESH_FLOWER_YELLOW = 18
            shapes::flower(Vec3::new(0.20, 0.30, 0.85)),  // MESH_FLOWER_BLUE = 19
            shapes::flower(Vec3::new(0.90, 0.90, 0.88)),  // MESH_FLOWER_WHITE = 20
            shapes::flower(Vec3::new(0.60, 0.20, 0.70)),  // MESH_FLOWER_PURPLE = 21
            shapes::ball_colored(16, 24, Vec3::new(0.15, 0.75, 0.20)), // MESH_SLIME = 22
            shapes::ball_colored(8, 12, Vec3::new(0.85, 0.70, 0.55)),  // MESH_FIST = 23
            shapes::ball_colored(10, 16, Vec3::new(1.0, 0.45, 0.1)),   // MESH_FIREBALL = 24
            shapes::ball_colored(6, 8, Vec3::new(0.5, 0.8, 1.0)),     // MESH_ICESHARD = 25
            shapes::skeleton(),                                           // MESH_SKELETON = 26
            shapes::goblin(),                                             // MESH_GOBLIN = 27
            shapes::golem(),                                              // MESH_GOLEM = 28
            shapes::arrow(),                                              // MESH_ARROW = 29
            shapes::block_slab(),                                             // MESH_BLOCK_SLAB = 30
            shapes::block_vertical_slab(),                                    // MESH_BLOCK_VSLAB = 31
            shapes::block_slope(),                                            // MESH_BLOCK_SLOPE = 32
            shapes::block_inner_corner_slope(),                               // MESH_BLOCK_INNER_CORNER = 33
            shapes::block_stairs(),                                           // MESH_BLOCK_STAIRS = 34
            shapes::block_fence(),                                            // MESH_BLOCK_FENCE = 35
            shapes::cactus(),                                                 // MESH_CACTUS = 36
            shapes::cactus_small(),                                           // MESH_CACTUS_SMALL = 37
            shapes::cactus_lod(),                                             // MESH_CACTUS_LOD = 38
            shapes::cactus_small_lod(),                                       // MESH_CACTUS_SMALL_LOD = 39
            shapes::leaf_particle(),                                             // MESH_LEAF_PARTICLE = 40
            shapes::bark_chip(),                                                 // MESH_BARK_CHIP = 41
            shapes::torch(),                                                         // MESH_TORCH = 42
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

        let ui0 = create_ui_buffer(&context)?;
        let ui1 = create_ui_buffer(&context)?;
        let ui_buffers = [ui0, ui1];

        // Upload static font bitmap data once into every UI buffer so that
        // the per-frame `upload_ui` only needs to write the header + primitives.
        {
            let font = ui::font_gpu_data();
            let font_offset = std::mem::size_of::<ui::UiHeader>();
            let font_bytes = 192 * std::mem::size_of::<u32>();
            for buf in &ui_buffers {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        font.as_ptr() as *const u8,
                        buf.mapped.add(font_offset),
                        font_bytes,
                    );
                }
            }
        }

        let pl0 = create_point_light_buffer(&context)?;
        let pl1 = create_point_light_buffer(&context)?;
        let point_light_buffers = [pl0, pl1];

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
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * 5), // vertex, index, mesh_offsets, UI, point_lights
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
            ui_buffers,
            point_light_buffers,
            mesh_offsets_buf,
            current_frame: 0,
            extent,
            surface_width: extent.width,
            surface_height: extent.height,
            swapchain_dirty: false,
            render_scale: 1.0,
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

        let ui_info = vk::DescriptorBufferInfo::default()
            .buffer(self.ui_buffers[i].buffer)
            .offset(0)
            .range(self.ui_buffers[i].capacity as vk::DeviceSize);

        let point_light_info = vk::DescriptorBufferInfo::default()
            .buffer(self.point_light_buffers[i].buffer)
            .offset(0)
            .range(POINT_LIGHT_BUFFER_BYTES as vk::DeviceSize);

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
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&ui_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&point_light_info)),
        ];

        unsafe { self.context.device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_width = width;
        self.surface_height = height;
        self.swapchain_dirty = true;
    }

    pub fn set_render_scale(&mut self, scale: f32) {
        self.render_scale = scale;
        self.swapchain_dirty = true;
    }

    /// Upload UI primitives for the current frame.
    /// Must be called before `draw_frame`.
    /// Font bitmap data is written once at init; only the header and dynamic
    /// primitives are memcpy'd here.
    pub fn upload_ui(&mut self, prims: &[ui::UiPrimitive], screen_w: u32, screen_h: u32) {
        let fi = self.current_frame;
        let buf = &self.ui_buffers[fi];
        let count = prims.len().min(ui::MAX_UI_PRIMS);

        let header = ui::UiHeader {
            count: count as u32,
            screen_w,
            screen_h,
            _pad: 0,
        };

        unsafe {
            let dst = buf.mapped;
            // Header (16 bytes).
            std::ptr::copy_nonoverlapping(
                &header as *const _ as *const u8,
                dst,
                std::mem::size_of::<ui::UiHeader>(),
            );
            // Font data lives at a fixed offset and was written once at init —
            // skip it here.
            // Primitives.
            if count > 0 {
                let prim_dst = dst.add(
                    std::mem::size_of::<ui::UiHeader>() + 192 * std::mem::size_of::<u32>(),
                );
                std::ptr::copy_nonoverlapping(
                    prims.as_ptr() as *const u8,
                    prim_dst,
                    count * std::mem::size_of::<ui::UiPrimitive>(),
                );
            }
        }
    }

    /// Upload point light data for the current frame.
    /// Must be called before `draw_frame`.
    pub fn upload_point_lights(&mut self, lights: &[GpuPointLight]) {
        let fi = self.current_frame;
        let buf = &self.point_light_buffers[fi];
        let count = lights.len().min(MAX_POINT_LIGHTS);

        unsafe {
            // Header: u32 count + 3 padding u32s.
            let header = buf.mapped as *mut u32;
            *header = count as u32;

            if count > 0 {
                let lights_dst = buf.mapped.add(16) as *mut GpuPointLight;
                std::ptr::copy_nonoverlapping(
                    lights.as_ptr(),
                    lights_dst,
                    count,
                );
            }
        }
    }

    /// Wait for all in-flight frames to finish on the GPU.
    /// Much cheaper than device_wait_idle() — only waits on our frame fences
    /// rather than flushing all driver-internal state.
    #[allow(dead_code)]
    fn wait_all_frames(&self) -> Result<()> {
        let fences: Vec<_> = self.frames.iter().map(|f| f.in_flight).collect();
        unsafe {
            self.context.device.wait_for_fences(&fences, true, u64::MAX)?;
        }
        Ok(())
    }

    fn recreate_swapchain_and_storage(&mut self) -> Result<()> {
        unsafe { self.context.device.device_wait_idle()? };

        self.swapchain.recreate(&self.context, self.surface_width, self.surface_height)?;
        self.extent = vk::Extent2D {
            width: ((self.swapchain.extent.width as f32 * self.render_scale) as u32).max(1),
            height: ((self.swapchain.extent.height as f32 * self.render_scale) as u32).max(1),
        };

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
        // GPU-side pipeline barriers in the staging copy ensure safe synchronization
        // with any in-flight frames — no need to stall the CPU with wait_all_frames().
        if let Some((vert_cap, idx_cap)) = self.building_capacity {
            if new_vert_count <= vert_cap && new_idx_count <= idx_cap {
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
    /// Takes ownership of `updates` so vertex/index data can be moved into
    /// `base_mesh_data` without cloning.
    pub fn update_terrain_chunks(
        &mut self,
        updates: Vec<(usize, Vec<mesh::Vertex>, Vec<u32>)>,
    ) -> Result<()> {
        // GPU-side pipeline barriers in the batched staging copy ensure safe
        // synchronization with any in-flight frames — no CPU stall needed.

        // Build batch references into the owned updates (pre-allocated, no resize).
        let mut batch: Vec<(&mesh::SubMeshInfo, &[mesh::Vertex], &[u32])> =
            Vec::with_capacity(updates.len());
        for (chunk_idx, verts, indices) in &updates {
            let mesh_idx = SHAPE_MESH_COUNT + *chunk_idx;
            batch.push((&self.sub_mesh_infos[mesh_idx], verts.as_slice(), indices.as_slice()));
        }
        self.mesh.update_regions_batched(&self.context, &batch)?;
        drop(batch);

        // Rebuild only dirty BLASes and move CPU data in sync (no clone).
        for (chunk_idx, verts, indices) in updates {
            let mesh_idx = SHAPE_MESH_COUNT + chunk_idx;
            self.blas_list[mesh_idx].destroy(&self.context);
            self.blas_list[mesh_idx] =
                Blas::from_range(&self.context, &self.mesh, &self.sub_mesh_infos[mesh_idx])?;
            self.base_mesh_data[mesh_idx] = (verts, indices);
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
        sun_moon: Vec4,
        moon_info: Vec4,
        blizzard_info: Vec4,
        weather_info: Vec4,
        wind_info: Vec4,
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
                sun_moon,
                moon_info,
                blizzard_info,
                weather_info,
                wind_info,
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

        // Build TLAS instances — write directly into the persistently-mapped
        // GPU instance buffer, avoiding an intermediate Vec and memcpy.
        let instance_count = transforms.len();
        if self.tlas.ensure_capacity(&self.context, instance_count as u32)? {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.write_descriptor_set(i);
            }
        }
        let blas_max = self.blas_list.len() - 1;
        {
            let mapped = unsafe { self.tlas.mapped_instances_mut() };
            let cull_disable_flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8;
            let sbt_offset = vk::Packed24_8::new(0, cull_disable_flags);
            for (i, (&t, &packed_id)) in transforms.iter().zip(instance_ids.iter()).enumerate() {
                let shadow_only = packed_id & SHADOW_ONLY_BIT != 0;
                let clean_id = packed_id & !SHADOW_ONLY_BIT;
                let mesh_type = ((clean_id >> 16) as usize).min(blas_max);
                let is_water = mesh_type == MESH_WATER as usize;
                // Detail geometry (grass, flowers, LOD billboards, particles):
                // mask 0x01 so water bounce rays (mask 0xF8) skip them.
                let is_detail = matches!(mesh_type,
                    11..=21 | 38 | 39 | 40 | 41
                );
                let mask = if shadow_only { 0x02u8 }
                           else if is_water { 0x04u8 }
                           else if is_detail { 0x01u8 }
                           else { 0xFFu8 };
                mapped[i] = vk::AccelerationStructureInstanceKHR {
                    transform: mat4_to_transform(t),
                    instance_custom_index_and_mask: vk::Packed24_8::new(clean_id, mask),
                    instance_shader_binding_table_record_offset_and_flags: sbt_offset,
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: self.blas_list[mesh_type].device_address,
                    },
                };
            }
        }

        self.tlas.record_build(&self.context, cb, instance_count as u32);

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
        for ui_buf in &self.ui_buffers {
            unsafe {
                self.context.device.unmap_memory(ui_buf.memory);
                self.context.device.destroy_buffer(ui_buf.buffer, None);
                self.context.device.free_memory(ui_buf.memory, None);
            }
        }
        for pl_buf in &self.point_light_buffers {
            unsafe {
                self.context.device.unmap_memory(pl_buf.memory);
                self.context.device.destroy_buffer(pl_buf.buffer, None);
                self.context.device.free_memory(pl_buf.memory, None);
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
