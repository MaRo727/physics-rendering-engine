use anyhow::{Context, Result};
use ash::vk;
use glam::Mat4;

use super::context::VulkanContext;
use super::mesh::{self, create_buffer};

// ---------------------------------------------------------------------------
// SceneUBO
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct SceneUBO {
    pub inv_view: Mat4,
    pub inv_proj: Mat4,
    pub light_dir: glam::Vec4,
    pub light_color: glam::Vec4,
    pub player_vp: Mat4,       // frozen player view-projection (for ghost mode highlight)
    pub ghost_mode: glam::Vec4, // .x > 0 when ghost mode is active
    pub debug_info: glam::Vec4, // .x = show debug UI, .y = biome (0=forest,1=desert,2=mountain,3=dungeon), .z = hp fraction, .w = mana fraction
    pub debug_info2: glam::Vec4, // .x = level, .y = stamina fraction, .z = player x, .w = player z
    pub sun_moon: glam::Vec4, // .xyz = sun direction, .w = sun altitude
    pub moon_info: glam::Vec4, // .xyz = moon direction, .w = moon altitude
    pub blizzard_info: glam::Vec4, // .x = snow intensity (0..1), .y = time, .z = water level, .w = unused
    pub weather_info: glam::Vec4,  // .x = rain intensity, .y = fog density, .z = lightning flash, .w = cloud coverage
    pub wind_info: glam::Vec4,     // .x = wind strength, .y = wind dir x, .z = wind dir z, .w = weather time
}

// ---------------------------------------------------------------------------
// RtPipeline
// ---------------------------------------------------------------------------

pub struct RtPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    sbt_buffer: vk::Buffer,
    sbt_memory: vk::DeviceMemory,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
    pub callable_region: vk::StridedDeviceAddressRegionKHR,
}

impl RtPipeline {
    pub fn new(context: &VulkanContext) -> Result<Self> {
        let descriptor_set_layout = create_descriptor_set_layout(&context.device)?;
        let layout = create_pipeline_layout(&context.device, descriptor_set_layout)?;
        let pipeline = create_rt_pipeline(context, layout)?;
        let (sbt_buffer, sbt_memory, raygen_region, miss_region, hit_region) =
            create_sbt(context, pipeline)?;

        Ok(Self {
            pipeline,
            layout,
            descriptor_set_layout,
            sbt_buffer,
            sbt_memory,
            raygen_region,
            miss_region,
            hit_region,
            callable_region: vk::StridedDeviceAddressRegionKHR::default(),
        })
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_buffer(self.sbt_buffer, None);
            device.free_memory(self.sbt_memory, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Descriptor set layout
// ---------------------------------------------------------------------------

fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        // 0: TLAS
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        // 1: output image
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        // 2: SceneUBO
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::MISS_KHR),
        // 3: vertex SSBO
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        // 4: index SSBO
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        // 5: mesh offsets SSBO (index_base, vertex_base per mesh type)
        vk::DescriptorSetLayoutBinding::default()
            .binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        // 6: UI SSBO (immediate-mode primitives: header + font + quads/glyphs)
        vk::DescriptorSetLayoutBinding::default()
            .binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        // 7: Point lights SSBO
        vk::DescriptorSetLayoutBinding::default()
            .binding(7)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
    ];

    unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .context("Failed to create RT descriptor set layout")
}

fn create_pipeline_layout(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout)),
            None,
        )
    }
    .context("Failed to create RT pipeline layout")
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

fn create_rt_pipeline(context: &VulkanContext, layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
    let raygen_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/raygen.rgen.spv"));
    let miss_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/miss.rmiss.spv"));
    let shadow_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.rmiss.spv"));
    let chit_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/closesthit.rchit.spv"));

    let raygen_spv = ash::util::read_spv(&mut std::io::Cursor::new(raygen_bytes))
        .context("Failed to parse raygen SPIR-V")?;
    let miss_spv = ash::util::read_spv(&mut std::io::Cursor::new(miss_bytes))
        .context("Failed to parse miss SPIR-V")?;
    let shadow_spv = ash::util::read_spv(&mut std::io::Cursor::new(shadow_bytes))
        .context("Failed to parse shadow SPIR-V")?;
    let chit_spv = ash::util::read_spv(&mut std::io::Cursor::new(chit_bytes))
        .context("Failed to parse closest-hit SPIR-V")?;

    let device = &context.device;

    let raygen_module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&raygen_spv), None)
    }
    .context("Failed to create raygen shader module")?;
    let miss_module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&miss_spv), None)
    }
    .context("Failed to create miss shader module")?;
    let shadow_module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&shadow_spv), None)
    }
    .context("Failed to create shadow shader module")?;
    let chit_module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&chit_spv), None)
    }
    .context("Failed to create closest-hit shader module")?;

    let entry = c"main";

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(raygen_module)
            .name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(miss_module)
            .name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(shadow_module)
            .name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(chit_module)
            .name(entry),
    ];

    // Groups: raygen(0), miss(1), shadow miss(2), closest-hit(3)
    let groups = [
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(1)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(2)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(3)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
    ];

    let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
        .stages(&stages)
        .groups(&groups)
        .max_pipeline_ray_recursion_depth(3)
        .layout(layout);

    let pipeline = unsafe {
        context.rt_pipeline_loader.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| e)
    .context("Failed to create RT pipeline")?[0];

    unsafe {
        device.destroy_shader_module(raygen_module, None);
        device.destroy_shader_module(miss_module, None);
        device.destroy_shader_module(shadow_module, None);
        device.destroy_shader_module(chit_module, None);
    }

    Ok(pipeline)
}

// ---------------------------------------------------------------------------
// Shader Binding Table
// ---------------------------------------------------------------------------

fn create_sbt(
    context: &VulkanContext,
    pipeline: vk::Pipeline,
) -> Result<(
    vk::Buffer,
    vk::DeviceMemory,
    vk::StridedDeviceAddressRegionKHR,
    vk::StridedDeviceAddressRegionKHR,
    vk::StridedDeviceAddressRegionKHR,
)> {
    let rt_props = &context.rt_props;
    let handle_size = rt_props.shader_group_handle_size as usize;
    let handle_align = rt_props.shader_group_handle_alignment as usize;
    let base_align = rt_props.shader_group_base_alignment as usize;

    // Aligned handle size.
    let handle_stride = align_up(handle_size, handle_align);

    // Group layout: 1 raygen, 2 miss, 1 hit.
    let group_count = 4usize;
    let raygen_count = 1usize;
    let miss_count = 2usize;
    let hit_count = 1usize;

    // Each region is aligned to base_align.
    let raygen_region_size = align_up(raygen_count * handle_stride, base_align);
    let miss_region_size   = align_up(miss_count   * handle_stride, base_align);
    let hit_region_size    = align_up(hit_count    * handle_stride, base_align);

    let total_size = raygen_region_size + miss_region_size + hit_region_size;

    // Fetch all group handles from the pipeline.
    let handles = unsafe {
        context.rt_pipeline_loader.get_ray_tracing_shader_group_handles(
            pipeline,
            0,
            group_count as u32,
            group_count * handle_size,
        )
    }
    .context("Failed to get SBT handles")?;

    // Allocate host-visible SBT buffer.
    let (sbt_buffer, sbt_memory) = create_buffer(
        &context.device,
        &context.memory_properties,
        total_size as vk::DeviceSize,
        vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        true,
    )?;

    // Map and write handles into the SBT regions.
    let sbt_ptr = unsafe {
        context.device.map_memory(
            sbt_memory,
            0,
            total_size as vk::DeviceSize,
            vk::MemoryMapFlags::empty(),
        )
    }
    .context("Failed to map SBT memory")? as *mut u8;

    unsafe {
        let src = handles.as_ptr();

        // Raygen (group 0) at offset 0.
        sbt_ptr.copy_from_nonoverlapping(src, handle_size);

        // Miss groups (1, 2) after raygen region.
        let miss_base = sbt_ptr.add(raygen_region_size);
        miss_base.copy_from_nonoverlapping(src.add(handle_size), handle_size);
        miss_base.add(handle_stride).copy_from_nonoverlapping(src.add(2 * handle_size), handle_size);

        // Hit group (3) after miss region.
        let hit_base = sbt_ptr.add(raygen_region_size + miss_region_size);
        hit_base.copy_from_nonoverlapping(src.add(3 * handle_size), handle_size);

        context.device.unmap_memory(sbt_memory);
    }

    let sbt_address = mesh::get_device_address(&context.device, sbt_buffer);

    let raygen_region = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_address,
        stride: raygen_region_size as vk::DeviceSize, // raygen stride == size
        size: raygen_region_size as vk::DeviceSize,
    };
    let miss_region = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_address + raygen_region_size as vk::DeviceSize,
        stride: handle_stride as vk::DeviceSize,
        size: miss_region_size as vk::DeviceSize,
    };
    let hit_region = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_address + raygen_region_size as vk::DeviceSize + miss_region_size as vk::DeviceSize,
        stride: handle_stride as vk::DeviceSize,
        size: hit_region_size as vk::DeviceSize,
    };

    Ok((sbt_buffer, sbt_memory, raygen_region, miss_region, hit_region))
}

fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}
