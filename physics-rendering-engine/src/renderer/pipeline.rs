use anyhow::{Context, Result};
use ash::{vk, Device, Instance};

use super::mesh::Vertex;

pub struct Pipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

// ---------------------------------------------------------------------------
// Depth format
// ---------------------------------------------------------------------------

pub fn find_depth_format(instance: &Instance, physical_device: vk::PhysicalDevice) -> vk::Format {
    [
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ]
    .into_iter()
    .find(|&fmt| {
        let props =
            unsafe { instance.get_physical_device_format_properties(physical_device, fmt) };
        props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
    })
    .expect("No supported depth format found")
}

// ---------------------------------------------------------------------------
// Render pass
// ---------------------------------------------------------------------------

pub fn create_render_pass(
    device: &Device,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let depth_attachment = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref))
        .depth_stencil_attachment(&depth_ref);

    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    let attachments = [color_attachment, depth_attachment];
    let subpasses = [subpass];
    let dependencies = [dependency];

    unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies),
            None,
        )
    }
    .context("Failed to create render pass")
}

// ---------------------------------------------------------------------------
// Descriptor set layout
// ---------------------------------------------------------------------------

/// One UBO binding at set=0, binding=0 — holds view + proj matrices.
pub fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(std::slice::from_ref(&binding)),
            None,
        )
    }
    .context("Failed to create descriptor set layout")
}

// ---------------------------------------------------------------------------
// Graphics pipeline
// ---------------------------------------------------------------------------

pub fn create_graphics_pipeline(
    device: &Device,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<Pipeline> {
    let vert_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/mesh.vert.spv"));
    let frag_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/mesh.frag.spv"));

    let vert_spv = ash::util::read_spv(&mut std::io::Cursor::new(vert_bytes))
        .context("Failed to parse vertex shader SPIR-V")?;
    let frag_spv = ash::util::read_spv(&mut std::io::Cursor::new(frag_bytes))
        .context("Failed to parse fragment shader SPIR-V")?;

    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_spv),
            None,
        )
    }
    .context("Failed to create vertex shader module")?;

    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_spv),
            None,
        )
    }
    .context("Failed to create fragment shader module")?;

    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry),
    ];

    let binding = Vertex::binding_description();
    let attributes = Vertex::attribute_descriptions();
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(std::slice::from_ref(&binding))
        .vertex_attribute_descriptions(&attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false);

    let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(std::slice::from_ref(&color_blend_attachment));

    // One Mat4 per object, uploaded as a push constant before each draw call.
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(std::mem::size_of::<glam::Mat4>() as u32);

    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                .push_constant_ranges(std::slice::from_ref(&push_range)),
            None,
        )
    }
    .context("Failed to create pipeline layout")?;

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blending)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let handle = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| e)
    .context("Failed to create graphics pipeline")?[0];

    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }

    Ok(Pipeline { handle, layout })
}
