use anyhow::{Context, Result};
use ash::vk;

use super::context::VulkanContext;

pub struct Swapchain {
    pub loader: ash::khr::swapchain::Device,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    /// One semaphore per swapchain image — see Phase 3 comment in mod.rs.
    pub render_finished: Vec<vk::Semaphore>,
    // Depth buffer — one shared image for all swapchain images (only one
    // frame renders at a time per the in-flight fence).
    depth_image: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(
        context: &VulkanContext,
        render_pass: vk::RenderPass,
        depth_format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        create_swapchain(context, render_pass, depth_format, vk::SwapchainKHR::null(), width, height)
    }

    pub fn recreate(
        &mut self,
        context: &VulkanContext,
        render_pass: vk::RenderPass,
        depth_format: vk::Format,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let old_handle = self.handle;
        unsafe {
            for &sem in &self.render_finished {
                context.device.destroy_semaphore(sem, None);
            }
            for &fb in &self.framebuffers {
                context.device.destroy_framebuffer(fb, None);
            }
            context.device.destroy_image_view(self.depth_view, None);
            context.device.destroy_image(self.depth_image, None);
            context.device.free_memory(self.depth_memory, None);
            for &iv in &self.image_views {
                context.device.destroy_image_view(iv, None);
            }
        }
        let new = create_swapchain(context, render_pass, depth_format, old_handle, width, height)?;
        unsafe { self.loader.destroy_swapchain(old_handle, None) };
        *self = new;
        Ok(())
    }

    pub fn destroy(&self, context: &VulkanContext) {
        unsafe {
            for &sem in &self.render_finished {
                context.device.destroy_semaphore(sem, None);
            }
            for &fb in &self.framebuffers {
                context.device.destroy_framebuffer(fb, None);
            }
            context.device.destroy_image_view(self.depth_view, None);
            context.device.destroy_image(self.depth_image, None);
            context.device.free_memory(self.depth_memory, None);
            for &iv in &self.image_views {
                context.device.destroy_image_view(iv, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Surface format query
// ---------------------------------------------------------------------------

pub fn query_surface_format(context: &VulkanContext) -> Result<vk::SurfaceFormatKHR> {
    let formats = unsafe {
        context
            .surface_loader
            .get_physical_device_surface_formats(context.physical_device, context.surface)
    }
    .context("Failed to query surface formats")?;
    Ok(choose_format(&formats))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn create_swapchain(
    context: &VulkanContext,
    render_pass: vk::RenderPass,
    depth_format: vk::Format,
    old_swapchain: vk::SwapchainKHR,
    width: u32,
    height: u32,
) -> Result<Swapchain> {
    let capabilities = unsafe {
        context
            .surface_loader
            .get_physical_device_surface_capabilities(context.physical_device, context.surface)
    }
    .context("Failed to get surface capabilities")?;

    let formats = unsafe {
        context
            .surface_loader
            .get_physical_device_surface_formats(context.physical_device, context.surface)
    }
    .context("Failed to get surface formats")?;

    let present_modes = unsafe {
        context
            .surface_loader
            .get_physical_device_surface_present_modes(context.physical_device, context.surface)
    }
    .context("Failed to get surface present modes")?;

    let format = choose_format(&formats);
    let present_mode = choose_present_mode(&present_modes);
    let extent = choose_extent(&capabilities, width, height);

    let image_count = {
        let preferred = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 {
            preferred.min(capabilities.max_image_count)
        } else {
            preferred
        }
    };

    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(context.surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(old_swapchain);

    let loader = ash::khr::swapchain::Device::new(&context.instance, &context.device);
    let handle = unsafe { loader.create_swapchain(&create_info, None) }
        .context("Failed to create swapchain")?;

    let images =
        unsafe { loader.get_swapchain_images(handle) }.context("Failed to get swapchain images")?;

    let image_views = create_image_views(&context.device, &images, format.format)?;

    let (depth_image, depth_memory, depth_view) =
        create_depth_resources(context, depth_format, extent)?;

    let framebuffers =
        create_framebuffers(&context.device, &image_views, depth_view, render_pass, extent)?;

    let sem_info = vk::SemaphoreCreateInfo::default();
    let render_finished = (0..images.len())
        .map(|_| {
            unsafe { context.device.create_semaphore(&sem_info, None) }
                .context("Failed to create render_finished semaphore")
        })
        .collect::<Result<Vec<_>>>()?;

    log::debug!(
        "Swapchain: {}x{} {:?} {} images {present_mode:?}",
        extent.width,
        extent.height,
        format.format,
        images.len()
    );

    Ok(Swapchain {
        loader,
        handle,
        images,
        image_views,
        framebuffers,
        render_finished,
        depth_image,
        depth_memory,
        depth_view,
        format: format.format,
        extent,
    })
}

fn create_depth_resources(
    context: &VulkanContext,
    depth_format: vk::Format,
    extent: vk::Extent2D,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let image = unsafe {
        context.device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(depth_format)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .context("Failed to create depth image")?;

    let reqs = unsafe { context.device.get_image_memory_requirements(image) };
    let memory_type = find_memory_type(
        &context.memory_properties,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .context("No suitable memory type for depth image")?;

    let memory = unsafe {
        context.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(memory_type),
            None,
        )
    }
    .context("Failed to allocate depth image memory")?;

    unsafe { context.device.bind_image_memory(image, memory, 0) }
        .context("Failed to bind depth image memory")?;

    let aspect = if depth_format == vk::Format::D32_SFLOAT {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    };

    let view = unsafe {
        context.device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(depth_format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            None,
        )
    }
    .context("Failed to create depth image view")?;

    Ok((image, memory, view))
}

fn choose_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .copied()
        .unwrap_or(formats[0])
}

fn choose_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    }
}

fn choose_extent(capabilities: &vk::SurfaceCapabilitiesKHR, width: u32, height: u32) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

fn create_image_views(
    device: &ash::Device,
    images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    images
        .iter()
        .map(|&image| {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .components(vk::ComponentMapping::default())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe { device.create_image_view(&create_info, None) }
                .context("Failed to create image view")
        })
        .collect()
}

fn create_framebuffers(
    device: &ash::Device,
    image_views: &[vk::ImageView],
    depth_view: vk::ImageView,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    image_views
        .iter()
        .map(|&color_view| {
            let attachments = [color_view, depth_view];
            let create_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe { device.create_framebuffer(&create_info, None) }
                .context("Failed to create framebuffer")
        })
        .collect()
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        type_filter & (1 << i) != 0
            && props.memory_types[i as usize].property_flags.contains(required)
    })
}
