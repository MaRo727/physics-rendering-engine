use anyhow::{Context, Result};
use ash::vk;

use super::context::VulkanContext;

pub struct Swapchain {
    pub loader: ash::khr::swapchain::Device,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    /// One semaphore per swapchain image. Indexed by the image_index returned
    /// by acquire_next_image. This avoids semaphore reuse while the
    /// presentation engine still holds the semaphore from a previous present.
    pub render_finished: Vec<vk::Semaphore>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(
        context: &VulkanContext,
        render_pass: vk::RenderPass,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        create_swapchain(context, render_pass, vk::SwapchainKHR::null(), width, height)
    }

    /// Destroy the old swapchain resources and create fresh ones at the new size.
    pub fn recreate(
        &mut self,
        context: &VulkanContext,
        render_pass: vk::RenderPass,
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
            for &iv in &self.image_views {
                context.device.destroy_image_view(iv, None);
            }
        }

        // Pass old_handle as a hint — driver can reuse some memory.
        // We replace self before destroying old_handle so the new loader
        // (which shares the same device function pointers) can destroy it.
        let new = create_swapchain(context, render_pass, old_handle, width, height)?;
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
            for &iv in &self.image_views {
                context.device.destroy_image_view(iv, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Surface format helpers (pub so Renderer can query the format before
// creating the render pass)
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

    let loader =
        ash::khr::swapchain::Device::new(&context.instance, &context.device);
    let handle = unsafe { loader.create_swapchain(&create_info, None) }
        .context("Failed to create swapchain")?;

    let images =
        unsafe { loader.get_swapchain_images(handle) }.context("Failed to get swapchain images")?;

    let image_views = create_image_views(&context.device, &images, format.format)?;
    let framebuffers =
        create_framebuffers(&context.device, &image_views, render_pass, extent)?;

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
        format: format.format,
        extent,
    })
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

fn choose_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    width: u32,
    height: u32,
) -> vk::Extent2D {
    // u32::MAX signals "pick whatever size the window is"
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
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    image_views
        .iter()
        .map(|&view| {
            let attachments = [view];
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
