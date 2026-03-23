use std::sync::Arc;

use anyhow::{Context, Result};
use ash::vk;
use winit::window::Window;

use super::context::VulkanContext;
use super::mesh;
use super::swapchain::Swapchain;

const BAR_HEIGHT: u32 = 4;

/// Fill colour (B, G, R, A) for the completed portion of the bar.
const FILL_COLOR: [u8; 4] = [210, 210, 220, 255];
/// Track colour (B, G, R, A) for the unfilled portion of the bar.
const TRACK_COLOR: [u8; 4] = [45, 42, 40, 255];

/// Lightweight loading screen that can present a progress bar before the full
/// renderer is ready.  Owns VulkanContext + Swapchain so they can be handed
/// off to `Renderer::new` afterwards via `into_parts()`.
pub struct LoadingScreen {
    context: VulkanContext,
    swapchain: Swapchain,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    fence: vk::Fence,
    bar_buffer: vk::Buffer,
    bar_memory: vk::DeviceMemory,
    bar_mapped: *mut u8,
    bar_width: u32,
    bar_x: u32,
    bar_y: u32,
}

unsafe impl Send for LoadingScreen {}

impl LoadingScreen {
    pub fn new(window: &Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let context = VulkanContext::new(window.as_ref())?;
        let swapchain = Swapchain::new(&context, size.width, size.height)?;

        let pool = unsafe {
            context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.graphics_queue_family)
                    .flags(
                        vk::CommandPoolCreateFlags::TRANSIENT
                            | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    ),
                None,
            )
        }
        .context("Failed to create loading command pool")?;

        let cb = unsafe {
            context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .context("Failed to allocate loading command buffer")?[0];

        let sem = unsafe {
            context
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        }
        .context("Failed to create loading semaphore")?;

        let fence = unsafe {
            context.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .context("Failed to create loading fence")?;

        // Bar dimensions — 40 % of screen width, centred, at 60 % height.
        let bar_width = (swapchain.extent.width * 2 / 5).max(100);
        let bar_x = (swapchain.extent.width - bar_width) / 2;
        let bar_y = swapchain.extent.height * 3 / 5;

        // Staging buffer for bar pixels.
        let buf_size = (bar_width * BAR_HEIGHT * 4) as vk::DeviceSize;
        let (bar_buffer, bar_memory) = mesh::create_buffer(
            &context.device,
            &context.memory_properties,
            buf_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            false,
        )?;
        let bar_mapped = unsafe {
            context
                .device
                .map_memory(bar_memory, 0, buf_size, vk::MemoryMapFlags::empty())
        }
        .context("Failed to map bar staging buffer")? as *mut u8;

        Ok(Self {
            context,
            swapchain,
            command_pool: pool,
            command_buffer: cb,
            image_available: sem,
            fence,
            bar_buffer,
            bar_memory,
            bar_mapped,
            bar_width,
            bar_x,
            bar_y,
        })
    }

    /// Present a loading frame with a progress bar (`progress` from 0.0 to 1.0).
    pub fn present_progress(&self, progress: f32) -> Result<()> {
        let progress = progress.clamp(0.0, 1.0);
        let filled = ((progress * self.bar_width as f32) as u32).max(if progress > 0.0 { 1 } else { 0 });

        // Fill staging buffer row-by-row.
        let row_bytes = self.bar_width as usize * 4;
        for row in 0..BAR_HEIGHT as usize {
            let row_start = row * row_bytes;
            for col in 0..self.bar_width as usize {
                let color = if col < filled as usize {
                    &FILL_COLOR
                } else {
                    &TRACK_COLOR
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        color.as_ptr(),
                        self.bar_mapped.add(row_start + col * 4),
                        4,
                    );
                }
            }
        }

        let device = &self.context.device;

        unsafe { device.wait_for_fences(&[self.fence], true, u64::MAX)? };
        unsafe { device.reset_fences(&[self.fence])? };

        let image_index = match unsafe {
            self.swapchain.loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        } {
            Ok((idx, _)) => idx,
            Err(_) => return Ok(()),
        };

        let cb = self.command_buffer;
        unsafe {
            device.reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        let sc_image = self.swapchain.images[image_index as usize];
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        // Transition UNDEFINED → TRANSFER_DST.
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(sc_image)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
            );
        }

        // Clear entire image to dark background.
        let clear_color = vk::ClearColorValue {
            float32: [0.02, 0.02, 0.03, 1.0],
        };
        unsafe {
            device.cmd_clear_color_image(
                cb,
                sc_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &[subresource_range],
            );
        }

        // Copy bar pixels from staging buffer onto the swapchain image.
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: self.bar_width,
            buffer_image_height: BAR_HEIGHT,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D {
                x: self.bar_x as i32,
                y: self.bar_y as i32,
                z: 0,
            },
            image_extent: vk::Extent3D {
                width: self.bar_width,
                height: BAR_HEIGHT,
                depth: 1,
            },
        };
        unsafe {
            device.cmd_copy_buffer_to_image(
                cb,
                self.bar_buffer,
                sc_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        // Transition TRANSFER_DST → PRESENT_SRC.
        unsafe {
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(sc_image)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())],
            );

            device.end_command_buffer(cb)?;
        }

        let render_finished = self.swapchain.render_finished[image_index as usize];
        unsafe {
            device.queue_submit(
                self.context.graphics_queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&[self.image_available])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                    .command_buffers(&[cb])
                    .signal_semaphores(&[render_finished])],
                self.fence,
            )?;

            self.swapchain
                .loader
                .queue_present(
                    self.context.graphics_queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&[render_finished])
                        .swapchains(&[self.swapchain.handle])
                        .image_indices(&[image_index]),
                )
                .ok();
        }

        Ok(())
    }

    /// Consume the loading screen and return VulkanContext + Swapchain for the
    /// full renderer.
    pub fn into_parts(self) -> Result<(VulkanContext, Swapchain)> {
        unsafe {
            self.context.device.device_wait_idle()?;
            self.context.device.destroy_semaphore(self.image_available, None);
            self.context.device.destroy_fence(self.fence, None);
            self.context.device.destroy_command_pool(self.command_pool, None);
            self.context.device.unmap_memory(self.bar_memory);
            self.context.device.destroy_buffer(self.bar_buffer, None);
            self.context.device.free_memory(self.bar_memory, None);
        }
        Ok((self.context, self.swapchain))
    }
}
