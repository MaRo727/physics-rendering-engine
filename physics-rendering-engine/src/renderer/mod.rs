pub mod context;
pub mod frame;
pub mod mesh;
pub mod pipeline;
pub mod swapchain;

use anyhow::Result;
use ash::vk;
use std::sync::Arc;
use winit::window::Window;

use context::VulkanContext;
use frame::{create_frame, FrameData, MAX_FRAMES_IN_FLIGHT};
use swapchain::Swapchain;

pub struct Renderer {
    render_pass: vk::RenderPass,
    swapchain: Swapchain,
    frames: [FrameData; MAX_FRAMES_IN_FLIGHT],
    current_frame: usize,
    surface_width: u32,
    surface_height: u32,
    swapchain_dirty: bool,
    // context must be last: Rust drops fields in declaration order,
    // so it's destroyed after all child Vulkan objects above.
    context: VulkanContext,
}

impl Renderer {
    pub fn new(window: &Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let context = VulkanContext::new(window.as_ref())?;

        // Query the surface format before creating the render pass — the
        // render pass must match the swapchain format exactly.
        let surface_format = swapchain::query_surface_format(&context)?;
        let render_pass =
            pipeline::create_render_pass(&context.device, surface_format.format)?;

        let swapchain =
            Swapchain::new(&context, render_pass, size.width, size.height)?;

        let frames = [
            create_frame(&context.device, context.graphics_queue_family)?,
            create_frame(&context.device, context.graphics_queue_family)?,
        ];

        Ok(Self {
            render_pass,
            swapchain,
            frames,
            current_frame: 0,
            surface_width: size.width,
            surface_height: size.height,
            swapchain_dirty: false,
            context,
        })
    }

    /// Called when the window is resized. Marks the swapchain for recreation
    /// at the start of the next draw_frame.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_width = width;
        self.surface_height = height;
        self.swapchain_dirty = true;
    }

    pub fn draw_frame(&mut self) -> Result<()> {
        // Skip rendering while the window is minimized.
        if self.surface_width == 0 || self.surface_height == 0 {
            return Ok(());
        }

        if self.swapchain_dirty {
            unsafe { self.context.device.device_wait_idle()? };
            self.swapchain.recreate(
                &self.context,
                self.render_pass,
                self.surface_width,
                self.surface_height,
            )?;
            self.swapchain_dirty = false;
        }

        let frame = &self.frames[self.current_frame];

        // Wait for the previous use of this frame slot to finish.
        unsafe {
            self.context.device.wait_for_fences(
                std::slice::from_ref(&frame.in_flight),
                true,
                u64::MAX,
            )?;
        }

        // Acquire the next swapchain image.
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

        // Only reset the fence after we know we're going to submit work.
        unsafe {
            self.context
                .device
                .reset_fences(std::slice::from_ref(&frame.in_flight))?;
        }

        // Record: clear-color render pass.
        let cb = frame.command_buffer;
        unsafe {
            self.context.device.reset_command_pool(
                frame.command_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.context.device.begin_command_buffer(cb, &begin_info)?;

            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.118, 0.118, 0.180, 1.0], // #1E1E2E catppuccin base
                },
            }];

            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.swapchain.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent,
                })
                .clear_values(&clear_values);

            self.context.device.cmd_begin_render_pass(
                cb,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            self.context.device.cmd_end_render_pass(cb);
            self.context.device.end_command_buffer(cb)?;
        }

        // render_finished is indexed by swapchain image (not frame slot) so that
        // we never signal a semaphore still held by the presentation engine.
        let render_finished = self.swapchain.render_finished[image_index as usize];

        // Submit.
        let wait_semaphores = [frame.image_available];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [cb];
        let signal_semaphores = [render_finished];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.context.device.queue_submit(
                self.context.graphics_queue,
                std::slice::from_ref(&submit_info),
                frame.in_flight,
            )?;
        }

        // Present.
        let swapchains = [self.swapchain.handle];
        let indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&indices);

        match unsafe {
            self.swapchain
                .loader
                .queue_present(self.context.graphics_queue, &present_info)
        } {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.swapchain_dirty = true;
            }
            Ok(false) => {}
            Err(e) => return Err(e.into()),
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.device_wait_idle().ok();

            for frame in &self.frames {
                self.context
                    .device
                    .destroy_semaphore(frame.image_available, None);
                self.context.device.destroy_fence(frame.in_flight, None);
                self.context
                    .device
                    .destroy_command_pool(frame.command_pool, None);
            }

            self.swapchain.destroy(&self.context);
            self.context.device.destroy_render_pass(self.render_pass, None);
        }
        // context drops here — destroys device, surface, instance in VulkanContext::drop
    }
}
