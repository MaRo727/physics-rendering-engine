pub mod context;
pub mod frame;
pub mod mesh;
pub mod pipeline;
pub mod swapchain;

use anyhow::Result;
use ash::vk;
use glam::Mat4;
use std::sync::Arc;
use winit::window::Window;

use context::VulkanContext;
use frame::{create_frame, FrameData, MAX_FRAMES_IN_FLIGHT};
use mesh::Mesh;
use pipeline::Pipeline;
use swapchain::Swapchain;

pub struct Renderer {
    render_pass: vk::RenderPass,
    pipeline: Pipeline,
    mesh: Mesh,
    swapchain: Swapchain,
    frames: [FrameData; MAX_FRAMES_IN_FLIGHT],
    current_frame: usize,
    depth_format: vk::Format,
    surface_width: u32,
    surface_height: u32,
    swapchain_dirty: bool,
    // context must be last — destroyed after all child Vulkan objects.
    context: VulkanContext,
}

impl Renderer {
    pub fn new(window: &Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let context = VulkanContext::new(window.as_ref())?;

        let surface_format = swapchain::query_surface_format(&context)?;
        let depth_format =
            pipeline::find_depth_format(&context.instance, context.physical_device);

        let render_pass = pipeline::create_render_pass(
            &context.device,
            surface_format.format,
            depth_format,
        )?;
        let pipeline = pipeline::create_graphics_pipeline(&context.device, render_pass)?;

        let swapchain = Swapchain::new(
            &context,
            render_pass,
            depth_format,
            size.width,
            size.height,
        )?;

        let frames = [
            create_frame(&context.device, context.graphics_queue_family)?,
            create_frame(&context.device, context.graphics_queue_family)?,
        ];

        let (vertices, indices) = mesh::cube();
        let mesh = Mesh::new(&context, &vertices, &indices)?;

        Ok(Self {
            render_pass,
            pipeline,
            mesh,
            swapchain,
            frames,
            current_frame: 0,
            depth_format,
            surface_width: size.width,
            surface_height: size.height,
            swapchain_dirty: false,
            context,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_width = width;
        self.surface_height = height;
        self.swapchain_dirty = true;
    }

    pub fn draw_frame(&mut self, transforms: &[Mat4]) -> Result<()> {
        if self.surface_width == 0 || self.surface_height == 0 {
            return Ok(());
        }

        if self.swapchain_dirty {
            unsafe { self.context.device.device_wait_idle()? };
            self.swapchain.recreate(
                &self.context,
                self.render_pass,
                self.depth_format,
                self.surface_width,
                self.surface_height,
            )?;
            self.swapchain_dirty = false;
        }

        let frame = &self.frames[self.current_frame];

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
            self.context
                .device
                .reset_fences(std::slice::from_ref(&frame.in_flight))?;
        }

        let cb = frame.command_buffer;
        unsafe {
            self.context.device.reset_command_pool(
                frame.command_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;

            self.context.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.118, 0.118, 0.180, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            self.context.device.cmd_begin_render_pass(
                cb,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.swapchain.framebuffers[image_index as usize])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.swapchain.extent,
                    })
                    .clear_values(&clear_values),
                vk::SubpassContents::INLINE,
            );

            let extent = self.swapchain.extent;
            self.context.device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.handle,
            );
            self.context.device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.context.device.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }],
            );
            self.context.device.cmd_bind_vertex_buffers(
                cb,
                0,
                &[self.mesh.vertex_buffer],
                &[0],
            );
            self.context.device.cmd_bind_index_buffer(
                cb,
                self.mesh.index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            for transform in transforms {
                let bytes: &[u8] = std::slice::from_raw_parts(
                    transform as *const Mat4 as *const u8,
                    std::mem::size_of::<Mat4>(),
                );
                self.context.device.cmd_push_constants(
                    cb,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytes,
                );
                self.context
                    .device
                    .cmd_draw_indexed(cb, self.mesh.index_count, 1, 0, 0, 0);
            }

            self.context.device.cmd_end_render_pass(cb);
            self.context.device.end_command_buffer(cb)?;
        }

        let render_finished = self.swapchain.render_finished[image_index as usize];
        let wait_semaphores = [frame.image_available];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [cb];
        let signal_semaphores = [render_finished];

        unsafe {
            self.context.device.queue_submit(
                self.context.graphics_queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&wait_stages)
                    .command_buffers(&command_buffers)
                    .signal_semaphores(&signal_semaphores)],
                frame.in_flight,
            )?;
        }

        match unsafe {
            self.swapchain
                .loader
                .queue_present(self.context.graphics_queue, &vk::PresentInfoKHR::default()
                    .wait_semaphores(&signal_semaphores)
                    .swapchains(&[self.swapchain.handle])
                    .image_indices(&[image_index]))
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
        unsafe {
            self.context.device.device_wait_idle().ok();

            for frame in &self.frames {
                self.context.device.destroy_semaphore(frame.image_available, None);
                self.context.device.destroy_fence(frame.in_flight, None);
                self.context.device.destroy_command_pool(frame.command_pool, None);
            }

            self.mesh.destroy(&self.context.device);
            self.swapchain.destroy(&self.context);
            self.context.device.destroy_pipeline(self.pipeline.handle, None);
            self.context.device.destroy_pipeline_layout(self.pipeline.layout, None);
            self.context.device.destroy_render_pass(self.render_pass, None);
        }
    }
}
