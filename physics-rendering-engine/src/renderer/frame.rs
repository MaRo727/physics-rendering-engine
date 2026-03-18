use anyhow::{Context, Result};
use ash::{vk, Device};

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct FrameData {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub image_available: vk::Semaphore,
    pub in_flight: vk::Fence,
}

pub fn create_frame(device: &Device, queue_family: u32) -> Result<FrameData> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
        .context("Failed to create command pool")?;

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info) }
        .context("Failed to allocate command buffer")?[0];

    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let image_available = unsafe { device.create_semaphore(&semaphore_info, None) }
        .context("Failed to create image_available semaphore")?;

    // Start signaled so the first draw_frame doesn't wait forever on an unsignaled fence.
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    let in_flight = unsafe { device.create_fence(&fence_info, None) }
        .context("Failed to create in_flight fence")?;

    Ok(FrameData {
        command_pool,
        command_buffer,
        image_available,
        in_flight,
    })
}
