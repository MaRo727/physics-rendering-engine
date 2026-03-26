use anyhow::{Context, Result};
use ash::vk;
use glam::Mat4;

use super::context::VulkanContext;
use super::mesh::{self, Mesh, create_buffer};

// ---------------------------------------------------------------------------
// BLAS — built once from a Mesh
// ---------------------------------------------------------------------------

pub struct Blas {
    pub handle: vk::AccelerationStructureKHR,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub device_address: vk::DeviceAddress,
}

impl Blas {
    /// Build a BLAS from a sub-range of a combined mesh buffer.
    ///
    /// # Panics
    /// Panics if the sub-mesh has zero vertices or indices. Creating a BLAS with
    /// zero triangles leaks Vulkan buffers on shutdown and violates the spec.
    pub fn from_range(
        context: &VulkanContext,
        mesh: &Mesh,
        info: &super::mesh::SubMeshInfo,
    ) -> Result<Self> {
        assert!(
            info.vertex_count > 0 && info.index_count > 0,
            "Cannot create BLAS from empty sub-mesh (vertex_count={}, index_count={})",
            info.vertex_count,
            info.index_count,
        );

        let vertex_address = mesh::get_device_address(&context.device, mesh.vertex_buffer)
            + (info.vertex_offset as vk::DeviceSize)
                * std::mem::size_of::<super::mesh::Vertex>() as vk::DeviceSize;
        let index_address = mesh::get_device_address(&context.device, mesh.index_buffer)
            + (info.index_offset as vk::DeviceSize)
                * std::mem::size_of::<u32>() as vk::DeviceSize;
        let triangle_count = info.index_count / 3;

        Self::build(context, vertex_address, index_address, info.vertex_count, triangle_count)
    }

    pub fn new(context: &VulkanContext, mesh: &Mesh) -> Result<Self> {
        let vertex_address = mesh::get_device_address(&context.device, mesh.vertex_buffer);
        let index_address = mesh::get_device_address(&context.device, mesh.index_buffer);
        let triangle_count = mesh.index_count / 3;

        Self::build(context, vertex_address, index_address, mesh.vertex_count, triangle_count)
    }

    fn build(
        context: &VulkanContext,
        vertex_address: vk::DeviceAddress,
        index_address: vk::DeviceAddress,
        vertex_count: u32,
        triangle_count: u32,
    ) -> Result<Self> {
        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vertex_address })
            .vertex_stride(std::mem::size_of::<super::mesh::Vertex>() as vk::DeviceSize)
            .max_vertex(vertex_count - 1)
            .index_type(vk::IndexType::UINT32)
            .index_data(vk::DeviceOrHostAddressConstKHR { device_address: index_address });

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geometry));

        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            context.accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[triangle_count],
                &mut sizes,
            )
        };

        // Allocate scratch buffer.
        let (scratch_buffer, scratch_memory) = create_buffer(
            &context.device,
            &context.memory_properties,
            sizes.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;
        let scratch_address = mesh::get_device_address(&context.device, scratch_buffer);

        // Allocate AS buffer.
        let (as_buffer, as_memory) = create_buffer(
            &context.device,
            &context.memory_properties,
            sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;

        let handle = unsafe {
            context.accel_loader.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(as_buffer)
                    .size(sizes.acceleration_structure_size)
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
                None,
            )
        }
        .context("Failed to create BLAS")?;

        // Build BLAS in a one-shot command buffer.
        let build_info_with_scratch = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(handle)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_address });

        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(triangle_count);

        one_shot(context, |cb| unsafe {
            context.accel_loader.cmd_build_acceleration_structures(
                cb,
                std::slice::from_ref(&build_info_with_scratch),
                &[std::slice::from_ref(&range)],
            );

            // Barrier: BLAS write → BLAS read.
            context.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                    .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)],
                &[],
                &[],
            );
        })?;

        // Free scratch.
        unsafe {
            context.device.destroy_buffer(scratch_buffer, None);
            context.device.free_memory(scratch_memory, None);
        }

        let device_address = unsafe {
            context.accel_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(handle),
            )
        };

        Ok(Self { handle, buffer: as_buffer, memory: as_memory, device_address })
    }

    pub fn destroy(&self, context: &VulkanContext) {
        unsafe {
            context.accel_loader.destroy_acceleration_structure(self.handle, None);
            context.device.destroy_buffer(self.buffer, None);
            context.device.free_memory(self.memory, None);
        }
    }
}

// ---------------------------------------------------------------------------
// TLAS — rebuilt every frame with instances
// ---------------------------------------------------------------------------

pub struct Tlas {
    pub handle: vk::AccelerationStructureKHR,
    as_buffer: vk::Buffer,
    as_memory: vk::DeviceMemory,
    pub instance_buffer: vk::Buffer,
    instance_memory: vk::DeviceMemory,
    instance_mapped: *mut vk::AccelerationStructureInstanceKHR,
    scratch_buffer: vk::Buffer,
    scratch_memory: vk::DeviceMemory,
    /// Allocated instance capacity (buffer + AS sized for this many instances).
    capacity: u32,
    /// Whether an initial BUILD has been recorded, enabling UPDATE on subsequent frames.
    has_been_built: bool,
    /// Consecutive frames where instance count stayed below half the capacity.
    shrink_frames: u32,
    /// Frames since last full BUILD. Periodic rebuilds prevent BVH quality
    /// degradation from instance reordering caused by frustum culling.
    frames_since_build: u32,
    /// The active instance count from the previous frame. Used to avoid
    /// redundantly re-deactivating unused slots that were already cleared.
    /// Initialized to `capacity` so that the first frame after allocation
    /// fills all unused slots (uninitialized memory).
    prev_active_count: u32,
}

// SAFETY: only accessed from the render thread.
unsafe impl Send for Tlas {}

impl Tlas {
    pub fn new(context: &VulkanContext, instance_count: u32) -> Result<Self> {
        let instance_size = (instance_count as usize)
            * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>();

        // Host-visible instance buffer (written every frame).
        let (instance_buffer, instance_memory) = create_buffer(
            &context.device,
            &context.memory_properties,
            instance_size as vk::DeviceSize,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        )?;

        let instance_mapped = unsafe {
            context.device.map_memory(
                instance_memory,
                0,
                instance_size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .context("Failed to map instance buffer")?
            as *mut vk::AccelerationStructureInstanceKHR;

        // Build sizes — use BUILD mode for initial sizing.
        let geometry = make_instance_geometry(
            mesh::get_device_address(&context.device, instance_buffer),
        );

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
            )
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geometry));

        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            context.accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[instance_count],
                &mut sizes,
            )
        };

        // Persistent scratch (large enough for update too; update_scratch_size ≤ build_scratch_size).
        let scratch_size = sizes.build_scratch_size.max(sizes.update_scratch_size);
        let (scratch_buffer, scratch_memory) = create_buffer(
            &context.device,
            &context.memory_properties,
            scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;

        let (as_buffer, as_memory) = create_buffer(
            &context.device,
            &context.memory_properties,
            sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;

        let handle = unsafe {
            context.accel_loader.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(as_buffer)
                    .size(sizes.acceleration_structure_size)
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                None,
            )
        }
        .context("Failed to create TLAS")?;

        Ok(Self {
            handle,
            as_buffer,
            as_memory,
            instance_buffer,
            instance_memory,
            instance_mapped,
            scratch_buffer,
            scratch_memory,
            capacity: instance_count,
            has_been_built: false,
            shrink_frames: 0,
            frames_since_build: 0,
            prev_active_count: instance_count,
        })
    }

    /// Return a mutable slice into the persistently-mapped instance buffer.
    ///
    /// # Safety
    /// The returned slice is only valid until the next GPU submit that reads
    /// the instance buffer. The caller must ensure the GPU is not currently
    /// reading this memory (e.g. the in-flight fence has been waited on).
    pub unsafe fn mapped_instances_mut(&mut self) -> &mut [vk::AccelerationStructureInstanceKHR] {
        unsafe { std::slice::from_raw_parts_mut(self.instance_mapped, self.capacity as usize) }
    }

    /// Ensure the TLAS can hold at least `min_count` instances.
    /// Returns `true` if the TLAS was reallocated (caller must update descriptor sets).
    pub fn ensure_capacity(&mut self, context: &VulkanContext, min_count: u32) -> Result<bool> {
        if min_count < self.capacity / 2 {
            self.shrink_frames += 1;
        } else {
            self.shrink_frames = 0;
        }

        let needs_grow = min_count > self.capacity;
        // Shrink after ~2 seconds at 60 fps of sustained under-use.
        let needs_shrink = self.shrink_frames > 120 && self.capacity > 64;

        if !needs_grow && !needs_shrink {
            return Ok(false);
        }

        // Wait for all GPU work before destroying resources.
        unsafe { context.device.device_wait_idle()? };

        let new_capacity = pad_capacity(min_count);
        let new_tlas = Self::new(context, new_capacity)?;
        let old = std::mem::replace(self, new_tlas);
        old.destroy(context);

        Ok(true)
    }

    /// Record a TLAS build/update into `cb` for `count` instances that have
    /// already been written into [`mapped_instances_mut`].
    ///
    /// Unused slots (`count..capacity`) are filled with copies of active
    /// instances but with `mask = 0` so no rays can intersect them.  This
    /// keeps `primitive_count` equal to `capacity` every frame (required by
    /// the Vulkan spec for UPDATE) while distributing the padding instances
    /// across real scene positions — avoiding the degenerate zero-volume
    /// AABBs at origin that would poison the BVH structure.
    ///
    /// A periodic full BUILD (every ~60 frames) prevents BVH quality
    /// degradation from instance reordering caused by frustum culling.
    pub fn record_build(
        &mut self,
        context: &VulkanContext,
        cb: vk::CommandBuffer,
        count: u32,
    ) {
        assert!(count > 0 && count <= self.capacity);

        // Fill unused slots with mask=0 copies so they are invisible to all
        // rays but still provide the BVH builder with a valid BLAS reference
        // and transform (avoiding degenerate zero-volume AABBs at origin).
        //
        // We read ONE active instance into a stack-local template (single WC
        // read) and then write it sequentially to every inactive slot.
        // Sequential writes to write-combined mapped memory are fast; the
        // previous round-robin approach did N reads from WC memory which was
        // orders of magnitude slower over PCIe.
        //
        // Optimization: only clear slots that transitioned from active to
        // unused since last frame. Slots beyond `prev_active_count` were
        // already deactivated in a prior frame and don't need re-writing.
        // On the first frame after allocation, `prev_active_count` equals
        // `capacity`, so all unused slots `[count..capacity]` are filled
        // (the buffer contains uninitialized memory).
        let clear_end = self.prev_active_count;
        if count < clear_end {
            unsafe {
                let mut template = *self.instance_mapped;
                template.instance_custom_index_and_mask = vk::Packed24_8::new(0, 0);
                for i in count as usize..clear_end as usize {
                    std::ptr::write(self.instance_mapped.add(i), template);
                }
            }
        }
        self.prev_active_count = count;

        // Memory barrier: ensure instance writes are visible before AS build.
        unsafe {
            context.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)],
                &[],
                &[],
            );
        }

        let instance_address = mesh::get_device_address(&context.device, self.instance_buffer);
        let scratch_address = mesh::get_device_address(&context.device, self.scratch_buffer);

        let geometry = make_instance_geometry(instance_address);

        // UPDATE is much faster than BUILD but the BVH quality degrades as
        // frustum culling reorders instances into different slots.  Periodic
        // full BUILDs reconstruct an optimal tree.
        const REBUILD_INTERVAL: u32 = 60;
        let do_update = self.has_been_built && self.frames_since_build < REBUILD_INTERVAL;
        let mode = if do_update {
            vk::BuildAccelerationStructureModeKHR::UPDATE
        } else {
            vk::BuildAccelerationStructureModeKHR::BUILD
        };
        self.frames_since_build = if do_update { self.frames_since_build + 1 } else { 0 };

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
            )
            .mode(mode)
            .dst_acceleration_structure(self.handle)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_address });

        if do_update {
            build_info.src_acceleration_structure = self.handle;
        }

        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(self.capacity);

        unsafe {
            context.accel_loader.cmd_build_acceleration_structures(
                cb,
                std::slice::from_ref(&build_info),
                &[std::slice::from_ref(&range)],
            );

            // Barrier: TLAS write → ray tracing shader read.
            context.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                    .dst_access_mask(
                        vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                            | vk::AccessFlags::SHADER_READ,
                    )],
                &[],
                &[],
            );
        }

        self.has_been_built = true;
    }

    pub fn destroy(&self, context: &VulkanContext) {
        unsafe {
            context.device.unmap_memory(self.instance_memory);
            context.accel_loader.destroy_acceleration_structure(self.handle, None);
            context.device.destroy_buffer(self.as_buffer, None);
            context.device.free_memory(self.as_memory, None);
            context.device.destroy_buffer(self.instance_buffer, None);
            context.device.free_memory(self.instance_memory, None);
            context.device.destroy_buffer(self.scratch_buffer, None);
            context.device.free_memory(self.scratch_memory, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Transform conversion
// ---------------------------------------------------------------------------

pub fn mat4_to_transform(m: Mat4) -> vk::TransformMatrixKHR {
    // glam is column-major; Vulkan wants row-major 3×4.
    vk::TransformMatrixKHR {
        matrix: [
            m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x,
            m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y,
            m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z,
        ],
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Round up to the next power of two (minimum 64) for TLAS capacity.
fn pad_capacity(count: u32) -> u32 {
    count.max(64).next_power_of_two()
}

fn make_instance_geometry(
    instance_address: vk::DeviceAddress,
) -> vk::AccelerationStructureGeometryKHR<'static> {
    let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR { device_address: instance_address });

    // We need the geometry to outlive this function when embedded in a build info,
    // so we return it by value and let the caller keep it alive.
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { instances: instances_data })
        .flags(vk::GeometryFlagsKHR::empty())
}

pub fn one_shot<F: FnOnce(vk::CommandBuffer)>(
    context: &VulkanContext,
    record: F,
) -> Result<()> {
    let pool = unsafe {
        context.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(context.graphics_queue_family)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
            None,
        )
    }
    .context("Failed to create transient command pool")?;

    let cb = unsafe {
        context.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .context("Failed to allocate one-shot command buffer")?[0];

    unsafe {
        context.device.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
    }

    record(cb);

    unsafe {
        context.device.end_command_buffer(cb)?;
        let fence = context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .context("Failed to create one-shot fence")?;
        context.device.queue_submit(
            context.graphics_queue,
            &[vk::SubmitInfo::default().command_buffers(&[cb])],
            fence,
        )?;
        context.device.wait_for_fences(&[fence], true, u64::MAX)?;
        context.device.destroy_fence(fence, None);
        context.device.destroy_command_pool(pool, None);
    }

    Ok(())
}
