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
    pub fn from_range(
        context: &VulkanContext,
        mesh: &Mesh,
        info: &super::mesh::SubMeshInfo,
    ) -> Result<Self> {
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
    instance_count: u32,
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
            instance_count,
        );

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD)
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
            instance_count,
        })
    }

    /// Write new instance transforms and record TLAS build/update into `cb`.
    pub fn update(
        &mut self,
        context: &VulkanContext,
        cb: vk::CommandBuffer,
        instances: &[vk::AccelerationStructureInstanceKHR],
    ) {
        assert!(instances.len() <= self.instance_count as usize);

        // Write instances to the persistently-mapped buffer.
        unsafe {
            self.instance_mapped
                .copy_from_nonoverlapping(instances.as_ptr(), instances.len());
        }

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

        let geometry = make_instance_geometry(instance_address, instances.len() as u32);

        // Always full BUILD — instance count can change between frames due
        // to frustum culling, and UPDATE requires the same primitive count.
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.handle)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_address });

        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instances.len() as u32);

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

fn make_instance_geometry(
    instance_address: vk::DeviceAddress,
    instance_count: u32,
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
