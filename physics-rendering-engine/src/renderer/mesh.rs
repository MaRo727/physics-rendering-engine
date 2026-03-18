use glam::Vec3;

// Phase 5: GPU vertex/index buffers, upload helpers

#[repr(C)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
}
