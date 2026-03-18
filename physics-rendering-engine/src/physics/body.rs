use rapier3d::dynamics::RigidBodyHandle;
use rapier3d::geometry::ColliderHandle;

// Phase 6: factory functions for creating PhysicsBody instances

pub struct PhysicsBody {
    pub rigid_body: RigidBodyHandle,
    pub collider: ColliderHandle,
}
