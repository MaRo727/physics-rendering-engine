use glam::{Mat4, Quat, Vec3};
use rapier3d::prelude::*;
use rapier3d::geometry::Ray;

pub struct PhysicsWorld {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    gravity: Vector<Real>,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
}

impl PhysicsWorld {
    pub fn new(gravity: Vec3) -> Self {
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            gravity: vector![gravity.x, gravity.y, gravity.z],
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
        }
    }

    pub fn step(&mut self, dt: f32) {
        self.integration_parameters.dt = dt;
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        );
        self.query_pipeline.update(&self.collider_set);
    }

    /// Extract the world-space transform of a rigid body as a glam Mat4.
    pub fn body_transform(&self, handle: RigidBodyHandle) -> Mat4 {
        let body = &self.rigid_body_set[handle];
        let pos = body.position();
        let t = pos.translation.vector;
        let r = pos.rotation;
        Mat4::from_rotation_translation(
            Quat::from_xyzw(r.i, r.j, r.k, r.w),
            Vec3::new(t.x, t.y, t.z),
        )
    }

    pub fn body_position(&self, handle: RigidBodyHandle) -> Vec3 {
        let t = self.rigid_body_set[handle].position().translation.vector;
        Vec3::new(t.x, t.y, t.z)
    }

    pub fn body_linvel_y(&self, handle: RigidBodyHandle) -> f32 {
        self.rigid_body_set[handle].linvel().y
    }

    pub fn set_body_linvel(&mut self, handle: RigidBodyHandle, vel: Vec3) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.set_linvel(vector![vel.x, vel.y, vel.z], true);
        }
    }

    /// Returns true if the collider has a contact with a surface whose normal
    /// points upward (Y > 0.7), i.e. the body is standing on something.
    pub fn is_on_ground(&self, collider: ColliderHandle) -> bool {
        for pair in self.narrow_phase.contact_pairs_with(collider) {
            for manifold in &pair.manifolds {
                if manifold.points.iter().any(|pt| pt.dist <= 0.01) {
                    // The normal points from shape 1 toward shape 2.
                    // Figure out which side is ours to get the correct sign.
                    let normal_y = if pair.collider1 == collider {
                        -manifold.local_n1.y
                    } else {
                        -manifold.local_n2.y
                    };
                    if normal_y > 0.7 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Cast a ray and return the RigidBodyHandle of the first hit (excluding `exclude`).
    pub fn cast_ray(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: f32,
        exclude: ColliderHandle,
    ) -> Option<RigidBodyHandle> {
        let ray = Ray::new(
            point![origin.x, origin.y, origin.z],
            vector![direction.x, direction.y, direction.z],
        );
        let filter = QueryFilter::default()
            .exclude_collider(exclude);
        self.query_pipeline
            .cast_ray(
                &self.rigid_body_set,
                &self.collider_set,
                &ray,
                max_toi,
                true,
                filter,
            )
            .and_then(|(collider_handle, _toi)| {
                self.collider_set[collider_handle].parent()
            })
    }

    /// Enable or disable gravity for a rigid body.
    pub fn set_gravity_enabled(&mut self, handle: RigidBodyHandle, enabled: bool) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.set_gravity_scale(if enabled { 1.0 } else { 0.0 }, true);
        }
    }

    /// Returns true if the body is dynamic (not static or kinematic).
    pub fn is_dynamic(&self, handle: RigidBodyHandle) -> bool {
        self.rigid_body_set[handle].is_dynamic()
    }
}
