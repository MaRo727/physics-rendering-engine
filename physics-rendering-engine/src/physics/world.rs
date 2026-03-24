use glam::{Mat4, Quat, Vec3};
use rapier3d::prelude::*;
use rapier3d::geometry::Ray;
use smallvec::SmallVec;

/// Maximum physics timestep — clamp to avoid instability on frame spikes
/// while keeping exactly one step per frame (the game sets velocities
/// directly each frame and expects a single integration step).
const MAX_DT: f32 = 1.0 / 30.0;

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
        self.integration_parameters.dt = dt.min(MAX_DT);
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

    pub fn set_body_position(&mut self, handle: RigidBodyHandle, pos: Vec3) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.set_translation(vector![pos.x, pos.y, pos.z], true);
        }
    }

    pub fn set_body_rotation(&mut self, handle: RigidBodyHandle, rotation: Quat) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            let r = rapier3d::na::UnitQuaternion::new_normalize(
                rapier3d::na::Quaternion::new(rotation.w, rotation.x, rotation.y, rotation.z),
            );
            body.set_rotation(r, true);
        }
    }

    pub fn body_linvel_y(&self, handle: RigidBodyHandle) -> f32 {
        self.rigid_body_set[handle].linvel().y
    }

    /// Return the full linear velocity as a Vec3.
    pub fn body_linvel_xz(&self, handle: RigidBodyHandle) -> Vec3 {
        let v = self.rigid_body_set[handle].linvel();
        Vec3::new(v.x, v.y, v.z)
    }

    pub fn set_body_linvel(&mut self, handle: RigidBodyHandle, vel: Vec3) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.set_linvel(vector![vel.x, vel.y, vel.z], true);
        }
    }

    /// Returns true if two colliders are actively in contact.
    pub fn are_colliders_in_contact(&self, a: ColliderHandle, b: ColliderHandle) -> bool {
        self.narrow_phase.contact_pair(a, b)
            .map_or(false, |pair| pair.has_any_active_contact)
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

    /// Collect wall contact normals (horizontal-ish surfaces the player is pressed against).
    /// Returns outward-pointing normals for contacts with normal Y ≤ 0.7 (i.e. not ground).
    pub fn wall_normals(&self, collider: ColliderHandle) -> SmallVec<[Vec3; 4]> {
        let mut normals = SmallVec::new();
        for pair in self.narrow_phase.contact_pairs_with(collider) {
            for manifold in &pair.manifolds {
                if !manifold.points.iter().any(|pt| pt.dist <= 0.02) {
                    continue;
                }
                let (nx, ny, nz) = if pair.collider1 == collider {
                    (-manifold.local_n1.x, -manifold.local_n1.y, -manifold.local_n1.z)
                } else {
                    (-manifold.local_n2.x, -manifold.local_n2.y, -manifold.local_n2.z)
                };
                // Skip ground contacts (those are fine).
                if ny <= 0.7 {
                    normals.push(Vec3::new(nx, 0.0, nz).normalize_or_zero());
                }
            }
        }
        normals
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

    /// Apply an impulse (instantaneous velocity change scaled by mass) to a body.
    pub fn apply_impulse(&mut self, handle: RigidBodyHandle, impulse: Vec3) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.apply_impulse(vector![impulse.x, impulse.y, impulse.z], true);
        }
    }

    /// Apply a force (accumulated over the next timestep, not divided by mass) to a body.
    pub fn apply_force(&mut self, handle: RigidBodyHandle, force: Vec3) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.add_force(vector![force.x, force.y, force.z], true);
        }
    }

    /// Get the mass of a rigid body.
    pub fn body_mass(&self, handle: RigidBodyHandle) -> f32 {
        self.rigid_body_set[handle].mass()
    }

    /// Add a fixed (static) rigid body with a box collider. Returns handles.
    pub fn add_static_box(
        &mut self,
        position: Vec3,
        half_extents: Vec3,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let rb = RigidBodyBuilder::fixed()
            .translation(vector![position.x, position.y, position.z])
            .build();
        let rb_handle = self.rigid_body_set.insert(rb);

        let col = ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z)
            .build();
        let col_handle =
            self.collider_set
                .insert_with_parent(col, rb_handle, &mut self.rigid_body_set);

        (rb_handle, col_handle)
    }

    /// Add a fixed rigid body with an arbitrary shared shape. Returns handles.
    pub fn add_static_shape(
        &mut self,
        position: Vec3,
        shape: SharedShape,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let rb = RigidBodyBuilder::fixed()
            .translation(vector![position.x, position.y, position.z])
            .build();
        let rb_handle = self.rigid_body_set.insert(rb);

        let col = ColliderBuilder::new(shape).build();
        let col_handle =
            self.collider_set
                .insert_with_parent(col, rb_handle, &mut self.rigid_body_set);

        (rb_handle, col_handle)
    }

    /// Add a dynamic rigid body with an arbitrary shared shape. Returns handles.
    pub fn add_dynamic_shape(
        &mut self,
        position: Vec3,
        shape: SharedShape,
        mass: f32,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let rb = RigidBodyBuilder::dynamic()
            .translation(vector![position.x, position.y, position.z])
            .additional_mass(mass)
            .build();
        let rb_handle = self.rigid_body_set.insert(rb);

        let col = ColliderBuilder::new(shape)
            .restitution(0.2)
            .build();
        let col_handle =
            self.collider_set
                .insert_with_parent(col, rb_handle, &mut self.rigid_body_set);

        (rb_handle, col_handle)
    }

    /// Add a compound static collider made of multiple cuboids on a single fixed body.
    /// Returns the rigid body handle so callers can identify hits on this body.
    pub fn add_compound_static(&mut self, boxes: &[(Vec3, Vec3)]) -> RigidBodyHandle {
        let rb = RigidBodyBuilder::fixed().build();
        let rb_handle = self.rigid_body_set.insert(rb);

        let shapes: Vec<_> = boxes.iter().map(|(pos, half)| {
            let iso = rapier3d::na::Isometry3::translation(pos.x, pos.y, pos.z);
            (iso, SharedShape::cuboid(half.x, half.y, half.z))
        }).collect();

        let col = ColliderBuilder::compound(shapes).build();
        self.collider_set
            .insert_with_parent(col, rb_handle, &mut self.rigid_body_set);
        rb_handle
    }

    /// Add a trimesh collider on a fixed body.
    pub fn add_trimesh(
        &mut self,
        vertices: Vec<Vec3>,
        triangles: Vec<[u32; 3]>,
    ) {
        let points: Vec<_> = vertices
            .iter()
            .map(|v| point![v.x, v.y, v.z])
            .collect();

        let rb = RigidBodyBuilder::fixed().build();
        let rb_handle = self.rigid_body_set.insert(rb);

        let col = ColliderBuilder::trimesh(points, triangles).build();
        self.collider_set
            .insert_with_parent(col, rb_handle, &mut self.rigid_body_set);
    }

    // -----------------------------------------------------------------------
    // Heightfield collider (for deformable terrain)
    // -----------------------------------------------------------------------

    /// Add a heightfield collider on a fixed body centered at origin.
    /// `heights` layout: heights\[row * ncols + col\], row = Z, col = X.
    /// `scale` is the total world-space extent (e.g. (900, 1, 900)).
    pub fn add_heightfield(
        &mut self,
        heights: &[f32],
        nrows: usize,
        ncols: usize,
        scale: Vec3,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let mat = rapier3d::na::DMatrix::from_fn(nrows, ncols, |i, j| {
            heights[i * ncols + j]
        });
        let rb = RigidBodyBuilder::fixed().build();
        let rb_handle = self.rigid_body_set.insert(rb);
        let col = ColliderBuilder::heightfield(mat, vector![scale.x, scale.y, scale.z])
            .build();
        let col_handle = self.collider_set
            .insert_with_parent(col, rb_handle, &mut self.rigid_body_set);
        (rb_handle, col_handle)
    }

    /// Replace the shape of an existing heightfield collider with updated heights.
    pub fn update_heightfield(
        &mut self,
        col: ColliderHandle,
        heights: &[f32],
        nrows: usize,
        ncols: usize,
        scale: Vec3,
    ) {
        if let Some(collider) = self.collider_set.get_mut(col) {
            let mat = rapier3d::na::DMatrix::from_fn(nrows, ncols, |i, j| {
                heights[i * ncols + j]
            });
            let shape = SharedShape::heightfield(mat, vector![scale.x, scale.y, scale.z]);
            collider.set_shape(shape);
        }
    }

    /// Add a small heightfield chunk collider on a fixed body at (center_x, 0, center_z).
    pub fn add_heightfield_chunk(
        &mut self,
        heights: &[f32],
        nrows: usize,
        ncols: usize,
        scale: Vec3,
        center_x: f32,
        center_z: f32,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let mat = rapier3d::na::DMatrix::from_fn(nrows, ncols, |i, j| {
            heights[i * ncols + j]
        });
        let rb = RigidBodyBuilder::fixed()
            .translation(vector![center_x, 0.0, center_z])
            .build();
        let rb_handle = self.rigid_body_set.insert(rb);
        let col = ColliderBuilder::heightfield(mat, vector![scale.x, scale.y, scale.z])
            .build();
        let col_handle = self.collider_set
            .insert_with_parent(col, rb_handle, &mut self.rigid_body_set);
        (rb_handle, col_handle)
    }

    /// Update a single heightfield chunk collider's shape.
    pub fn update_heightfield_chunk(
        &mut self,
        col: ColliderHandle,
        heights: &[f32],
        nrows: usize,
        ncols: usize,
        scale: Vec3,
    ) {
        if let Some(collider) = self.collider_set.get_mut(col) {
            let mat = rapier3d::na::DMatrix::from_fn(nrows, ncols, |i, j| {
                heights[i * ncols + j]
            });
            let shape = SharedShape::heightfield(mat, vector![scale.x, scale.y, scale.z]);
            collider.set_shape(shape);
        }
    }

    // -----------------------------------------------------------------------
    // Raycasting
    // -----------------------------------------------------------------------

    /// Cast a ray and return (RigidBodyHandle, hit_position, surface_normal).
    pub fn cast_ray_full(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: f32,
        exclude: ColliderHandle,
    ) -> Option<(RigidBodyHandle, Vec3, Vec3)> {
        let ray = Ray::new(
            point![origin.x, origin.y, origin.z],
            vector![direction.x, direction.y, direction.z],
        );
        let filter = QueryFilter::default().exclude_collider(exclude);
        self.query_pipeline
            .cast_ray_and_get_normal(
                &self.rigid_body_set,
                &self.collider_set,
                &ray,
                max_toi,
                true,
                filter,
            )
            .and_then(|(collider_handle, intersection)| {
                let body = self.collider_set[collider_handle].parent()?;
                let hit_pos = origin + direction * intersection.time_of_impact;
                let n = intersection.normal;
                Some((body, hit_pos, Vec3::new(n.x, n.y, n.z)))
            })
    }

    /// Raycast without excluding any collider.
    pub fn cast_ray_unfiltered(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: f32,
    ) -> Option<(RigidBodyHandle, Vec3, Vec3)> {
        let ray = Ray::new(
            point![origin.x, origin.y, origin.z],
            vector![direction.x, direction.y, direction.z],
        );
        let filter = QueryFilter::default();
        self.query_pipeline
            .cast_ray_and_get_normal(
                &self.rigid_body_set,
                &self.collider_set,
                &ray,
                max_toi,
                true,
                filter,
            )
            .and_then(|(collider_handle, intersection)| {
                let body = self.collider_set[collider_handle].parent()?;
                let hit_pos = origin + direction * intersection.time_of_impact;
                let n = intersection.normal;
                Some((body, hit_pos, Vec3::new(n.x, n.y, n.z)))
            })
    }

    /// Replace a collider's shape on an existing rigid body. Returns the new collider handle.
    pub fn replace_collider(
        &mut self,
        rb: RigidBodyHandle,
        old_col: ColliderHandle,
        shape: SharedShape,
    ) -> ColliderHandle {
        self.collider_set.remove(
            old_col,
            &mut self.island_manager,
            &mut self.rigid_body_set,
            true,
        );
        let col = ColliderBuilder::new(shape).build();
        self.collider_set
            .insert_with_parent(col, rb, &mut self.rigid_body_set)
    }

    /// Remove a rigid body and its collider from the world.
    pub fn remove_body(&mut self, rb: RigidBodyHandle, col: ColliderHandle) {
        self.collider_set.remove(
            col,
            &mut self.island_manager,
            &mut self.rigid_body_set,
            true,
        );
        self.rigid_body_set.remove(
            rb,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            true,
        );
    }

    /// Cast a ray and return the distance to the first hit (excluding `exclude`).
    pub fn cast_ray_distance(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: f32,
        exclude: ColliderHandle,
    ) -> Option<f32> {
        let ray = Ray::new(
            point![origin.x, origin.y, origin.z],
            vector![direction.x, direction.y, direction.z],
        );
        let filter = QueryFilter::default().exclude_collider(exclude);
        self.query_pipeline
            .cast_ray(
                &self.rigid_body_set,
                &self.collider_set,
                &ray,
                max_toi,
                true,
                filter,
            )
            .map(|(_handle, toi)| toi)
    }

    /// Cast a ray and return hit position + surface normal (excluding `exclude`).
    pub fn cast_ray_detailed(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_toi: f32,
        exclude: ColliderHandle,
    ) -> Option<(Vec3, Vec3)> {
        let ray = Ray::new(
            point![origin.x, origin.y, origin.z],
            vector![direction.x, direction.y, direction.z],
        );
        let filter = QueryFilter::default().exclude_collider(exclude);
        self.query_pipeline
            .cast_ray_and_get_normal(
                &self.rigid_body_set,
                &self.collider_set,
                &ray,
                max_toi,
                true,
                filter,
            )
            .map(|(_handle, intersection)| {
                let hit_pos = origin + direction * intersection.time_of_impact;
                let n = intersection.normal;
                (hit_pos, Vec3::new(n.x, n.y, n.z))
            })
    }
}
