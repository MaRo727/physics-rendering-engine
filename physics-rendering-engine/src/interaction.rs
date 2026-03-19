use glam::Vec3;
use rapier3d::prelude::RigidBodyHandle;

use crate::physics::body::WeightClass;
use crate::physics::world::PhysicsWorld;

const PICKUP_RANGE: f32 = 5.0;
const HOLD_DISTANCE: f32 = 3.0;
const HOLD_STIFFNESS: f32 = 20.0;
const PUNCH_RANGE: f32 = 3.0;
const BARE_PUNCH_FORCE: f32 = 8.0;

use crate::scene::WorldObject;

/// Manages picking up, holding, throwing, and punching objects.
pub struct Interaction {
    pub held_body: Option<RigidBodyHandle>,
    interact_prev: bool,
    punch_prev: bool,
}

impl Default for Interaction {
    fn default() -> Self {
        Self {
            held_body: None,
            interact_prev: false,
            punch_prev: false,
        }
    }
}

impl Interaction {
    /// Drop the currently held object (if any), re-enabling gravity.
    pub fn drop_held(&mut self, physics: &mut PhysicsWorld) {
        if let Some(held) = self.held_body.take() {
            physics.set_gravity_enabled(held, true);
        }
    }

    /// Process interact (E) and throw/punch (LMB) input for this frame.
    pub fn update(
        &mut self,
        physics: &mut PhysicsWorld,
        objects: &[WorldObject],
        eye: Vec3,
        look_dir: Vec3,
        interact_pressed: bool,
        throw_pressed: bool,
        player_collider: rapier3d::prelude::ColliderHandle,
    ) {
        // --- Interact (E) — edge-triggered pickup / drop ---
        let interact_edge = interact_pressed && !self.interact_prev;
        self.interact_prev = interact_pressed;

        if interact_edge {
            if let Some(held) = self.held_body.take() {
                physics.set_gravity_enabled(held, true);
            } else {
                let hit = physics.cast_ray(eye, look_dir, PICKUP_RANGE, player_collider);
                if let Some(handle) = hit {
                    if physics.is_dynamic(handle) {
                        self.held_body = Some(handle);
                        physics.set_gravity_enabled(handle, false);
                    }
                }
            }
        }

        // --- LMB: throw held object, or punch ---
        let lmb_edge = throw_pressed && !self.punch_prev;
        self.punch_prev = throw_pressed;

        if lmb_edge {
            if let Some(held) = self.held_body.take() {
                let throw_speed = weight_class_of(objects, held).throw_speed();
                physics.set_gravity_enabled(held, true);
                physics.set_body_linvel(held, look_dir * throw_speed);
            } else {
                let hit = physics.cast_ray(eye, look_dir, PUNCH_RANGE, player_collider);
                if let Some(target_body) = hit {
                    if physics.is_dynamic(target_body) {
                        let wc = weight_class_of(objects, target_body);
                        let force = look_dir * BARE_PUNCH_FORCE * wc.punch_knockback();
                        physics.apply_impulse(target_body, force);
                    }
                }
            }
        }

        // --- Hold: steer held object toward target point ---
        if let Some(held) = self.held_body {
            let target = eye + look_dir * HOLD_DISTANCE;
            let obj_pos = physics.body_position(held);
            let delta = target - obj_pos;
            physics.set_body_linvel(held, delta * HOLD_STIFFNESS);
        }
    }
}

fn weight_class_of(objects: &[WorldObject], handle: RigidBodyHandle) -> WeightClass {
    for obj in objects {
        if obj.body.rigid_body == handle {
            return obj.body.weight_class;
        }
    }
    WeightClass::Medium
}
