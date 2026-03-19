use glam::Vec3;
use rapier3d::prelude::RigidBodyHandle;

use crate::physics::body::WeightClass;
use crate::physics::world::PhysicsWorld;

const PICKUP_RANGE: f32 = 5.0;
const HOLD_DISTANCE: f32 = 3.0;
const HOLD_STIFFNESS: f32 = 20.0;
const PUNCH_RANGE: f32 = 3.0;
const BARE_PUNCH_FORCE: f32 = 8.0;
const PRY_DURATION: f32 = 1.0;
const AXE_RANGE: f32 = 4.0;

use crate::scene::WorldObject;

// ---------------------------------------------------------------------------
// Tool types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToolType {
    Hands,
    Axe,
}

impl ToolType {
    pub fn next(self) -> Self {
        match self {
            ToolType::Hands => ToolType::Axe,
            ToolType::Axe => ToolType::Hands,
        }
    }

    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            ToolType::Hands => "Hands",
            ToolType::Axe => "Axe",
        }
    }
}

// ---------------------------------------------------------------------------
// Interaction result
// ---------------------------------------------------------------------------

pub struct InteractionResult {
    pub pried_cell: Option<(i32, i32, i32)>,
    /// Body handle of an object hit by the axe swing.
    pub axe_hit: Option<RigidBodyHandle>,
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

/// Manages picking up, holding, throwing, punching, and prying building cubes.
pub struct Interaction {
    pub held_body: Option<RigidBodyHandle>,
    pub equipped_tool: ToolType,
    interact_prev: bool,
    punch_prev: bool,
    cycle_prev: bool,
    pry_timer: f32,
    pry_target: Option<(i32, i32, i32)>,
}

impl Default for Interaction {
    fn default() -> Self {
        Self {
            held_body: None,
            equipped_tool: ToolType::Hands,
            interact_prev: false,
            punch_prev: false,
            cycle_prev: false,
            pry_timer: 0.0,
            pry_target: None,
        }
    }
}

impl Interaction {
    /// Drop the currently held object (if any), re-enabling gravity.
    pub fn drop_held(&mut self, physics: &mut PhysicsWorld) {
        if let Some(held) = self.held_body.take() {
            physics.set_gravity_enabled(held, true);
        }
        self.pry_timer = 0.0;
        self.pry_target = None;
    }

    /// Cycle to the next equipped tool. Returns true if tool changed.
    pub fn cycle_tool(&mut self, pressed: bool) -> bool {
        let edge = pressed && !self.cycle_prev;
        self.cycle_prev = pressed;
        if edge {
            self.equipped_tool = self.equipped_tool.next();
            true
        } else {
            false
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
        dt: f32,
        building_cell_aimed_at: Option<(i32, i32, i32)>,
    ) -> InteractionResult {
        let interact_edge = interact_pressed && !self.interact_prev;
        self.interact_prev = interact_pressed;

        let mut result = InteractionResult {
            pried_cell: None,
            axe_hit: None,
        };

        // --- E press (edge): drop held OR pick up dynamic object OR start prying ---
        if interact_edge {
            if let Some(held) = self.held_body.take() {
                // Drop held object.
                physics.set_gravity_enabled(held, true);
                self.pry_timer = 0.0;
                self.pry_target = None;
            } else {
                // Try to pick up a dynamic object.
                let hit = physics.cast_ray(eye, look_dir, PICKUP_RANGE, player_collider);
                if let Some(handle) = hit {
                    if physics.is_dynamic(handle) {
                        self.held_body = Some(handle);
                        physics.set_gravity_enabled(handle, false);
                        self.pry_timer = 0.0;
                        self.pry_target = None;
                    }
                }
                // If we didn't pick up anything dynamic but aimed at a building cell,
                // start the pry timer (it will accumulate while E is held).
                if self.held_body.is_none() {
                    if let Some(cell) = building_cell_aimed_at {
                        self.pry_target = Some(cell);
                        self.pry_timer = 0.0;
                    }
                }
            }
        }

        // --- Hold E to pry a building cube ---
        if interact_pressed && self.held_body.is_none() && self.pry_target.is_some() {
            if building_cell_aimed_at == self.pry_target {
                self.pry_timer += dt;
                if self.pry_timer >= PRY_DURATION {
                    result.pried_cell = self.pry_target.take();
                    self.pry_timer = 0.0;
                }
            } else {
                // Looked away — reset.
                self.pry_timer = 0.0;
                self.pry_target = None;
            }
        }

        // E released before threshold — reset pry.
        if !interact_pressed {
            self.pry_timer = 0.0;
            self.pry_target = None;
        }

        // --- LMB: throw held object, or punch/axe ---
        let lmb_edge = throw_pressed && !self.punch_prev;
        self.punch_prev = throw_pressed;

        if lmb_edge {
            if let Some(held) = self.held_body.take() {
                let throw_speed = weight_class_of(objects, held).throw_speed();
                physics.set_gravity_enabled(held, true);
                physics.set_body_linvel(held, look_dir * throw_speed);
            } else {
                match self.equipped_tool {
                    ToolType::Hands => {
                        // Bare-fist punch.
                        let hit = physics.cast_ray(eye, look_dir, PUNCH_RANGE, player_collider);
                        if let Some(target_body) = hit {
                            if physics.is_dynamic(target_body) {
                                let wc = weight_class_of(objects, target_body);
                                let force = look_dir * BARE_PUNCH_FORCE * wc.punch_knockback();
                                physics.apply_impulse(target_body, force);
                            }
                        }
                    }
                    ToolType::Axe => {
                        // Axe swing — report hit target for engine to split.
                        let hit = physics.cast_ray(eye, look_dir, AXE_RANGE, player_collider);
                        if let Some(target_body) = hit {
                            if physics.is_dynamic(target_body) {
                                result.axe_hit = Some(target_body);
                            }
                        }
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

        result
    }

    /// Progress of the current pry action (0.0 to 1.0). 0 when not prying.
    pub fn pry_progress(&self) -> f32 {
        if self.pry_target.is_some() {
            self.pry_timer / PRY_DURATION
        } else {
            0.0
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
