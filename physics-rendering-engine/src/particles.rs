/// Simple particle system — small cubes with velocity, gravity, and lifetime.

use glam::{Mat4, Vec3};

use crate::renderer::{pack_instance_id, MESH_CUBE};

const MAX_PARTICLES: usize = 512;
const GRAVITY: f32 = -9.81;
const PARTICLE_OBJECT_BASE: u32 = 0xFF00;

pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub color_id: u32, // packed into instance_id upper bits via scale trick
    pub scale: f32,
    pub lifetime: f32,
    pub age: f32,
    pub gravity_scale: f32,
}

pub struct ParticleSystem {
    particles: Vec<Particle>,
    seed: u32,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self {
            particles: Vec::with_capacity(MAX_PARTICLES),
            seed: 42,
        }
    }

    /// Cheap deterministic pseudo-random [0, 1).
    fn rand(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
        (self.seed >> 16) as f32 / 65536.0
    }

    fn rand_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.rand() * (hi - lo)
    }

    fn rand_dir(&mut self) -> Vec3 {
        let theta = self.rand() * std::f32::consts::TAU;
        let phi = self.rand() * std::f32::consts::PI;
        Vec3::new(
            phi.sin() * theta.cos(),
            phi.cos(),
            phi.sin() * theta.sin(),
        )
    }

    pub fn update(&mut self, dt: f32) {
        self.particles.retain_mut(|p| {
            p.age += dt;
            if p.age >= p.lifetime {
                return false;
            }
            p.velocity.y += GRAVITY * p.gravity_scale * dt;
            p.position += p.velocity * dt;
            true
        });
    }

    /// Emit N particles at `origin` with random velocities in [speed_lo, speed_hi].
    fn emit(&mut self, origin: Vec3, count: usize, speed_lo: f32, speed_hi: f32,
            scale: f32, lifetime_lo: f32, lifetime_hi: f32, upward_bias: f32) {
        let budget = MAX_PARTICLES.saturating_sub(self.particles.len());
        let count = count.min(budget);
        for _ in 0..count {
            let mut dir = self.rand_dir();
            dir.y = dir.y.abs() + upward_bias;
            dir = dir.normalize_or_zero();
            let speed = self.rand_range(speed_lo, speed_hi);
            let lt = self.rand_range(lifetime_lo, lifetime_hi);
            self.particles.push(Particle {
                position: origin,
                velocity: dir * speed,
                color_id: 0,
                scale,
                lifetime: lt,
                age: 0.0,
                gravity_scale: 1.0,
            });
        }
    }

    // ----- Preset effects -----

    /// Melee hit sparks.
    pub fn emit_hit(&mut self, pos: Vec3) {
        self.emit(pos, 8, 3.0, 6.0, 0.06, 0.2, 0.4, 0.3);
    }

    /// Fireball explosion.
    pub fn emit_fireball(&mut self, pos: Vec3) {
        self.emit(pos, 18, 4.0, 8.0, 0.1, 0.3, 0.7, 0.2);
    }

    /// Ice shard shatter.
    pub fn emit_ice(&mut self, pos: Vec3) {
        self.emit(pos, 10, 3.0, 6.0, 0.07, 0.2, 0.5, 0.1);
    }

    /// Heal effect — gentle upward drift.
    pub fn emit_heal(&mut self, pos: Vec3) {
        self.emit(pos, 12, 1.0, 2.5, 0.08, 0.6, 1.0, 2.0);
    }

    /// Enemy death burst.
    pub fn emit_death(&mut self, pos: Vec3) {
        self.emit(pos, 15, 2.0, 5.0, 0.1, 0.3, 0.6, 0.5);
    }

    /// Level up fountain.
    pub fn emit_level_up(&mut self, pos: Vec3) {
        self.emit(pos, 25, 3.0, 7.0, 0.08, 0.7, 1.3, 3.0);
    }

    /// Item pickup sparkle.
    pub fn emit_pickup(&mut self, pos: Vec3) {
        self.emit(pos, 5, 1.0, 2.0, 0.05, 0.15, 0.3, 1.0);
    }

    /// Continuous fire particle for torches — slow upward drift, no gravity.
    pub fn emit_fire(&mut self, pos: Vec3, count: usize) {
        let budget = MAX_PARTICLES.saturating_sub(self.particles.len());
        let count = count.min(budget);
        for _ in 0..count {
            let mut dir = self.rand_dir();
            dir.y = dir.y.abs() * 0.3 + 1.0; // strongly upward
            dir.x *= 0.3;
            dir.z *= 0.3;
            dir = dir.normalize_or_zero();
            let speed = self.rand_range(0.4, 1.2);
            let lt = self.rand_range(0.3, 0.7);
            let sc = self.rand_range(0.02, 0.05);
            self.particles.push(Particle {
                position: pos,
                velocity: dir * speed,
                color_id: 0,
                scale: sc,
                lifetime: lt,
                age: 0.0,
                gravity_scale: -0.15, // slight upward buoyancy
            });
        }
    }

    /// Push particle transforms into the frame render arrays.
    pub fn render(&self, transforms: &mut Vec<Mat4>, instance_ids: &mut Vec<u32>) {
        for (i, p) in self.particles.iter().enumerate() {
            // Fade out: shrink near end of life.
            let life_frac = 1.0 - (p.age / p.lifetime);
            let s = p.scale * life_frac;
            if s < 0.001 { continue; }
            let t = Mat4::from_translation(p.position) * Mat4::from_scale(Vec3::splat(s));
            transforms.push(t);
            instance_ids.push(pack_instance_id(MESH_CUBE, PARTICLE_OBJECT_BASE + i as u32));
        }
    }

    pub fn count(&self) -> usize {
        self.particles.len()
    }
}
