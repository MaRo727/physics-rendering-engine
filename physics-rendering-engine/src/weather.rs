use crate::terrain::Biome;

// ---------------------------------------------------------------------------
// Weather system — state machine with smooth transitions and world effects
// ---------------------------------------------------------------------------

/// Discrete weather states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeatherKind {
    Clear,
    Cloudy,
    Rain,
    Thunderstorm,
    Fog,
    Windy,
}

/// Interpolated weather parameters (all 0..1 unless noted).
#[derive(Debug, Clone)]
pub struct Weather {
    pub kind: WeatherKind,
    prev_kind: WeatherKind,
    transition: f32, // 0 = fully prev, 1 = fully current

    // --- Interpolated outputs ---
    pub wind_strength: f32,   // 0..1
    pub wind_angle: f32,      // radians, world-space direction of wind
    pub rain_intensity: f32,  // 0..1
    pub fog_density: f32,     // 0..1
    pub lightning_flash: f32, // 0..1 (instantaneous brightness)
    pub cloud_coverage: f32,  // 0..1

    // --- Internal timers ---
    pub weather_time: f32,    // monotonic time for shader animation
    change_timer: f32,        // seconds until next weather change
    lightning_timer: f32,     // countdown to next lightning strike
    lightning_cooldown: f32,  // time since last strike (for flash decay)
    wind_gust_timer: f32,     // modulates wind strength with gusts
    seed: u32,                // deterministic RNG state
}

/// Duration of a smooth transition between weather states (seconds).
const TRANSITION_DURATION: f32 = 15.0;
/// Minimum time between weather changes (seconds).
const MIN_WEATHER_DURATION: f32 = 60.0;
/// Maximum time between weather changes (seconds).
const MAX_WEATHER_DURATION: f32 = 180.0;
/// Lightning flash decay speed.
const LIGHTNING_DECAY: f32 = 6.0;

// Target values for each weather kind.
fn targets(kind: WeatherKind) -> (f32, f32, f32, f32) {
    // (wind_strength, rain_intensity, fog_density, cloud_coverage)
    match kind {
        WeatherKind::Clear        => (0.05, 0.0, 0.0,  0.0),
        WeatherKind::Cloudy       => (0.15, 0.0, 0.05, 0.6),
        WeatherKind::Rain         => (0.25, 0.7, 0.15, 0.85),
        WeatherKind::Thunderstorm => (0.55, 1.0, 0.25, 1.0),
        WeatherKind::Fog          => (0.05, 0.0, 0.8,  0.4),
        WeatherKind::Windy        => (0.8,  0.0, 0.0,  0.2),
    }
}

fn hash_step(state: &mut u32) -> u32 {
    let mut h = *state;
    h = h.wrapping_mul(1664525).wrapping_add(1013904223);
    h ^= h >> 16;
    *state = h;
    h
}

fn hash_f32(state: &mut u32) -> f32 {
    (hash_step(state) & 0xFFFFFF) as f32 / 16777216.0
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Weather {
    pub fn new(seed: u32) -> Self {
        let mut s = seed.wrapping_add(54321);
        let initial_angle = hash_f32(&mut s) * std::f32::consts::TAU;
        Self {
            kind: WeatherKind::Clear,
            prev_kind: WeatherKind::Clear,
            transition: 1.0,
            wind_strength: 0.05,
            wind_angle: initial_angle,
            rain_intensity: 0.0,
            fog_density: 0.0,
            lightning_flash: 0.0,
            cloud_coverage: 0.0,
            weather_time: 0.0,
            change_timer: 90.0, // first change after ~90s
            lightning_timer: 5.0,
            lightning_cooldown: 10.0,
            wind_gust_timer: 0.0,
            seed: s,
        }
    }

    /// Force an immediate transition to a random non-Clear weather.
    pub fn force_random(&mut self) {
        let kinds = [
            WeatherKind::Cloudy,
            WeatherKind::Rain,
            WeatherKind::Thunderstorm,
            WeatherKind::Fog,
            WeatherKind::Windy,
        ];
        let idx = (hash_step(&mut self.seed) as usize) % kinds.len();
        self.prev_kind = self.kind;
        self.kind = kinds[idx];
        self.transition = 0.0;
        self.change_timer = MAX_WEATHER_DURATION; // don't auto-change while debugging
        self.wind_angle += (hash_f32(&mut self.seed) - 0.5) * 1.5;
    }

    /// Force an immediate transition back to Clear weather.
    pub fn force_clear(&mut self) {
        self.prev_kind = self.kind;
        self.kind = WeatherKind::Clear;
        self.transition = 0.0;
        self.change_timer = MAX_WEATHER_DURATION;
        self.lightning_flash = 0.0;
    }

    /// Advance the weather simulation by `dt` seconds.
    /// `biome` is the biome the player is currently standing in.
    pub fn update(&mut self, dt: f32, biome: Biome) {
        self.weather_time += dt;

        // --- Transition blending ---
        if self.transition < 1.0 {
            self.transition = (self.transition + dt / TRANSITION_DURATION).min(1.0);
        }

        let (pw, pr, pf, pc) = targets(self.prev_kind);
        let (cw, cr, cf, cc) = targets(self.kind);
        let t = smoothstep(0.0, 1.0, self.transition);

        let mut base_wind = pw + (cw - pw) * t;
        let base_rain = pr + (cr - pr) * t;
        let base_fog = pf + (cf - pf) * t;
        let base_cloud = pc + (cc - pc) * t;

        // Biome-based minimum wind: forests always have a mild breeze.
        let biome_min_wind = match biome {
            Biome::Forest => 0.18,
            Biome::Plains => 0.08,
            _ => 0.0,
        };
        base_wind = base_wind.max(biome_min_wind);

        // --- Wind gusts ---
        self.wind_gust_timer += dt;
        let gust = (self.wind_gust_timer * 0.7).sin() * 0.3
            + (self.wind_gust_timer * 1.9).sin() * 0.15
            + (self.wind_gust_timer * 3.7).sin() * 0.05;
        self.wind_strength = (base_wind + gust * base_wind).clamp(0.0, 1.0);

        // Slowly drift wind angle.
        self.wind_angle += (self.wind_gust_timer * 0.1).sin() * 0.02 * dt;

        self.rain_intensity = base_rain;
        self.fog_density = base_fog;
        self.cloud_coverage = base_cloud;

        // --- Lightning ---
        if self.kind == WeatherKind::Thunderstorm && self.transition > 0.5 {
            self.lightning_timer -= dt;
            self.lightning_cooldown += dt;
            if self.lightning_timer <= 0.0 {
                self.lightning_flash = 1.0;
                self.lightning_cooldown = 0.0;
                // Next strike in 2-8 seconds.
                self.lightning_timer = 2.0 + hash_f32(&mut self.seed) * 6.0;
            }
        }
        // Decay flash.
        if self.lightning_flash > 0.0 {
            self.lightning_flash = (self.lightning_flash - LIGHTNING_DECAY * dt).max(0.0);
        }

        // --- Auto-transition timer ---
        self.change_timer -= dt;
        if self.change_timer <= 0.0 {
            let next = self.pick_next(biome);
            if next != self.kind {
                self.prev_kind = self.kind;
                self.kind = next;
                self.transition = 0.0;
                // Shift wind angle on weather change.
                self.wind_angle += (hash_f32(&mut self.seed) - 0.5) * 1.5;
            }
            self.change_timer = MIN_WEATHER_DURATION
                + hash_f32(&mut self.seed) * (MAX_WEATHER_DURATION - MIN_WEATHER_DURATION);
        }
    }

    /// Wind direction as a normalized (x, z) vector.
    pub fn wind_dir(&self) -> (f32, f32) {
        (self.wind_angle.cos(), self.wind_angle.sin())
    }

    /// Pick the next weather state based on biome-weighted probabilities.
    fn pick_next(&mut self, biome: Biome) -> WeatherKind {
        // Probability weights: [Clear, Cloudy, Rain, Thunderstorm, Fog, Windy]
        let weights: [f32; 6] = match biome {
            Biome::Plains => [0.30, 0.25, 0.20, 0.05, 0.05, 0.15],
            Biome::Forest => [0.20, 0.20, 0.25, 0.10, 0.15, 0.10],
            Biome::Desert => [0.45, 0.15, 0.02, 0.02, 0.06, 0.30],
            Biome::Mountains => [0.20, 0.15, 0.15, 0.10, 0.20, 0.20],
            Biome::Dungeon => [0.10, 0.15, 0.10, 0.05, 0.50, 0.10],
        };

        // Reduce chance of repeating the same weather.
        let mut adjusted = weights;
        let current_idx = self.kind as usize;
        adjusted[current_idx] *= 0.3;

        // Natural progression: cloudy often leads to rain, rain to thunderstorm.
        match self.kind {
            WeatherKind::Cloudy => {
                adjusted[WeatherKind::Rain as usize] *= 1.8;
            }
            WeatherKind::Rain => {
                adjusted[WeatherKind::Thunderstorm as usize] *= 1.5;
                adjusted[WeatherKind::Cloudy as usize] *= 1.3;
            }
            WeatherKind::Thunderstorm => {
                adjusted[WeatherKind::Rain as usize] *= 1.5;
                adjusted[WeatherKind::Cloudy as usize] *= 1.2;
            }
            _ => {}
        }

        let total: f32 = adjusted.iter().sum();
        let roll = hash_f32(&mut self.seed) * total;

        let mut accum = 0.0;
        for (i, &w) in adjusted.iter().enumerate() {
            accum += w;
            if roll < accum {
                return match i {
                    0 => WeatherKind::Clear,
                    1 => WeatherKind::Cloudy,
                    2 => WeatherKind::Rain,
                    3 => WeatherKind::Thunderstorm,
                    4 => WeatherKind::Fog,
                    _ => WeatherKind::Windy,
                };
            }
        }
        WeatherKind::Clear
    }
}
