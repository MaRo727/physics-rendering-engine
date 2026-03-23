use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};

use crate::terrain::Biome;

// ---------------------------------------------------------------------------
// Music context – determines what music should be playing
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MusicContext {
    Biome(Biome),
    Location(String),
}

// ---------------------------------------------------------------------------
// Footstep sounds per biome (walk + optional run variants)
// ---------------------------------------------------------------------------

struct FootstepSet {
    walk: Vec<PathBuf>,
    run: Vec<PathBuf>,
}

// ---------------------------------------------------------------------------
// Audio manager
// ---------------------------------------------------------------------------

pub struct AudioManager {
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
    sink: Sink,
    current: Option<MusicContext>,
    biome_tracks: HashMap<Biome, PathBuf>,
    location_tracks: HashMap<String, PathBuf>,
    music_volume: f32,
    muted: bool,
    fade_timer: f32,
    fade_duration: f32,
    fade_state: FadeState,
    pending_context: Option<MusicContext>,
    // Footstep SFX
    sfx_volume: f32,
    footstep_sounds: HashMap<Biome, FootstepSet>,
    step_timer: f32,
    step_index: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum FadeState {
    None,
    FadingOut,
    FadingIn,
}

impl AudioManager {
    pub fn new(assets_dir: &Path) -> Option<Self> {
        let (stream, stream_handle) = match OutputStream::try_default() {
            Ok(s) => s,
            Err(e) => {
                log::warn!("Failed to open audio output: {e}. Audio disabled.");
                return None;
            }
        };

        let sink = Sink::try_new(&stream_handle).ok()?;
        sink.set_volume(0.05);

        let music_dir = assets_dir.join("music");
        let footstep_dir = assets_dir.join("sfx").join("footsteps");

        let mut biome_tracks = HashMap::new();

        // Register known biome tracks.
        let biome_files: &[(Biome, &str)] = &[
            (Biome::Plains, "plains_forest.mp3"),
            (Biome::Forest, "plains_forest.mp3"),
            (Biome::Mountains, "mountain.mp3"),
            (Biome::Desert, "desert.mp3"),
            (Biome::Dungeon, "dungeon.mp3"),
        ];

        for &(biome, filename) in biome_files {
            let path = music_dir.join(filename);
            if path.exists() {
                log::info!("Registered music for {biome:?}: {}", path.display());
                biome_tracks.insert(biome, path);
            }
        }

        // Register footstep sounds per biome.
        let mut footstep_sounds = HashMap::new();
        let footstep_defs: &[(Biome, &[&str], &[&str])] = &[
            (Biome::Plains,    &["forest_walk1.wav", "forest_walk2.wav"], &["forest_walk1.wav", "forest_walk2.wav"]),
            (Biome::Forest,    &["forest_walk1.wav", "forest_walk2.wav"], &["forest_walk1.wav", "forest_walk2.wav"]),
            (Biome::Desert,    &["sand_walk.wav"],                    &["sand_walk.wav"]),
            (Biome::Mountains, &["gravel_walk.wav"],                  &["gravel_run.wav"]),
            (Biome::Dungeon,   &["concrete_walk1.wav", "concrete_walk2.wav"], &["concrete_walk1.wav", "concrete_walk2.wav"]),
        ];

        for &(biome, walk_files, run_files) in footstep_defs {
            let collect = |files: &[&str]| -> Vec<PathBuf> {
                files.iter()
                    .map(|f| footstep_dir.join(f))
                    .filter(|p| p.exists())
                    .collect()
            };
            let walk = collect(walk_files);
            let run = collect(run_files);
            if !walk.is_empty() || !run.is_empty() {
                log::info!("Registered footsteps for {biome:?}: {} walk, {} run", walk.len(), run.len());
                footstep_sounds.insert(biome, FootstepSet { walk, run });
            }
        }

        Some(Self {
            _stream: stream,
            stream_handle,
            sink,
            current: None,
            biome_tracks,
            location_tracks: HashMap::new(),
            music_volume: 0.05,
            muted: false,
            fade_timer: 0.0,
            fade_duration: 2.0,
            fade_state: FadeState::None,
            pending_context: None,
            sfx_volume: 0.15,
            footstep_sounds,
            step_timer: 0.0,
            step_index: 0,
        })
    }

    /// Register a music track for a named location (castle, tavern, etc.).
    #[allow(dead_code)]
    pub fn register_location_track(&mut self, name: &str, path: PathBuf) {
        if path.exists() {
            self.location_tracks.insert(name.to_string(), path);
        }
    }

    /// Call each frame with the player's current biome.
    /// If a location override is active, biome music is suppressed.
    pub fn update(&mut self, dt: f32, biome: Biome, location: Option<&str>) {
        // Determine desired context: location takes priority over biome.
        let desired = if let Some(loc) = location {
            if self.location_tracks.contains_key(loc) {
                MusicContext::Location(loc.to_string())
            } else {
                MusicContext::Biome(biome)
            }
        } else {
            MusicContext::Biome(biome)
        };

        // Check if we need to change tracks.
        let needs_change = match &self.current {
            Some(ctx) => *ctx != desired,
            None => true,
        };

        if needs_change && self.fade_state == FadeState::None {
            if self.current.is_some() && !self.sink.empty() {
                // Start crossfade: fade out current, then fade in new.
                self.fade_state = FadeState::FadingOut;
                self.fade_timer = 0.0;
                self.pending_context = Some(desired);
            } else {
                // Nothing playing, start immediately.
                self.play_context(&desired);
                self.current = Some(desired);
            }
        }

        // Effective volume (0 when muted).
        let effective = if self.muted { 0.0 } else { self.music_volume };

        // Handle fading.
        match self.fade_state {
            FadeState::FadingOut => {
                self.fade_timer += dt;
                let t = (self.fade_timer / self.fade_duration).min(1.0);
                self.sink.set_volume(effective * (1.0 - t));

                if t >= 1.0 {
                    self.sink.stop();
                    self.fade_state = FadeState::FadingIn;
                    self.fade_timer = 0.0;

                    if let Some(ctx) = self.pending_context.take() {
                        self.play_context(&ctx);
                        self.current = Some(ctx);
                    }
                    self.sink.set_volume(0.0);
                }
            }
            FadeState::FadingIn => {
                self.fade_timer += dt;
                let t = (self.fade_timer / self.fade_duration).min(1.0);
                self.sink.set_volume(effective * t);

                if t >= 1.0 {
                    self.sink.set_volume(effective);
                    self.fade_state = FadeState::None;
                }
            }
            FadeState::None => {
                // If the track ended (not looping properly), restart it.
                if self.sink.empty() {
                    if let Some(ctx) = &self.current {
                        self.play_context(ctx);
                    }
                }
            }
        }
    }

    fn play_context(&self, ctx: &MusicContext) {
        let path = match ctx {
            MusicContext::Biome(b) => self.biome_tracks.get(b),
            MusicContext::Location(name) => self.location_tracks.get(name),
        };

        if let Some(path) = path {
            if let Err(e) = self.play_file(path) {
                log::warn!("Failed to play {}: {e}", path.display());
            }
        }
    }

    /// Update footstep sounds based on player movement.
    /// `horizontal_speed` is the XZ speed of the player.
    /// `on_ground` indicates whether the player is touching the ground.
    pub fn update_footsteps(&mut self, dt: f32, biome: Biome, horizontal_speed: f32, on_ground: bool) {
        const WALK_THRESHOLD: f32 = 0.5;
        const RUN_THRESHOLD: f32 = 8.0;
        const WALK_INTERVAL: f32 = 0.5;
        const RUN_INTERVAL: f32 = 0.3;

        let moving = on_ground && horizontal_speed >= WALK_THRESHOLD && !self.muted;

        if !moving {
            // Don't reset timer — just stop advancing it so footsteps resume
            // immediately when the player touches ground again.
            return;
        }

        let is_running = horizontal_speed > RUN_THRESHOLD;
        let interval = if is_running { RUN_INTERVAL } else { WALK_INTERVAL };

        self.step_timer += dt;
        if self.step_timer >= interval {
            self.step_timer = 0.0;

            if let Some(set) = self.footstep_sounds.get(&biome) {
                let sounds = if is_running && !set.run.is_empty() {
                    &set.run
                } else {
                    &set.walk
                };

                if !sounds.is_empty() {
                    let idx = self.step_index % sounds.len();
                    self.step_index = self.step_index.wrapping_add(1);
                    if let Err(e) = self.play_sfx(&sounds[idx]) {
                        log::warn!("Failed to play footstep: {e}");
                    }
                }
            }
        }
    }

    fn play_sfx(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let source = Decoder::new(BufReader::new(file))?
            .amplify(self.sfx_volume)
            .convert_samples::<f32>();
        self.stream_handle.play_raw(source)?;
        Ok(())
    }

    fn play_file(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let source = Decoder::new(BufReader::new(file))?;
        self.sink.append(source);
        Ok(())
    }

    pub fn toggle_mute(&mut self) {
        self.muted = !self.muted;
        if self.muted {
            self.sink.set_volume(0.0);
        } else if self.fade_state == FadeState::None {
            self.sink.set_volume(self.music_volume);
        }
    }

    #[allow(dead_code)]
    pub fn set_volume(&mut self, volume: f32) {
        self.music_volume = volume.clamp(0.0, 1.0);
        if !self.muted && self.fade_state == FadeState::None {
            self.sink.set_volume(self.music_volume);
        }
    }
}
