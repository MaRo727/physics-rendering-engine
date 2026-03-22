use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Duration;

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

        let mut biome_tracks = HashMap::new();

        // Register known biome tracks.
        let biome_files: &[(Biome, &str)] = &[
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
