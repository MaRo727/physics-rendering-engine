// ---------------------------------------------------------------------------
// Input state
// ---------------------------------------------------------------------------

pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    pub descend: bool,     // Left Shift — ghost descend
    pub sprint: bool,      // Left Shift — sprint
    pub interact: bool,    // E — pick up / drop
    pub throw: bool,       // Left mouse button — throw held object
    pub place: bool,       // Right mouse button — place held cube into grid
    pub toggle_ghost: bool, // G — toggle ghost / debug camera
    pub spawn: bool,        // F — spawn a new block
    pub cycle_tool: bool,    // Tab — cycle equipped tool
    pub debug_stats: bool,   // F1 — print player stats to console
    pub toggle_fast: bool,   // F2 — toggle fast move speed
    pub toggle_debug_ui: bool, // F3 — toggle debug overlay
    pub toggle_mute: bool,     // M — toggle music mute
    pub fast_time: bool,        // F4 — speed up day/night cycle (10x)
    pub cycle_spell: bool,      // Q — cycle active spell
    pub cast_spell: bool,       // R — cast active spell
    pub toggle_inventory: bool,  // I — toggle inventory screen
    pub quick_save: bool,         // F5 — quick save
    pub quick_load: bool,         // F9 — quick load
    pub toggle_god: bool,         // F6 — toggle god mode (oneshot + invulnerable)
    pub cycle_block_type: bool,    // B — cycle selected block type
    pub rotate_block: bool,        // V — rotate selected block
    pub toggle_weather: bool,       // F7 — debug: cycle random weather / clear
    pub toggle_editor: bool,        // F8 — toggle structure editor
    pub editor_save: bool,          // F9 — save blueprint (in editor)
    pub editor_load: bool,          // F10 — load blueprint (in editor)
    pub editor_prev_group: bool,     // Left arrow — select prev baked group
    pub editor_next_group: bool,     // Right arrow — select next baked group
    pub editor_unbake: bool,         // U — unbake selected group back to cells
    pub editor_color_slot: Option<u8>, // 1-9 number keys
    pub place_torch: bool,              // T — place a torch
    pub toggle_perf: bool,              // F11 — toggle performance mode
    pub scroll_delta: f32,           // Mouse wheel scroll
    pub mouse_dx: f32,
    pub mouse_dy: f32,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            jump: false,
            descend: false,
            sprint: false,
            interact: false,
            throw: false,
            place: false,
            toggle_ghost: false,
            spawn: false,
            cycle_tool: false,
            debug_stats: false,
            toggle_fast: false,
            toggle_debug_ui: false,
            toggle_mute: false,
            fast_time: false,
            cycle_spell: false,
            cast_spell: false,
            toggle_inventory: false,
            quick_save: false,
            quick_load: false,
            toggle_god: false,
            cycle_block_type: false,
            rotate_block: false,
            toggle_weather: false,
            toggle_editor: false,
            editor_save: false,
            editor_load: false,
            editor_prev_group: false,
            editor_next_group: false,
            editor_unbake: false,
            place_torch: false,
            toggle_perf: false,
            editor_color_slot: None,
            scroll_delta: 0.0,
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }
    }
}
