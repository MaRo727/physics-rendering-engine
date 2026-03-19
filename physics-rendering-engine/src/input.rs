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
    pub interact: bool,    // E — pick up / drop
    pub throw: bool,       // Left mouse button — throw held object
    pub place: bool,       // Right mouse button — place held cube into grid
    pub toggle_ghost: bool, // G — toggle ghost / debug camera
    pub spawn: bool,        // F — spawn a new block
    pub cycle_tool: bool,    // Tab — cycle equipped tool
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
            interact: false,
            throw: false,
            place: false,
            toggle_ghost: false,
            spawn: false,
            cycle_tool: false,
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }
    }
}
