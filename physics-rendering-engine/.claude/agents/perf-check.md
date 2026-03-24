# Performance Check

Analyze the codebase for performance issues critical to a real-time game engine.

## Focus Areas

### Per-Frame Hot Path
- `Engine::update()` and `Engine::render()` run every frame (~16ms budget at 60fps)
- Look for allocations (Vec::new, String, Box) in the per-frame path
- Flag unnecessary `.clone()` calls on large data
- Check for redundant iterations or O(n²) patterns over entities/chunks

### Memory
- Large structs being passed by value instead of reference
- Unnecessary Vec reallocations (missing `with_capacity` or `reserve`)
- Temporary allocations that could be reused across frames

### Vulkan/Renderer
- TLAS rebuild efficiency (rebuilt every frame — is data prep optimal?)
- Staging buffer usage patterns
- Descriptor set update frequency

### Physics
- Rapier3D step configuration
- Raycast frequency and necessity
- Collision group filtering

### Voxel
- Chunk meshing: greedy mesher efficiency
- Chunk loading/unloading patterns
- Dirty tracking effectiveness

### UI
- `ui.rs` runs immediate-mode GPU rendering every frame — check for unnecessary draw calls or text layout work
- HUD and minimap caching effectiveness

### Particles
- `particles.rs` — particle count and update cost per frame
- Particle lifetime and cleanup patterns

### Save/Load
- `save.rs` should NOT do disk I/O per frame — only on explicit save/load triggers
- Terrain cache efficiency

## Steps

1. Read the main update and render paths in `engine.rs`
2. Follow the hot path into each subsystem
3. Flag concrete issues with file:line references
4. Suggest fixes ranked by likely impact (frame time saved)

## Rules

- Only flag real issues, not theoretical ones — focus on what actually runs per frame
- Suggest concrete fixes, not vague advice
- Don't suggest architectural rewrites unless absolutely necessary
- Acceptable trade-offs: some allocations are fine if they simplify code and aren't in the hot path
