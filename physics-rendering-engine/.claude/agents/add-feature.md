# Add Feature

Plan and implement a new game feature for the physics rendering engine.

## Input

The user will describe a feature they want added. Use $ARGUMENTS as the feature description.

## Steps

1. **Understand**: Parse the feature request from $ARGUMENTS
2. **Analyze**: Read the relevant existing modules to understand current state and integration points
3. **Plan**: Identify which files need changes and propose an implementation plan. Consider:
   - Which module(s) does this belong in? (renderer, physics, game, voxel, engine, new module?)
   - What new structs/functions are needed?
   - How does it integrate with the update loop in `engine.rs`?
   - Does it need new shader code?
   - Does it need new physics bodies or raycasts?
4. **Present the plan** to the user and wait for approval before coding
5. **Implement**: Write the code, following existing patterns in the codebase
6. **Verify**: Run `cargo clippy` and `cargo build` to ensure it compiles

## Rules

- Respect module boundaries: ash types in renderer/, rapier3d types in physics/
- Follow existing code style and patterns (check nearby code for conventions)
- Keep engine.rs as the orchestrator — add new systems as separate structs/modules
- Prefer struct-based design (no ECS) — this project uses plain Rust structs
- Wire new per-frame logic into `Engine::update()` and rendering into `Engine::render()`
- Do not over-engineer: implement the simplest working version first
