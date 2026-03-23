# Shader Check

Validate that all GLSL shaders compile successfully.

## Steps

1. Find all shader files in `src/shaders/` (`.rgen`, `.rchit`, `.rmiss`, `.vert`, `.frag`)
2. Run `glslangValidator --target-env vulkan1.2 -V` on each shader file
3. If any shader fails to compile:
   - Show the exact error with file and line number
   - Read the shader source around the error
   - Fix the issue
   - Re-validate
4. Report which shaders passed and which needed fixes

## Rules

- Use `--target-env vulkan1.2` for ray tracing shader stages
- If `glslangValidator` is not installed, suggest installing it via the system package manager
- Do not modify shader logic — only fix compilation errors (typos, missing includes, type mismatches)
