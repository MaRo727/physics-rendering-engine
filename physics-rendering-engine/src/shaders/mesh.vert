#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    // Hardcoded isometric preview rotation so the cube looks 3D without a
    // camera UBO. Phase 7 replaces this with a proper MVP matrix.
    const float cy = 0.707, sy = 0.707; // cos/sin(45 deg) — Y rotation
    const float cx = 0.866, sx = 0.500; // cos/sin(30 deg) — X rotation

    vec3 p = inPosition;
    p = vec3(cy * p.x + sy * p.z,  p.y,              -sy * p.x + cy * p.z); // rotate Y
    p = vec3(p.x,                   cx * p.y - sx * p.z, sx * p.y + cx * p.z); // rotate X

    // Scale and remap Z into Vulkan's [0, 1] NDC depth range.
    gl_Position = vec4(p.xy * 1.2, p.z * 0.5 + 0.5, 1.0);
    fragColor = inColor;
}
