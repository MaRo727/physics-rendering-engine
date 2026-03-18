#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform PushConstants {
    mat4 model;
} push;

void main() {
    // Transform vertex to world space using the physics body's transform.
    vec3 world = (push.model * vec4(inPosition, 1.0)).xyz;

    // Fixed isometric "camera" — 45° Y rotation then 30° X rotation.
    // Phase 7 replaces this with a proper view-projection matrix via UBO.
    const float cy = 0.707, sy = 0.707; // cos/sin(45°)
    const float cx = 0.866, sx = 0.500; // cos/sin(30°)
    world = vec3(cy * world.x + sy * world.z,  world.y,                  -sy * world.x + cy * world.z);
    world = vec3(world.x,                       cx * world.y - sx * world.z, sx * world.y + cx * world.z);

    // Scale to NDC. Flip Y so +world.y is up on screen (Vulkan NDC y points down).
    gl_Position = vec4(world.x * 0.3, -world.y * 0.3, world.z * 0.15 + 0.5, 1.0);
    fragColor = inColor;
}
