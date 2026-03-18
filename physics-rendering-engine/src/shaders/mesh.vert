#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
} camera;

layout(push_constant) uniform PushConstants {
    mat4 model;
} push;

void main() {
    gl_Position = camera.proj * camera.view * push.model * vec4(inPosition, 1.0);
    fragColor = inColor;
}
