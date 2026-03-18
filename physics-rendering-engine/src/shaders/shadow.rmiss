#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT float shadowed;

void main() {
    // Ray reached light without hitting anything — not in shadow.
    shadowed = 1.0;
}
