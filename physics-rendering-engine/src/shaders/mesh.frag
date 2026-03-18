#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    vec3  lightDir = normalize(vec3(1.0, 3.0, 1.0));
    float ambient  = 0.25;
    float diffuse  = max(0.0, dot(fragNormal, lightDir)) * 0.75;
    outColor = vec4(fragColor * (ambient + diffuse), 1.0);
}
