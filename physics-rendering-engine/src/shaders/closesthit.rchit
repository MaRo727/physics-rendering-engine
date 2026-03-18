#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) rayPayloadInEXT vec3 payload;
layout(location = 1) rayPayloadEXT float shadowed;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 2) uniform SceneUBO {
    mat4 invView;
    mat4 invProj;
    vec4 lightDir;
    vec4 lightColor;
} scene;

layout(set = 0, binding = 3) readonly buffer VertexBuf { float verts[]; };
layout(set = 0, binding = 4) readonly buffer IndexBuf  { uint  idxs[];  };

hitAttributeEXT vec2 attribs;

void main() {
    // Fetch triangle indices.
    uint base = uint(gl_PrimitiveID) * 3u;
    uint i0 = idxs[base];
    uint i1 = idxs[base + 1u];
    uint i2 = idxs[base + 2u];

    // Stride = 9 floats: position(3) + normal(3) + color(3).
    vec3 n0 = vec3(verts[i0 * 9u + 3u], verts[i0 * 9u + 4u], verts[i0 * 9u + 5u]);
    vec3 n1 = vec3(verts[i1 * 9u + 3u], verts[i1 * 9u + 4u], verts[i1 * 9u + 5u]);
    vec3 n2 = vec3(verts[i2 * 9u + 3u], verts[i2 * 9u + 4u], verts[i2 * 9u + 5u]);

    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 normal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);

    // Transform normal to world space.
    normal = normalize(mat3(gl_ObjectToWorldEXT) * normal);

    vec3 color = vec3(verts[i0 * 9u + 6u], verts[i0 * 9u + 7u], verts[i0 * 9u + 8u]);

    // Shadow ray.
    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 L = normalize(scene.lightDir.xyz);
    shadowed = 0.0;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF,
        0,    // sbtRecordOffset
        1,    // sbtRecordStride
        1,    // missIndex → shadow.rmiss
        hitPos,
        0.001,
        L,
        10000.0,
        1     // payload location
    );

    float ambient = 0.2;
    float diffuse = max(0.0, dot(normal, L)) * shadowed;
    payload = color * scene.lightColor.xyz * scene.lightColor.w * (ambient + diffuse);
}
