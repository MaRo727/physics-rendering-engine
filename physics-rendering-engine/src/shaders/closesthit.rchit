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
    mat4 playerVP;
    vec4 ghostMode;   // .x > 0 when ghost mode active
} scene;

layout(set = 0, binding = 3) readonly buffer VertexBuf { float verts[]; };
layout(set = 0, binding = 4) readonly buffer IndexBuf  { uint  idxs[];  };

struct MeshOffset {
    uint index_base;
    uint vertex_base;
};
layout(set = 0, binding = 5) readonly buffer MeshOffsetBuf { MeshOffset mesh_offsets[]; };

hitAttributeEXT vec2 attribs;

void main() {
    // Decode mesh_type (upper 8 bits) and object_id (lower 16 bits).
    uint mesh_type = gl_InstanceCustomIndexEXT >> 16u;
    uint object_id = gl_InstanceCustomIndexEXT & 0xFFFFu;

    uint idx_base = mesh_offsets[mesh_type].index_base;
    uint vtx_base = mesh_offsets[mesh_type].vertex_base;

    // Fetch triangle indices (gl_PrimitiveID is relative to the BLAS geometry).
    uint base = idx_base + uint(gl_PrimitiveID) * 3u;
    uint i0 = idxs[base]     + vtx_base;
    uint i1 = idxs[base + 1u] + vtx_base;
    uint i2 = idxs[base + 2u] + vtx_base;

    // Stride = 9 floats: position(3) + normal(3) + color(3).
    vec3 n0 = vec3(verts[i0 * 9u + 3u], verts[i0 * 9u + 4u], verts[i0 * 9u + 5u]);
    vec3 n1 = vec3(verts[i1 * 9u + 3u], verts[i1 * 9u + 4u], verts[i1 * 9u + 5u]);
    vec3 n2 = vec3(verts[i2 * 9u + 3u], verts[i2 * 9u + 4u], verts[i2 * 9u + 5u]);

    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 normal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);

    // Transform normal to world space.
    normal = normalize(mat3(gl_ObjectToWorldEXT) * normal);

    // Read per-vertex color from the first vertex of the triangle.
    vec3 color = vec3(verts[i0 * 9u + 6u], verts[i0 * 9u + 7u], verts[i0 * 9u + 8u]);

    // Shadow ray — offset origin along surface normal to avoid self-intersection.
    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 shadowOrigin = hitPos + normal * 0.01;
    vec3 L = normalize(scene.lightDir.xyz);
    shadowed = 0.0;
    traceRayEXT(
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF,
        0,    // sbtRecordOffset
        1,    // sbtRecordStride
        1,    // missIndex → shadow.rmiss
        shadowOrigin,
        0.001,
        L,
        10000.0,
        1     // payload location
    );

    float NdotL = dot(normal, L);
    float ambient = 0.15;
    float fill = max(0.0, NdotL) * 0.15;
    float diffuse = max(0.0, NdotL) * shadowed;
    vec3 lit = color * scene.lightColor.xyz * scene.lightColor.w * (ambient + fill + diffuse);

    // Ghost mode: green highlight for surfaces inside the player's frozen frustum.
    if (scene.ghostMode.x > 0.0) {
        vec4 clip = scene.playerVP * vec4(hitPos, 1.0);
        vec3 ndc = clip.xyz / clip.w;
        // Vulkan clip space: x,y in [-1,1], z in [0,1].
        if (ndc.x >= -1.0 && ndc.x <= 1.0 &&
            ndc.y >= -1.0 && ndc.y <= 1.0 &&
            ndc.z >=  0.0 && ndc.z <= 1.0) {
            // Distance from frustum edge (0 at edge, 1 at center).
            float ex = 1.0 - abs(ndc.x);
            float ey = 1.0 - abs(ndc.y);
            float edge = min(ex, ey);
            // Bright green edge line, subtle tint inside.
            float edgeWidth = 0.02;
            if (edge < edgeWidth) {
                lit = mix(vec3(0.0, 1.0, 0.0), lit, 0.2);
            } else {
                lit = mix(lit, lit * vec3(0.7, 1.3, 0.7), 0.4);
            }
        }
    }

    payload = lit;
}
