#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) rayPayloadInEXT vec4 payload;
layout(location = 1) rayPayloadEXT float shadowed;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 2) uniform SceneUBO {
    mat4 invView;
    mat4 invProj;
    vec4 lightDir;
    vec4 lightColor;
    mat4 playerVP;
    vec4 ghostMode;   // .x > 0 when ghost mode active
    vec4 debugInfo;
    vec4 debugInfo2;
    vec4 sunMoon;  // .xyz = sun direction, .w = sun altitude
    vec4 moonInfo; // .xyz = moon direction, .w = moon altitude
    vec4 blizzardInfo; // .x = snow intensity (0..1), .y = time, .z = water level
    vec4 weatherInfo;  // .x = rain intensity, .y = fog density, .z = lightning flash, .w = cloud coverage
    vec4 windInfo;     // .x = wind strength, .y = wind dir x, .z = wind dir z, .w = weather time
} scene;

layout(set = 0, binding = 3) readonly buffer VertexBuf { float verts[]; };
layout(set = 0, binding = 4) readonly buffer IndexBuf  { uint  idxs[];  };

struct MeshOffset {
    uint index_base;
    uint vertex_base;
};
layout(set = 0, binding = 5) readonly buffer MeshOffsetBuf { MeshOffset mesh_offsets[]; };

struct PointLight {
    vec3 position;
    float radius;
    vec3 color;
    float intensity;
};
layout(set = 0, binding = 7, std430) readonly buffer PointLightBuf {
    uint light_count;
    uint _pad0, _pad1, _pad2;
    PointLight point_lights[];
};

hitAttributeEXT vec2 attribs;

void main() {
    bool perfMode = scene.blizzardInfo.w > 0.5;

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

    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // Emissive torch flame — bright orange diamond, skip shadow/lighting.
    uint TORCH_MESH_TYPE = 42u;
    if (mesh_type == TORCH_MESH_TYPE && color.r > 0.9 && color.g > 0.5 && color.b < 0.3) {
        float flicker = 0.85 + 0.15 * sin(scene.blizzardInfo.y * 10.0 + hitPos.x * 3.0);
        payload = vec4(color * 2.5 * flicker, gl_HitTEXT);
        return;
    }

    // Shadow ray — offset origin along surface normal to avoid self-intersection.
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
    // Day/night adaptive lighting.
    float sunAlt = scene.sunMoon.w;
    float dayFactor = smoothstep(-0.35, 0.45, sunAlt);
    // Night: dim ambient (0.03), Day: higher ambient from sky scattering (0.15).
    float ambient = mix(0.03, 0.15, dayFactor);
    float fill = max(0.0, NdotL) * 0.15;
    // Moon casts dimmer, softer shadows than sun.
    float shadowStrength = mix(0.4, 1.0, dayFactor); // night shadows are faint
    float diffuse = max(0.0, NdotL) * mix(1.0, shadowed, shadowStrength);
    vec3 lit = color * scene.lightColor.xyz * scene.lightColor.w * (ambient + fill + diffuse);

    // Point light contributions (capped at 8, or 4 in performance mode).
    uint numLights = min(light_count, perfMode ? 4u : 8u);
    if (numLights > 0u && gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT) {
        for (uint li = 0u; li < numLights; li++) {
            PointLight pl = point_lights[li];
            vec3 toLight = pl.position - hitPos;
            float dist2 = dot(toLight, toLight);
            float r2 = pl.radius * pl.radius;
            if (dist2 > r2) continue;

            float dist = sqrt(dist2);
            vec3 L_pt = toLight / dist;
            float NdotL_pt = max(0.0, dot(normal, L_pt));

            // Smooth distance attenuation.
            float atten = 1.0 / (1.0 + dist2 * 0.3);
            float edge = 1.0 - smoothstep(pl.radius * 0.6, pl.radius, dist);
            atten *= edge;

            lit += color * pl.color * pl.intensity * NdotL_pt * atten;
        }
    }

    // Underwater caustics on submerged terrain (skipped in perf mode).
    float waterLevel = scene.blizzardInfo.z;
    uint WATER_MESH_TYPE = 6u;
    if (hitPos.y < waterLevel && mesh_type != WATER_MESH_TYPE && !perfMode) {
        float wTimeCaustic = scene.windInfo.w;
        float depth = waterLevel - hitPos.y;
        float caustStrength = exp(-depth * 0.08);
        vec2 cp = hitPos.xz * 0.3 + wTimeCaustic * vec2(0.8, 0.6);
        float caustic = pow(0.5 + 0.5 * sin(cp.x * 3.0 + sin(cp.y * 2.0 + wTimeCaustic)), 4.0);
        lit += vec3(0.15, 0.2, 0.1) * caustic * caustStrength * shadowed;
    }

    // Water surface shading.
    if (mesh_type == WATER_MESH_TYPE) {
        vec3 viewDir = normalize(gl_WorldRayDirectionEXT);
        float wTime = scene.windInfo.w;
        float windStr = scene.windInfo.x;
        vec2 windDir = vec2(scene.windInfo.y, scene.windInfo.z);

        // Detect if ray hits water from below (underwater camera looking up).
        bool underwater = dot(viewDir, normal) > 0.0;
        if (underwater) {
            normal = -normal;
        }

        // --- Detail ripple normals (animated multi-octave sine waves) ---
        // Reduce octaves at distance to save ALU.
        float waterDist = gl_HitTEXT;
        vec2 wp = hitPos.xz;
        float dx = 0.0, dz = 0.0;

        // Octave 1: broad gentle swells (always)
        dx += 0.025 * 0.5 * cos(wp.x * 0.5 + wTime * 0.8) * cos(wp.y * 0.3 + wTime * 0.5);
        dz += 0.025 * (-0.3) * sin(wp.x * 0.5 + wTime * 0.8) * sin(wp.y * 0.3 + wTime * 0.5);

        // Octave 2: medium ripples (always)
        float phase2 = wp.x * 1.5 - wTime * 1.2 + wp.y * 0.8;
        dx += 0.015 * 1.5 * cos(phase2);
        dz += 0.015 * 0.8 * cos(phase2);

        if (waterDist < 400.0) {
            // Octave 3: small detail ripples
            dx += 0.006 * 3.0 * cos(wp.x * 3.0 + wTime * 2.5) * cos(wp.y * 2.8 - wTime * 1.8);
            dz += 0.006 * (-2.8) * sin(wp.x * 3.0 + wTime * 2.5) * sin(wp.y * 2.8 - wTime * 1.8);
        }
        if (waterDist < 200.0) {
            // Octave 4: tiny sparkle ripples
            dx += 0.003 * 6.0 * cos(wp.x * 6.0 - wTime * 3.5 + wp.y * 4.5);
            dz += 0.003 * 4.5 * cos(wp.x * 6.0 - wTime * 3.5 + wp.y * 4.5);
        }

        // Wind-driven ripple bias
        float windPhase = dot(wp, windDir) * 2.0 * windStr;
        dx += 0.012 * windStr * cos(windPhase + wTime * 3.0);
        dz += 0.012 * windStr * cos(windPhase + wTime * 3.0) * 0.7;

        // Perturb the mesh normal with detail ripples
        normal = normalize(normal + vec3(-dx, 0.0, -dz));

        // --- Fresnel (Schlick, water IOR ~1.33, F0 ≈ 0.02) ---
        // Capped to keep water body color visible at grazing angles
        float NdotV = max(0.0, dot(normal, -viewDir));
        float fresnel = min(0.02 + 0.98 * pow(1.0 - NdotV, 5.0), 0.7);

        // --- Reflection ray ---
        // Mask 0xF8: skip detail geometry (bit 0), shadow-only (bit 1), water (bit 2).
        vec3 reflDir = reflect(viewDir, normal);
        // Skip reflection trace when Fresnel contribution is negligible
        // or water is far away — use sky approximation instead.
        vec3 reflColor;
        if (!perfMode && fresnel > 0.08 && waterDist < 500.0) {
            vec3 reflOrigin = hitPos + normal * 0.05;
            traceRayEXT(
                topLevelAS,
                gl_RayFlagsOpaqueEXT,
                0xF8, // skip detail, shadow-only, water
                0, 1, 0,
                reflOrigin, 0.01, reflDir, 200.0,
                0
            );
            reflColor = payload.xyz;
        } else {
            // Approximate: use sky-tinted horizon color.
            reflColor = mix(vec3(0.5, 0.65, 0.85), vec3(0.15, 0.35, 0.75), max(0.0, reflDir.y))
                        * scene.lightColor.w;
        }

        // --- Refraction ray ---
        // IOR: air-to-water = 1/1.33, water-to-air = 1.33/1.0
        float eta = underwater ? 1.33 : (1.0 / 1.33);
        vec3 refrDir = refract(viewDir, normal, eta);

        vec3 refrColor;
        if (length(refrDir) < 0.001) {
            // Total internal reflection (underwater at grazing angles).
            refrColor = reflColor;
            fresnel = 1.0;
        } else if (!perfMode && waterDist < 500.0) {
            vec3 refrOrigin = hitPos - normal * 0.05; // offset through surface
            traceRayEXT(
                topLevelAS,
                gl_RayFlagsOpaqueEXT,
                0xF8, // skip detail, shadow-only, water
                0, 1, 0,
                refrOrigin, 0.01, refrDir, 200.0,
                0
            );
            float refrHitDist = payload.w;
            refrColor = payload.xyz;

            // Beer's law absorption — deeper water absorbs more red.
            vec3 refrHitPos = refrOrigin + refrDir * refrHitDist;
            float waterDepth = max(0.0, abs(hitPos.y - refrHitPos.y));
            vec3 absorption = exp(-vec3(0.4, 0.1, 0.05) * waterDepth);
            refrColor *= absorption;
        } else {
            // Far water: approximate with deep water body color.
            refrColor = vec3(0.04, 0.12, 0.18) * scene.lightColor.w;
        }

        // --- Sun specular ---
        vec3 halfVec = normalize(L - viewDir);
        float spec = pow(max(0.0, dot(normal, halfVec)), 192.0) * shadowed;
        vec3 specular = vec3(1.0, 0.95, 0.9) * spec * 0.9;

        // --- Moon specular (subtle, at night) ---
        vec3 moonDir = normalize(scene.moonInfo.xyz);
        float moonAlt = scene.moonInfo.w;
        if (moonAlt > -0.1) {
            vec3 moonHalf = normalize(moonDir - viewDir);
            float moonSpec = pow(max(0.0, dot(normal, moonHalf)), 128.0);
            float moonFade = smoothstep(-0.1, 0.1, moonAlt) * (1.0 - dayFactor);
            specular += vec3(0.3, 0.35, 0.5) * moonSpec * 0.2 * moonFade;
        }

        // --- Combine: Fresnel blends reflection vs refraction ---
        lit = mix(refrColor, reflColor, fresnel) + specular;
    }

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

    // Rain — wet surface darkening + distance fog.
    float rainIntensity = scene.weatherInfo.x;
    if (rainIntensity > 0.0 && mesh_type != WATER_MESH_TYPE) {
        // Wet surfaces: darken and slightly increase saturation.
        float wetness = rainIntensity * 0.3;
        lit *= (1.0 - wetness);
    }

    // Rain distance fog — gray-blue atmospheric scattering.
    if (rainIntensity > 0.0) {
        float dist = gl_HitTEXT;
        float fogNear = mix(200.0, 40.0, rainIntensity);
        float fogFar = mix(600.0, 120.0, rainIntensity);
        float fogFactor = clamp((dist - fogNear) / (fogFar - fogNear), 0.0, 1.0);
        vec3 rainFogColor = vec3(0.35, 0.38, 0.45) * scene.lightColor.w * 1.5;
        rainFogColor = max(rainFogColor, vec3(0.04, 0.05, 0.07));
        lit = mix(lit, rainFogColor, fogFactor * rainIntensity * 0.8);
    }

    // Fog weather — dense gray-blue fog.
    float fogDensity = scene.weatherInfo.y;
    if (fogDensity > 0.0) {
        float dist = gl_HitTEXT;
        float fogNear = mix(100.0, 10.0, fogDensity);
        float fogFar = mix(400.0, 60.0, fogDensity);
        float fogFactor = clamp((dist - fogNear) / (fogFar - fogNear), 0.0, 1.0);
        vec3 fogColor = vec3(0.55, 0.58, 0.62) * scene.lightColor.w * 1.2;
        fogColor = max(fogColor, vec3(0.06, 0.07, 0.09));
        lit = mix(lit, fogColor, fogFactor * fogDensity);
    }

    // Lightning flash — bright illumination.
    float lightningFlash = scene.weatherInfo.z;
    if (lightningFlash > 0.0) {
        float extraLight = lightningFlash * 2.0;
        lit += color * extraLight * max(0.0, normal.y * 0.5 + 0.5);
    }

    // Cloud coverage — dim overall lighting slightly.
    float cloudCoverage = scene.weatherInfo.w;
    if (cloudCoverage > 0.0) {
        lit *= mix(1.0, 0.7, cloudCoverage * 0.5);
    }

    // Blizzard distance fog — white fog that reduces visibility.
    float snowIntensity = scene.blizzardInfo.x;
    if (snowIntensity > 0.0) {
        float dist = gl_HitTEXT;
        // Fog range shrinks with intensity: full blizzard ~15-40 unit visibility.
        float fogNear = mix(80.0, 15.0, snowIntensity);
        float fogFar = mix(200.0, 40.0, snowIntensity);
        float fogFactor = clamp((dist - fogNear) / (fogFar - fogNear), 0.0, 1.0);
        // Blizzard fog color: slightly blue-white.
        vec3 fogColor = vec3(0.75, 0.78, 0.82) * scene.lightColor.w * 1.5;
        fogColor = max(fogColor, vec3(0.08, 0.09, 0.11));
        lit = mix(lit, fogColor, fogFactor * snowIntensity);
    }

    payload = vec4(lit, gl_HitTEXT);
}
