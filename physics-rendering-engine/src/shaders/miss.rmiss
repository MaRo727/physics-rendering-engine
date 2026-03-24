#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

layout(set = 0, binding = 2) uniform SceneUBO {
    mat4 invView;
    mat4 invProj;
    vec4 lightDir;
    vec4 lightColor;
    mat4 playerVP;
    vec4 ghostMode;
    vec4 debugInfo;
    vec4 debugInfo2;
    vec4 sunMoon;  // .xyz = sun direction, .w = sun altitude (-1 to 1)
    vec4 moonInfo; // .xyz = moon direction, .w = moon altitude (-1 to 1)
    vec4 blizzardInfo; // .x = snow intensity (0..1), .y = time
    vec4 weatherInfo;  // .x = rain intensity, .y = fog density, .z = lightning flash, .w = cloud coverage
    vec4 windInfo;     // .x = wind strength, .y = wind dir x, .z = wind dir z, .w = weather time
} scene;

void main() {
    vec3 dir = normalize(gl_WorldRayDirectionEXT);

    float sunAlt = scene.sunMoon.w;
    float moonAlt = scene.moonInfo.w;
    vec3 sunDir = normalize(scene.sunMoon.xyz);
    vec3 moonDir = normalize(scene.moonInfo.xyz);

    // --- Sky gradient ---
    float y = dir.y;

    // Blend factor: 0 = full night, 1 = full day. Wide range for gradual transition.
    float dayFactor = smoothstep(-0.35, 0.45, sunAlt);
    float nightness = 1.0 - dayFactor;
    // Sunset factor: strongest when sun is near horizon.
    float sunsetFactor = smoothstep(-0.2, 0.05, sunAlt) * smoothstep(0.5, 0.0, sunAlt);

    // Base sky colors.
    vec3 dayZenith  = vec3(0.15, 0.35, 0.75);
    vec3 dayHorizon = vec3(0.5, 0.65, 0.85);
    vec3 nightZenith  = vec3(0.005, 0.005, 0.02);
    vec3 nightHorizon = vec3(0.01, 0.02, 0.05);

    // Sunset/sunrise colors.
    vec3 sunsetHorizon = vec3(0.9, 0.4, 0.1);
    vec3 sunsetZenith  = vec3(0.3, 0.15, 0.4);

    vec3 zenith  = mix(nightZenith, dayZenith, dayFactor);
    vec3 horizon = mix(nightHorizon, dayHorizon, dayFactor);
    zenith  = mix(zenith, sunsetZenith, sunsetFactor * 0.6);
    horizon = mix(horizon, sunsetHorizon, sunsetFactor * 0.8);

    // Vertical gradient.
    float t = clamp(y * 2.0, 0.0, 1.0);
    vec3 sky = mix(horizon, zenith, t);

    // Below horizon: darken toward ground.
    if (y < 0.0) {
        vec3 groundColor = mix(vec3(0.02, 0.03, 0.02), vec3(0.05, 0.06, 0.04), dayFactor);
        sky = mix(sky, groundColor, clamp(-y * 4.0, 0.0, 1.0));
    }

    // --- Sun disc ---
    // Keep rendering until well below horizon so it visually sets smoothly.
    if (sunAlt > -0.3) {
        float sunAngle = acos(clamp(dot(dir, sunDir), -1.0, 1.0));
        float sunDiscRadius = 0.035;
        float sunGlowRadius = 0.18;
        // Fade out disc as sun goes below horizon.
        float horizonFade = smoothstep(-0.3, -0.05, sunAlt);

        // Sun disc (bright white-yellow).
        if (sunAngle < sunDiscRadius) {
            float discFade = 1.0 - sunAngle / sunDiscRadius;
            vec3 sunColor = mix(vec3(1.0, 0.95, 0.8), vec3(1.0, 0.5, 0.15), sunsetFactor);
            sky = mix(sky, sunColor * (2.0 + discFade * 2.0), horizonFade);
        }
        // Sun glow.
        else if (sunAngle < sunGlowRadius) {
            float glowStr = pow(1.0 - sunAngle / sunGlowRadius, 2.5);
            vec3 glowColor = mix(vec3(1.0, 0.85, 0.6), vec3(1.0, 0.4, 0.15), sunsetFactor);
            sky += glowColor * glowStr * 0.5 * horizonFade;
        }
    }

    // --- Moon disc ---
    if (moonAlt > -0.3) {
        float moonAngle = acos(clamp(dot(dir, moonDir), -1.0, 1.0));
        float moonDiscRadius = 0.025;
        float moonGlowRadius = 0.1;
        // Fade out disc as moon goes below horizon.
        float horizonFade = smoothstep(-0.3, -0.05, moonAlt);

        // Moon disc (pale white).
        if (moonAngle < moonDiscRadius) {
            float discFade = 1.0 - moonAngle / moonDiscRadius;
            vec3 moonColor = vec3(0.85, 0.88, 0.95);
            sky = mix(sky, moonColor * (0.8 + discFade * 0.8), (0.7 + 0.3 * nightness) * horizonFade);
        }
        // Moon glow (subtle).
        else if (moonAngle < moonGlowRadius) {
            float glowStr = pow(1.0 - moonAngle / moonGlowRadius, 3.0);
            sky += vec3(0.1, 0.12, 0.2) * glowStr * 0.25 * nightness * horizonFade;
        }
    }

    // --- Stars at night ---
    if (nightness > 0.1 && y > 0.0) {
        // Hash based on quantized direction for stable stars.
        // Use spherical coordinates for uniform distribution.
        float theta = atan(dir.z, dir.x); // -PI to PI
        float phi = asin(clamp(dir.y, -1.0, 1.0)); // 0 to PI/2

        // Quantize into a grid of small cells.
        float gridScale = 120.0;
        vec2 cell = floor(vec2(theta, phi) * gridScale);

        // Hash the cell coordinates.
        float h = fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
        float h2 = fract(sin(dot(cell, vec2(269.5, 183.3))) * 76543.2109);

        if (h > 0.985) {
            float brightness = (h - 0.985) / 0.015;
            brightness *= 0.3 + 0.7 * h2; // size variation
            // Slight color variation.
            vec3 starColor = mix(vec3(0.8, 0.85, 1.0), vec3(1.0, 0.9, 0.7), h2);
            sky += starColor * brightness * nightness * 0.5;
        }
    }

    // Cloud coverage — overcast gray sky blended over base sky.
    float cloudCoverage = scene.weatherInfo.w;
    if (cloudCoverage > 0.0) {
        // Overcast color: dark gray during day, darker at night.
        vec3 overcastDay = vec3(0.45, 0.48, 0.52);
        vec3 overcastNight = vec3(0.03, 0.04, 0.06);
        vec3 overcastColor = mix(overcastNight, overcastDay, dayFactor);

        // Procedural cloud-like noise variation using ray direction.
        float wTime = scene.windInfo.w;
        float cloudNoise = fract(sin(dot(dir.xz * 8.0 + vec2(wTime * 0.02), vec2(127.1, 311.7))) * 43758.5453);
        float cloudVar = 0.85 + cloudNoise * 0.15;

        // Blend: more clouds = more gray sky, mostly above horizon.
        float aboveHorizon = smoothstep(-0.05, 0.3, y);
        sky = mix(sky, overcastColor * cloudVar, cloudCoverage * 0.85 * aboveHorizon);

        // Dim sun/moon disc behind clouds.
        // (Already computed sun disc above, so this just darkens the overall result.)
    }

    // Rain — darken sky further, heavier fog.
    float rainIntensity = scene.weatherInfo.x;
    if (rainIntensity > 0.0) {
        vec3 rainSkyColor = mix(vec3(0.02, 0.03, 0.04), vec3(0.3, 0.32, 0.36), dayFactor);
        float aboveHorizon = smoothstep(-0.1, 0.2, y);
        sky = mix(sky, rainSkyColor, rainIntensity * 0.5 * aboveHorizon);
    }

    // Fog weather — blend sky to uniform fog color.
    float fogDensity = scene.weatherInfo.y;
    if (fogDensity > 0.0) {
        vec3 fogColor = vec3(0.55, 0.58, 0.62) * max(dayFactor * 1.2, 0.15);
        sky = mix(sky, fogColor, fogDensity * 0.9);
    }

    // Lightning flash — brighten sky.
    float lightningFlash = scene.weatherInfo.z;
    if (lightningFlash > 0.0) {
        sky = mix(sky, vec3(0.9, 0.92, 1.0), lightningFlash * 0.6);
    }

    // Blizzard: whiteout the sky.
    float snowIntensity = scene.blizzardInfo.x;
    if (snowIntensity > 0.0) {
        vec3 fogColor = vec3(0.75, 0.78, 0.82) * max(dayFactor * 1.2, 0.15);
        sky = mix(sky, fogColor, snowIntensity * 0.9);
    }

    payload = sky;
}
