#version 330 core

in vec2 vUV;

uniform sampler2D uDepthTex;      // smoothed eye-space depth (R32F)
uniform sampler2D uNormalTex;     // eye-space normal (RGB16F)
uniform sampler2D uThicknessTex;  // accumulated thickness (R16F)
uniform sampler2D uSceneTex;      // background scene (for refraction)
uniform samplerCube uSkybox;     // cubemap for environment reflections

uniform mat4 uProjInv;
uniform mat3 uViewRotInv;        // inverse view rotation (eye->world for reflections)
uniform vec2 uTexelSize;

// Material parameters
uniform vec3 uAbsorption;     // Beer-Lambert absorption color (default: blue tint)
uniform float uAbsorptionScale;  // absorption strength multiplier
uniform vec3 uFluidColor;    // base fluid color
uniform float uFresnelPower;  // Fresnel exponent (default 5.0)
uniform float uFresnelBias;   // Fresnel minimum reflectance (default 0.02)
uniform float uSpecularPower; // specular exponent (default 64.0)

out vec4 FragColor;

vec3 uvToEye(vec2 uv, float depth) {
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clip = vec4(ndc, 0.0, 1.0);
    vec4 eye = uProjInv * clip;
    eye.xyz /= eye.w;
    return eye.xyz * (depth / eye.z);
}

void main() {
    float depth = texture(uDepthTex, vUV).r;

    // Background: pass through scene
    if (depth >= -0.01) {
        FragColor = texture(uSceneTex, vUV);
        return;
    }

    vec3 normal = texture(uNormalTex, vUV).xyz;
    float thickness = texture(uThicknessTex, vUV).r;

    // Reconstruct eye-space position
    vec3 eyePos = uvToEye(vUV, depth);
    vec3 viewDir = normalize(-eyePos);  // toward camera

    // --- Fresnel (Schlick approximation) ---
    float NdotV = max(dot(normal, viewDir), 0.0);
    float fresnel = uFresnelBias + (1.0 - uFresnelBias) * pow(1.0 - NdotV, uFresnelPower);
    fresnel = clamp(fresnel, 0.0, 1.0);

    // --- Beer-Lambert absorption ---
    // Transmitted light is attenuated exponentially with thickness
    vec3 absorption = exp(-uAbsorption * thickness * uAbsorptionScale);

    // --- Refraction (simple screen-space offset) ---
    vec2 refractionOffset = normal.xy * uTexelSize * 20.0;
    vec3 background = texture(uSceneTex, vUV + refractionOffset).rgb;

    // Apply absorption to refracted background
    vec3 refracted = background * absorption * uFluidColor;

    // --- Diffuse lighting ---
    // Directional light from upper-right (eye-space)
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.5));
    float NdotL = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = uFluidColor * NdotL * 0.4;

    // --- Body color from absorption ---
    // Thick regions show the fluid's own color (light scattered within the medium)
    float opacity = 1.0 - (absorption.r + absorption.g + absorption.b) / 3.0;
    vec3 bodyColor = uFluidColor * opacity * 0.35;

    // --- Specular highlight ---
    vec3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVec), 0.0), uSpecularPower);
    vec3 specular = vec3(1.0) * spec * 0.6;

    // --- Environment reflection (cubemap skybox) ---
    vec3 reflectDir = reflect(-viewDir, normal);
    // Transform from eye-space to world-space for cubemap lookup
    vec3 worldReflect = uViewRotInv * reflectDir;
    vec3 envColor = texture(uSkybox, worldReflect).rgb;

    // --- Combine: Fresnel blend between refraction and reflection ---
    vec3 color = mix(refracted + diffuse + bodyColor, envColor, fresnel) + specular;

    // Slight edge darkening for depth cue
    float edge = 1.0 - pow(1.0 - NdotV, 2.0);
    color *= mix(0.8, 1.0, edge);

    FragColor = vec4(color, 1.0);
}
