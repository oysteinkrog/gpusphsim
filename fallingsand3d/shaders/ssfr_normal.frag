#version 330 core

in vec2 vUV;

uniform sampler2D uDepthTex;  // smoothed R32F eye-space depth
uniform vec2 uTexelSize;
uniform mat4 uProjInv;        // inverse projection for unprojection

out vec4 FragNormal;  // RGB16F: eye-space normal (xyz), w unused

vec3 uvToEye(vec2 uv, float depth) {
    // Reconstruct eye-space position from UV + linear depth
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clip = vec4(ndc, 0.0, 1.0);
    vec4 eye = uProjInv * clip;
    eye.xyz /= eye.w;
    // Scale ray to hit the correct depth plane
    return eye.xyz * (depth / eye.z);
}

void main() {
    float depth = texture(uDepthTex, vUV).r;

    // Background
    if (depth >= -0.01) {
        FragNormal = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // 5-tap central difference for better normals
    float dL = texture(uDepthTex, vUV - vec2(uTexelSize.x, 0.0)).r;
    float dR = texture(uDepthTex, vUV + vec2(uTexelSize.x, 0.0)).r;
    float dT = texture(uDepthTex, vUV + vec2(0.0, uTexelSize.y)).r;
    float dB = texture(uDepthTex, vUV - vec2(0.0, uTexelSize.y)).r;

    vec3 posC = uvToEye(vUV, depth);

    // Use forward/backward differences depending on which neighbor is valid
    vec3 ddx, ddy;

    if (dL < -0.01 && dR < -0.01) {
        ddx = uvToEye(vUV + vec2(uTexelSize.x, 0.0), dR) -
              uvToEye(vUV - vec2(uTexelSize.x, 0.0), dL);
    } else if (dR < -0.01) {
        ddx = (uvToEye(vUV + vec2(uTexelSize.x, 0.0), dR) - posC) * 2.0;
    } else if (dL < -0.01) {
        ddx = (posC - uvToEye(vUV - vec2(uTexelSize.x, 0.0), dL)) * 2.0;
    } else {
        FragNormal = vec4(0.0, 0.0, 1.0, 0.0);
        return;
    }

    if (dT < -0.01 && dB < -0.01) {
        ddy = uvToEye(vUV + vec2(0.0, uTexelSize.y), dT) -
              uvToEye(vUV - vec2(0.0, uTexelSize.y), dB);
    } else if (dT < -0.01) {
        ddy = (uvToEye(vUV + vec2(0.0, uTexelSize.y), dT) - posC) * 2.0;
    } else if (dB < -0.01) {
        ddy = (posC - uvToEye(vUV - vec2(0.0, uTexelSize.y), dB)) * 2.0;
    } else {
        FragNormal = vec4(0.0, 0.0, 1.0, 0.0);
        return;
    }

    vec3 normal = normalize(cross(ddx, ddy));

    // Ensure normal points toward camera (positive Z in eye space)
    if (normal.z < 0.0) normal = -normal;

    FragNormal = vec4(normal, 1.0);
}
