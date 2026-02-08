#version 330 core

// Same as particle.vert but culls FLUID particles (color.w < 0.1).
// Used when SSFR is enabled: FLUID rendered via SSFR, others via point sprites.

layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aColor;

uniform mat4 uMVP;
uniform mat4 uMV;
uniform float uPointScale;

out vec4 vColor;

void main() {
    float behavior = aColor.w;

    vec4 eyePos = uMV * vec4(aPos.xyz, 1.0);
    float dist = length(eyePos.xyz);
    gl_Position = uMVP * vec4(aPos.xyz, 1.0);
    gl_PointSize = clamp(uPointScale / dist, 1.0, 64.0);
    vColor = aColor;

    // Cull FLUID particles (behavior < 0.1) -- they're rendered by SSFR
    if (behavior < 0.1) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
    }
}
