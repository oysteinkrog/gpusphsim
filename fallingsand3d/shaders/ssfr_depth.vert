#version 330 core

layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aColor;

uniform mat4 uMVP;
uniform mat4 uMV;
uniform float uPointScale;
uniform mat4 uProj;

out float vEyeZ;      // eye-space Z (negative, closer = less negative)
out float vPointSize;  // for sphere impostor depth correction

void main() {
    // Filter: color.w encodes behavior class.
    // FLUID = 0.0, GRANULAR ~ 0.25, GAS ~ 0.5, STATIC ~ 0.75
    // We only want FLUID particles (w < 0.1)
    float behavior = aColor.w;

    vec4 eyePos = uMV * vec4(aPos.xyz, 1.0);
    vEyeZ = eyePos.z;

    gl_Position = uMVP * vec4(aPos.xyz, 1.0);

    float dist = length(eyePos.xyz);
    float ps = clamp(uPointScale / dist, 1.0, 64.0);
    gl_PointSize = ps;
    vPointSize = ps;

    // Cull non-FLUID by moving off-screen
    if (behavior > 0.1) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);  // clip away
        gl_PointSize = 0.0;
    }
}
