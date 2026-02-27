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
    vec4 eyePos = uMV * vec4(aPos.xyz, 1.0);
    vEyeZ = eyePos.z;

    gl_Position = uMVP * vec4(aPos.xyz, 1.0);

    float dist = length(eyePos.xyz);
    float ps = clamp(uPointScale / dist, 1.0, 64.0);
    gl_PointSize = ps;
    vPointSize = ps;
}
