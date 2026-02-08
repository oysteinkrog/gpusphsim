#version 330 core

layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aColor;

uniform mat4 uMVP;
uniform mat4 uMV;
uniform float uPointScale;

out vec4 vColor;

void main() {
    float behavior = aColor.w;

    vec4 eyePos = uMV * vec4(aPos.xyz, 1.0);
    gl_Position = uMVP * vec4(aPos.xyz, 1.0);

    float dist = length(eyePos.xyz);
    gl_PointSize = clamp(uPointScale / dist, 1.0, 64.0);

    vColor = aColor;

    // Cull non-FLUID
    if (behavior > 0.1) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
    }
}
