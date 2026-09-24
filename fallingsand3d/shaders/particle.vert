#version 330 core

layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in vec4 aVel;

uniform mat4 uMVP;
uniform mat4 uMV;
uniform float uPointScale;
uniform float uTrailScale;

out vec4 vColor;

void main() {
    // DEAD slot (color.w < 0): drop the vertex so killed particles vanish
    // before compaction.
    if (aColor.w < 0.0) {
        vColor = aColor;
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        return;
    }

    vec4 eyePos = uMV * vec4(aPos.xyz, 1.0);
    float dist = length(eyePos.xyz);
    float baseSize = uPointScale / dist;

    // Speed-scaled point size for motion blur effect. With blur off the
    // renderer binds a VAO without attribute 2 and sets uTrailScale = 0.
    float stretch = (uTrailScale > 0.0)
        ? 1.0 + uTrailScale * min(length(aVel.xyz) * 0.3, 3.0)
        : 1.0;
    gl_PointSize = clamp(baseSize * stretch, 1.0, 128.0);

    gl_Position = uMVP * vec4(aPos.xyz, 1.0);
    vColor = aColor;
}
