#version 330 core

// Foam/spray/bubble vertex shader.
// Renders secondary particles as small point sprites.
// Position in aPos.xyz, type in aPos.w (0=spray, 1=foam, 2=bubble, -1=dead).

layout(location = 0) in vec4 aPos;

uniform mat4 uMVP;
uniform mat4 uMV;
uniform float uPointScale;

out float vAlpha;

void main() {
    // Dead foam (type -1, see foam.cu): move it outside clip space.
    if (aPos.w < 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        vAlpha = 0.0;
        return;
    }

    vec4 eyePos = uMV * vec4(aPos.xyz, 1.0);
    float dist = length(eyePos.xyz);
    gl_Position = uMVP * vec4(aPos.xyz, 1.0);

    // Smaller point size than main particles
    gl_PointSize = clamp(uPointScale * 0.4 / dist, 1.0, 16.0);

    // Spray: bright white. Foam: softer. Bubble: faint.
    float foam_type = aPos.w;
    if (foam_type < 0.5) {
        vAlpha = 0.8;  // SPRAY
    } else if (foam_type < 1.5) {
        vAlpha = 0.5;  // FOAM
    } else {
        vAlpha = 0.3;  // BUBBLE
    }
}
