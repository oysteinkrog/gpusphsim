#version 330 core

// Foam/spray/bubble fragment shader.
// Renders as soft white circles with additive blending.

in float vAlpha;

out vec4 FragColor;

void main() {
    // Circular point sprite
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0)
        discard;

    // Soft falloff from center
    float falloff = 1.0 - r2;
    float alpha = vAlpha * falloff;

    // White with alpha for additive blend
    FragColor = vec4(1.0, 1.0, 1.0, alpha);
}
