#version 330 core

in vec4 vColor;

out float FragThickness;  // R16F: additive thickness contribution

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0)
        discard;

    // Thickness contribution: sphere chord length at this pixel
    // Proportional to sqrt(1 - r^2) for a sphere cross-section
    float thickness = sqrt(1.0 - r2);

    // Scale by a constant to get reasonable accumulated thickness values
    FragThickness = thickness * 0.02;
}
