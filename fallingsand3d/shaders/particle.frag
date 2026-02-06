#version 330 core

in vec4 vColor;
out vec4 FragColor;

void main() {
    // Discard fragments outside the unit circle -> round point sprites
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0)
        discard;

    // Simple depth-based shading: darken towards edges for a sphere look
    float shade = 1.0 - 0.4 * r2;
    FragColor = vec4(vColor.rgb * shade, vColor.a);
}
