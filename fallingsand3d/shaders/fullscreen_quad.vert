#version 330 core

// Fullscreen triangle: 3 vertices, no VBO needed.
// Use glDrawArrays(GL_TRIANGLES, 0, 3) with an empty VAO.

out vec2 vUV;

void main() {
    // Generate fullscreen triangle from vertex ID
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;
    vUV = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
