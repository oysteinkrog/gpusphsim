#version 330 core

// Vertex attributes - must match Ogre's attribute names
in vec4 vertex;      // position (VES_POSITION)
in vec4 colour;      // color (VES_DIFFUSE)

// Outputs to fragment shader
out vec4 vColor;

// Uniforms
uniform mat4 worldViewProjMatrix;
uniform float pointRadius;
uniform float pointScale;

void main()
{
    // Calculate output position
    gl_Position = worldViewProjMatrix * vec4(vertex.xyz, 1.0);

    // Pass color through
    vColor = colour;

    // Calculate point size based on distance
    float dist = length(gl_Position.xyz);
    gl_PointSize = pointRadius * (pointScale / max(dist, 1.0));
}
