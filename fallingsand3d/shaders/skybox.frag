#version 330 core

in vec3 vTexCoord;

uniform samplerCube uSkybox;

out vec4 FragColor;

void main() {
    FragColor = texture(uSkybox, vTexCoord);
}
