#version 330 core

layout(location = 0) in vec3 aPos;

uniform mat4 uViewRot;   // view matrix with translation zeroed (rotation only)
uniform mat4 uProj;

out vec3 vTexCoord;

void main() {
    vTexCoord = aPos;
    vec4 pos = uProj * uViewRot * vec4(aPos, 1.0);
    gl_Position = pos.xyww;  // depth = 1.0 (farthest, behind everything)
}
