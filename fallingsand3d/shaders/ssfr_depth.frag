#version 330 core

in float vEyeZ;
in float vPointSize;

uniform mat4 uProj;
uniform float uParticleRadius;  // world-space radius of particle

out float FragDepth;  // R32F output: eye-space depth

void main() {
    // Sphere impostor: discard outside circle
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0)
        discard;

    // Sphere depth correction: shift depth inward by the sphere's z-offset
    float z_offset = sqrt(1.0 - r2) * uParticleRadius;
    float eyeZ = vEyeZ + z_offset;  // less negative = closer to camera

    // Write linear eye-space depth (negative Z)
    FragDepth = eyeZ;

    // Also update hardware depth buffer for correct occlusion
    // Convert back to NDC: z_ndc = (proj[2][2] * eyeZ + proj[3][2]) / (-eyeZ)
    float z_ndc = (uProj[2][2] * eyeZ + uProj[3][2]) / (-eyeZ);
    gl_FragDepth = z_ndc * 0.5 + 0.5;  // [0, 1] range
}
