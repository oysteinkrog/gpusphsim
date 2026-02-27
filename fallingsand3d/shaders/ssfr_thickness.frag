#version 330 core

in vec4 vColor;

out vec4 FragColor;  // RGBA16F: (color.r*t, color.g*t, color.b*t, t)

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0)
        discard;

    // Thickness contribution: sphere chord length at this pixel
    // Proportional to sqrt(1 - r^2) for a sphere cross-section
    float base_t = sqrt(1.0 - r2) * 0.02;

    // Behavior class from color.w: FLUID=0.0, GRANULAR=0.25, GAS=0.5, STATIC=0.75
    // Per-behavior opacity:
    //   FLUID (water, oil, acid): translucent (1x)
    //   GRANULAR (sand, dirt, gravel): fully opaque (10x)
    //   GAS (steam, smoke, fire): wispy semi-transparent (0.5x)
    //   STATIC (stone, wood, metal, ice): fully opaque (10x)
    // The color-weighted average (rgb/a) is preserved since both scale equally.
    float behavior = vColor.w;
    float opacityMult;
    if (behavior < 0.1)       opacityMult = 1.0;   // FLUID
    else if (behavior < 0.4)  opacityMult = 10.0;   // GRANULAR
    else if (behavior < 0.6)  opacityMult = 0.5;    // GAS
    else                      opacityMult = 10.0;   // STATIC
    float t = base_t * opacityMult;

    // Output color-weighted thickness for per-pixel material color recovery
    // Additive blending accumulates: sum(c_i * t_i) in RGB, sum(t_i) in A
    // Per-pixel color = RGB / max(A, 0.001)
    FragColor = vec4(vColor.rgb * t, t);
}
