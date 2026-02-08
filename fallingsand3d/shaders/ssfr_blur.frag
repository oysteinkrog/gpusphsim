#version 330 core

in vec2 vUV;

uniform sampler2D uDepthTex;    // R32F eye-space depth
uniform vec2 uTexelSize;        // 1.0 / resolution
uniform vec2 uBlurDir;          // (1,0) for horizontal, (0,1) for vertical
uniform float uFilterRadius;    // blur kernel half-size in pixels (default 10)
uniform float uDepthRange;      // narrow range threshold (world units, default 0.1)

out float FragDepth;

void main() {
    float center = texture(uDepthTex, vUV).r;

    // Background: no depth written (0.0 or very large negative)
    if (center >= -0.01) {
        FragDepth = center;
        return;
    }

    float sum = 0.0;
    float wsum = 0.0;

    int radius = int(uFilterRadius);

    for (int i = -radius; i <= radius; i++) {
        vec2 offset = float(i) * uBlurDir * uTexelSize;
        float sample_depth = texture(uDepthTex, vUV + offset).r;

        // Skip background pixels
        if (sample_depth >= -0.01) continue;

        // Narrow-range bilateral weight: reject samples far from center depth
        float depth_diff = abs(sample_depth - center);
        if (depth_diff > uDepthRange) continue;

        // Gaussian spatial weight
        float spatial = exp(-0.5 * float(i * i) / (uFilterRadius * uFilterRadius * 0.16));
        // Depth weight
        float range_w = exp(-0.5 * (depth_diff * depth_diff) / (uDepthRange * uDepthRange * 0.25));

        float w = spatial * range_w;
        sum += sample_depth * w;
        wsum += w;
    }

    FragDepth = (wsum > 0.0) ? (sum / wsum) : center;
}
