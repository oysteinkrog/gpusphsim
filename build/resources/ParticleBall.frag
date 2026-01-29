#version 330 core

// Inputs from vertex shader
in vec4 vColor;

// Output
out vec4 fragColor;

void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // Calculate normal from point sprite coordinates
    // gl_PointCoord goes from 0 to 1, convert to -1 to 1
    vec3 N;
    N.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0 - mag);

    // Calculate lighting
    float diffuse = 0.5 + 0.5 * max(0.0, dot(lightDir, N));

    float alpha = 0.5;
    fragColor = vec4(vColor.rgb * diffuse, alpha);
}
