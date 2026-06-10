#version 330

// CRT + glitch fragment shader for Mr. House's portrait.
// Combines: barrel/curvature distortion, scanlines, chromatic aberration,
// horizontal "tearing" glitch bursts, RGB channel shift, rolling flicker and a
// vignette. The `u_glitch` uniform is driven up while Mr. House speaks.

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_tex;
uniform float u_time;
uniform float u_scanline_intensity;
uniform float u_scanline_count;
uniform float u_chromatic;
uniform float u_vignette;
uniform float u_flicker;
uniform float u_glitch;       // 0..N, scaled by speaking state
uniform float u_curvature;

// --- cheap hash noise ---
float hash(float n) { return fract(sin(n) * 43758.5453123); }
float hash2(vec2 p)  { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

// Barrel distortion to fake CRT glass curvature.
vec2 curve(vec2 uv, float amount) {
    uv = uv * 2.0 - 1.0;
    vec2 offset = abs(uv.yx) / vec2(1.0 / amount + 1.0);
    uv = uv + uv * offset * offset;
    return uv * 0.5 + 0.5;
}

void main() {
    vec2 uv = curve(v_uv, u_curvature);

    // Off-screen (curved past the edge) => black border.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float t = u_time;
    float glitch = u_glitch;

    // --- horizontal tearing: shift whole scanline rows occasionally ---
    float lineNoise = hash(floor(uv.y * 80.0) + floor(t * 12.0));
    float tear = step(0.985 - 0.05 * glitch, lineNoise) * (hash2(vec2(uv.y, t)) - 0.5);
    uv.x += tear * 0.06 * glitch;

    // --- block glitch: occasional rectangular displacement ---
    float blockY = floor(uv.y * 24.0);
    float blockTrig = step(0.92 - 0.06 * glitch, hash(blockY + floor(t * 6.0)));
    uv.x += blockTrig * (hash(blockY + t) - 0.5) * 0.04 * glitch;

    // --- chromatic aberration / RGB split (stronger during glitch) ---
    float ca = u_chromatic + 0.004 * glitch * blockTrig;
    float r = texture(u_tex, uv + vec2(ca, 0.0)).r;
    float g = texture(u_tex, uv).g;
    float b = texture(u_tex, uv - vec2(ca, 0.0)).b;
    vec3 col = vec3(r, g, b);

    // --- random color-channel dropout flashes ---
    float drop = step(0.995 - 0.02 * glitch, hash(floor(t * 20.0)));
    col *= mix(vec3(1.0), vec3(hash(t), hash(t + 1.0), hash(t + 2.0)), drop * 0.4 * glitch);

    // --- scanlines ---
    float scan = sin(uv.y * u_scanline_count * 3.14159) * 0.5 + 0.5;
    col *= 1.0 - u_scanline_intensity * scan;

    // Subtle vertical RGB mask (aperture grille feel).
    float mask = 0.92 + 0.08 * sin(uv.x * 1200.0);
    col *= mask;

    // --- rolling brightness bar ---
    float roll = sin((uv.y + t * 0.15) * 6.28318) * 0.02;
    col += roll;

    // --- flicker ---
    col *= 1.0 - u_flicker * (hash(floor(t * 24.0)) - 0.5);

    // --- vignette ---
    vec2 vd = uv - 0.5;
    float vig = 1.0 - dot(vd, vd) * u_vignette * 2.5;
    col *= clamp(vig, 0.0, 1.0);

    // Slight overall warm phosphor tint.
    col *= vec3(1.02, 1.0, 0.97);

    f_color = vec4(col, 1.0);
}

