#version 330

// Fullscreen-quad vertex shader. Positions arrive in clip space [-1, 1] and we
// pass through texture coordinates for the fragment shader.
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}

