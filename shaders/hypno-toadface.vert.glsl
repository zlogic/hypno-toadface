#version 460

/*
 * Generate a quad without any input data - instead, output vertices using only input vertex indices as input data.
 * From https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
 */

void main() {
    vec2 outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0f + -1.0, 0.0f, 1.0f);
}
