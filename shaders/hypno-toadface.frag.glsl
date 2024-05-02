#version 460

layout (binding = 0) uniform UBO {
    vec4 params;
} ubo;

layout (location = 0) out vec4 outFragColor;

float hue_to_rgb(float p, float q, float t) {
    if (t < 0.0)
        t += 1.0;
    if (t > 1.0)
        t -= 1.0;
    if (t < 1.0/6.0)
        return p + (q - p) * 6.0 * t;
    if (t < 1.0/2.0)
        return q;
    if (t < 2.0/3.0)
        return p + (q - p) * (2.0/3.0 - t) * 6.0;
    return p;
}

vec3 hsl_to_rgb(float h, float s, float l) {
    float q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    float p = 2.0 * l - q;

    const float r = hue_to_rgb(p, q, h + 1.0/3.0);
    const float g = hue_to_rgb(p, q, h);
    const float b = hue_to_rgb(p, q, h - 1.0/3.0);

    return vec3(r, g, b);
}

vec3 hsv_to_rgb(float h, float s, float v) {
    const float i = floor(h * 6.0);
    const float f = h * 6.0 - i;
    const float p = v * (1.0 - s);
    const float q = v * (1.0 - f * s);
    const float t = v * (1.0 - (1.0 - f) * s);
    const float h_sector = int(i) % 6;
    vec3 rgb = vec3(0.0, 0.0, 0.0);
    rgb += h_sector == 0 ? vec3(v, t, p) : vec3(0.0, 0.0, 0.0);
    rgb += h_sector == 1 ? vec3(q, v, p) : vec3(0.0, 0.0, 0.0);
    rgb += h_sector == 2 ? vec3(p, v, t) : vec3(0.0, 0.0, 0.0);
    rgb += h_sector == 3 ? vec3(p, q, v) : vec3(0.0, 0.0, 0.0);
    rgb += h_sector == 4 ? vec3(t, p, v) : vec3(0.0, 0.0, 0.0);
    rgb += h_sector == 5 ? vec3(v, p, q) : vec3(0.0, 0.0, 0.0);
    return rgb;
}

void main() {
    const float timecode = ubo.params[0];
    const float center_width = ubo.params[1];
    const float center_height = ubo.params[2];
    const float max_l = ubo.params[3];
    const float dx = gl_FragCoord.x - center_width;
    const float dy = gl_FragCoord.y - center_height;

    const float dist = (dx * dx + dy * dy)/max_l;

    // Increase multiplier to 500 or even more for a moire pattern
    const float h = dist * 128.0 + timecode;
    const float s = 1.0;
    const float v = 1.0;
    // Shows a tunnel vision effect
    /*
    const float s = clamp(1.0 - dist, 0.0, 1.0);
    const float v = clamp(1.0 - dist, 0.0, 1.0);
    */
    const float l = v * 0.5;
    const vec3 rgb = hsl_to_rgb(h - floor(h), s, l);
    //const vec3 rgb = hsv_to_rgb(h - floor(h), s, v);
    // The sRGB colorspace applies weird gamma correction, should use UNORM if possible
    outFragColor = vec4(rgb, 1.0);
}
