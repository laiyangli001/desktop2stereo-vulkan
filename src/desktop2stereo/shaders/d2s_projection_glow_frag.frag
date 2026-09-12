#version 450

layout(set = 0, binding = 0) uniform sampler2D glow_texture;
layout(set = 0, binding = 1, std430) readonly buffer GlowState {
    vec4 head_mode;
    vec4 glow;
    vec4 geometry;
    vec4 veil;
    vec4 reserved;
    vec4 shell;
} state;

layout(location = 0) in vec2 texture_uv;
layout(location = 1) in vec2 effect_uv;
layout(location = 0) out vec4 output_color;

vec3 sample_region_cell(vec2 cell, vec2 grid) {
    vec2 q = (clamp(cell, vec2(0.0), grid - vec2(1.0)) + vec2(0.5)) / grid;
    // The Vulkan Glow producer already mirrors the source rows to preserve
    // the legacy Filament external-image convention. Direct Vulkan sampling
    // must not apply Filament's second texture-coordinate inversion.
    return textureLod(glow_texture, q, 0.0).rgb;
}

vec3 sample_region_average(vec2 p) {
    vec2 grid = vec2(8.0, 6.0);
    vec2 position = clamp(p, vec2(0.0), vec2(1.0)) * grid - vec2(0.5);
    vec2 base = floor(position);
    vec2 blend = fract(position);
    blend = blend * blend * (vec2(3.0) - vec2(2.0) * blend);
    vec3 lower = mix(
        sample_region_cell(base, grid),
        sample_region_cell(base + vec2(1.0, 0.0), grid),
        blend.x
    );
    vec3 upper = mix(
        sample_region_cell(base + vec2(0.0, 1.0), grid),
        sample_region_cell(base + vec2(1.0, 1.0), grid),
        blend.x
    );
    return mix(lower, upper, blend.y);
}

void screen_glow() {
    float base_width = max(state.glow.y, 0.75);
    float screen_long = max(state.shell.z, 2.4);
    float distance_to_head = max(state.shell.w, 0.5);
    float range = base_width * (screen_long / 2.4)
        * (distance_to_head / 2.0) * 20.0;
    vec2 glow_size = state.geometry.xy;
    vec2 screen_half = vec2(
        state.geometry.z / max(glow_size.x, 0.00001),
        state.geometry.w / max(glow_size.y, 0.00001)
    );
    vec2 centered = texture_uv - vec2(0.5);
    vec2 p = abs(centered) - screen_half;
    float signed_distance = max(p.x, p.y);
    float inner = min(screen_half.x, screen_half.y) * 0.075;
    bool inside = signed_distance <= 0.0;
    if (inside) {
        if (inner <= 0.0 || signed_distance < -inner) discard;
    }
    float edge_position = inside ? signed_distance + inner : signed_distance;
    float glow_inverse_range = max(glow_size.x, glow_size.y) / max(range, 0.0001);
    float edge_t = clamp(edge_position * glow_inverse_range, 0.0, 1.0);
    float amount = pow(max(1.0 - smoothstep(0.0, 1.0, edge_t), 0.0), 0.58);
    if (inside) {
        float density = smoothstep(0.0, max(inner, 0.000001), signed_distance + inner);
        density = density * density * (3.0 - 2.0 * density);
        amount *= density;
    }
    amount *= state.glow.x * state.glow.z * state.veil.y;
    if (amount <= 0.001) discard;
    vec2 raw = (texture_uv - (vec2(0.5) - screen_half)) /
        max(screen_half * 2.0, vec2(0.00001));
    vec2 content_uv = raw;
    vec2 direction = normalize(content_uv - vec2(0.5) + vec2(0.00001, 0.0));
    vec2 denominator = max(abs(direction), vec2(0.00001));
    float edge_scale = min(0.5 / denominator.x, 0.5 / denominator.y);
    vec2 edge_uv = clamp(vec2(0.5) + direction * edge_scale, vec2(0.0), vec2(1.0));
    float blur = smoothstep(0.0, 1.0, edge_t);
    vec2 light_uv = edge_uv - direction * mix(0.015, 0.060, blur);
    vec3 local_color = textureLod(
        glow_texture, clamp(light_uv, vec2(0.0), vec2(1.0)), 0.0
    ).rgb;
    vec3 edge_color = textureLod(glow_texture, edge_uv, 0.0).rgb;
    vec3 color = mix(local_color, edge_color, blur * 0.5);
    output_color = vec4(color * min(amount, 1.0), 1.0);
}

void veil_glow() {
    vec2 uv = clamp(texture_uv, vec2(0.0), vec2(1.0));
    float depth = clamp(effect_uv.x, 0.0, 1.0);
    float thickness = 3.0;
    float beam = exp(-depth / 0.34);
    beam = pow(max(beam, 0.0), 1.0 / thickness);
    float inset = 0.02;
    float edge_distance = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    float edge = 1.0 - smoothstep(inset, inset * 4.0, edge_distance);
    vec2 sample_uv = uv;
    vec3 source = textureLod(glow_texture, sample_uv, 0.0).rgb;
    float alpha = edge * beam * state.veil.y * state.veil.x * state.glow.z;
    if (alpha <= 0.002) discard;
    alpha = min(alpha, 1.0);
    output_color = vec4(source * alpha, 1.0);
}

void surround_glow() {
    float radial_distance = clamp(effect_uv.x, 0.0, 1.0);
    float edge_field = exp2(-5.0 * radial_distance)
        * (1.0 - smoothstep(0.88, 1.0, radial_distance));
    float amount = edge_field * state.glow.x * state.glow.w * state.veil.y;
    if (amount <= 0.002) discard;
    vec3 shell_color = sample_region_average(texture_uv);
    output_color = vec4(shell_color * min(amount, 1.0), 1.0);
}

void main() {
    int mode = int(round(state.head_mode.w));
    if (mode == 1) {
        screen_glow();
    } else if (mode == 2) {
        veil_glow();
    } else {
        surround_glow();
    }
}
