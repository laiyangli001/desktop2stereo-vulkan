#import "macos_coreml_io.h"

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

enum {
    D2S_COREML_OK = 0,
    D2S_COREML_ERROR = -1,
    D2S_COREML_BUSY = -2,
    D2S_COREML_UNSUPPORTED = -3,
    D2S_COREML_OUTPUT_BACKING = -4,
};

typedef struct {
    uint32_t source_width;
    uint32_t source_height;
    uint32_t input_width;
    uint32_t input_height;
    float mean0;
    float mean1;
    float mean2;
    float std0;
    float std1;
    float std2;
} D2SPreprocessParams;

typedef struct {
    uint32_t count;
    float lo;
    float hi;
} D2SNormalizeParams;

typedef struct {
    uint32_t source_width;
    uint32_t source_height;
    uint32_t depth_width;
    uint32_t depth_height;
    uint32_t output_width;
    uint32_t output_height;
    float eye_offset;
    float depth_strength;
    float convergence;
    float smooth_texels;
} D2SWarpParams;

static const char *D2SMetalSource = R"D2S(
#include <metal_stdlib>
using namespace metal;

struct PreprocessParams {
    uint source_width;
    uint source_height;
    uint input_width;
    uint input_height;
    float mean0;
    float mean1;
    float mean2;
    float std0;
    float std1;
    float std2;
};

struct NormalizeParams {
    uint count;
    float lo;
    float hi;
};

struct WarpParams {
    uint source_width;
    uint source_height;
    uint depth_width;
    uint depth_height;
    uint output_width;
    uint output_height;
    float eye_offset;
    float depth_strength;
    float convergence;
    float smooth_texels;
};

kernel void d2s_preprocess(
    texture2d<float, access::sample> source [[texture(0)]],
    device float *output [[buffer(0)]],
    constant PreprocessParams& p [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.input_width || gid.y >= p.input_height) return;
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5f) /
        float2(p.input_width, p.input_height);
    float4 pixel = source.sample(s, uv);
    uint plane = p.input_width * p.input_height;
    uint offset = gid.y * p.input_width + gid.x;
    // MTLPixelFormatBGRA8Unorm exposes logical RGB components to Metal shaders.
    output[offset] = (pixel.r - p.mean0) / p.std0;
    output[plane + offset] = (pixel.g - p.mean1) / p.std1;
    output[2u * plane + offset] = (pixel.b - p.mean2) / p.std2;
}

kernel void d2s_normalize_half(
    device half *source [[buffer(0)]],
    device float *output [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant NormalizeParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) return;
    float value = float(source[gid]);
    if (!isfinite(value)) {
        value = 1.0f;
        atomic_fetch_add_explicit(status, 1u, memory_order_relaxed);
    }
    output[gid] = clamp((value - p.lo) / max(p.hi - p.lo, 1.0e-6f), 0.0f, 1.0f);
}

kernel void d2s_normalize_float(
    device float *source [[buffer(0)]],
    device float *output [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant NormalizeParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) return;
    float value = source[gid];
    if (!isfinite(value)) {
        value = 1.0f;
        atomic_fetch_add_explicit(status, 1u, memory_order_relaxed);
    }
    output[gid] = clamp((value - p.lo) / max(p.hi - p.lo, 1.0e-6f), 0.0f, 1.0f);
}

static inline float depth_sample(device float *depth, uint width, uint height,
                                 float u, float v) {
    float x = clamp(u * float(width) - 0.5f, 0.0f, float(width) - 1.001f);
    float y = clamp(v * float(height) - 0.5f, 0.0f, float(height) - 1.001f);
    int x0 = int(floor(x));
    int y0 = int(floor(y));
    int x1 = min(x0 + 1, int(width) - 1);
    int y1 = min(y0 + 1, int(height) - 1);
    float fx = x - float(x0);
    float fy = y - float(y0);
    float a = depth[y0 * width + x0];
    float b = depth[y0 * width + x1];
    float c = depth[y1 * width + x0];
    float d = depth[y1 * width + x1];
    return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

static inline float2 displaced_uv(float2 uv, float eye,
                                  device float *depth, uint depth_width,
                                  uint depth_height, constant WarpParams& p) {
    float du = p.smooth_texels / float(depth_width);
    float d0 = depth_sample(depth, depth_width, depth_height, uv.x, uv.y);
    float dm = depth_sample(depth, depth_width, depth_height, uv.x - du, uv.y);
    float dp = depth_sample(depth, depth_width, depth_height, uv.x + du, uv.y);
    float d = clamp(d0 * 0.7f + dm * 0.15f + dp * 0.15f, 0.0f, 1.0f);
    float shaped = d * (1.0f + 0.35f * (1.0f - d));
    float shift = (shaped - p.convergence) * p.depth_strength * eye;
    float edge = smoothstep(0.0f, 0.05f, uv.x) *
                 smoothstep(1.0f, 0.95f, uv.x);
    return float2(clamp(uv.x + shift * edge, 0.0f, 1.0f), uv.y);
}

kernel void d2s_warp_pack(
    texture2d<float, access::sample> color [[texture(0)]],
    device float *depth [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant WarpParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = p.output_width * p.output_height;
    if (gid >= total) return;
    uint x = gid % p.output_width;
    uint y = gid / p.output_width;
    uint eye_width = p.output_width / 2u;
    bool left = x < eye_width;
    uint eye_x = left ? x : x - eye_width;
    float eye = left ? -p.eye_offset : p.eye_offset;
    float2 uv = (float2(eye_x, y) + 0.5f) /
                float2(eye_width, p.output_height);
    float2 source_uv = displaced_uv(
        uv, eye, depth, p.depth_width, p.depth_height, p);
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float4 pixel = color.sample(s, source_uv);
    uint offset = gid * 4u;
    output[offset + 0u] = uchar(clamp(pixel.r * 255.0f + 0.5f, 0.0f, 255.0f));
    output[offset + 1u] = uchar(clamp(pixel.g * 255.0f + 0.5f, 0.0f, 255.0f));
    output[offset + 2u] = uchar(clamp(pixel.b * 255.0f + 0.5f, 0.0f, 255.0f));
    output[offset + 3u] = 255u;
}
)D2S";

typedef struct {
    __strong id<MTLBuffer> input_buffer;
    __strong id<MTLBuffer> raw_depth_buffer;
    __strong id<MTLBuffer> normalized_depth_buffer;
    __strong id<MTLBuffer> status_buffer;
    __strong id<MTLBuffer> packed_buffer;
    __strong id<MTLTexture> color_texture;
    __strong MLMultiArray *input_array;
    __strong MLMultiArray *output_array;
    __strong id<MLFeatureProvider> features;
    __strong MLPredictionOptions *prediction_options;
    __strong id<MLFeatureProvider> prediction;
    CVMetalTextureRef color_cv_texture;
    CVPixelBufferRef pixel_buffer;
    int32_t state;
    uint64_t frame_id;
    int32_t output_is_float16;
    float normalize_lo;
    float normalize_hi;
} D2SSlot;

typedef struct {
    __strong MLModel *model;
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLCommandQueue> pack_queue;
    __strong id<MTLComputePipelineState> preprocess_pipeline;
    __strong id<MTLComputePipelineState> normalize_half_pipeline;
    __strong id<MTLComputePipelineState> normalize_float_pipeline;
    __strong id<MTLComputePipelineState> warp_pipeline;
    CVMetalTextureCacheRef texture_cache;
    __strong NSString *input_name;
    __strong NSString *output_name;
    __strong NSArray<NSNumber *> *input_shape;
    __strong NSArray<NSNumber *> *output_shape;
    MLMultiArrayDataType output_type;
    int32_t input_width;
    int32_t input_height;
    int32_t depth_width;
    int32_t depth_height;
    size_t depth_count;
    D2SSlot slots[3];
    pthread_mutex_t mutex;
    int32_t mutex_initialized;
    char error[512];
} D2SCoreMLIO;

struct D2SLockGuard {
    pthread_mutex_t *mutex;
    explicit D2SLockGuard(pthread_mutex_t *value) : mutex(value) {
        pthread_mutex_lock(mutex);
    }
    ~D2SLockGuard() {
        pthread_mutex_unlock(mutex);
    }
};

static void d2s_set_error(D2SCoreMLIO *ctx, NSString *message) {
    const char *text = message.UTF8String ?: "unknown native CoreML error";
    snprintf(ctx->error, sizeof(ctx->error), "%s", text);
}

static void d2s_set_create_error(char *buffer, size_t capacity, NSString *message) {
    if (buffer == NULL || capacity == 0) return;
    const char *text = message.UTF8String ?: "unknown native CoreML error";
    snprintf(buffer, capacity, "%s", text);
}

static double d2s_now_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) return 0.0;
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

static MLComputeUnits d2s_compute_units(int32_t value) {
    switch (value) {
        case 0: return MLComputeUnitsCPUOnly;
        case 1: return MLComputeUnitsCPUAndGPU;
        case 3: return MLComputeUnitsCPUAndNeuralEngine;
        default: return MLComputeUnitsAll;
    }
}

static size_t d2s_shape_count(NSArray<NSNumber *> *shape) {
    size_t count = 1;
    for (NSNumber *value in shape) count *= (size_t)value.unsignedIntegerValue;
    return count;
}

static void d2s_release_slot(D2SSlot *slot) {
    // Core ML may reset its execution stream asynchronously after prediction
    // returns. Drop the bound objects before their Metal storage, and only
    // after the viewer has finished consuming this slot.
    slot->prediction = nil;
    slot->prediction_options = nil;
    slot->features = nil;
    slot->output_array = nil;
    slot->input_array = nil;
    // Core ML may finish releasing a backing array asynchronously after the
    // prediction call returns. Do not reuse these buffers across predictions.
    slot->input_buffer = nil;
    slot->raw_depth_buffer = nil;
    slot->normalized_depth_buffer = nil;
    slot->status_buffer = nil;
    slot->color_texture = nil;
    if (slot->color_cv_texture != NULL) {
        CFRelease(slot->color_cv_texture);
        slot->color_cv_texture = NULL;
    }
    if (slot->pixel_buffer != NULL) {
        CVPixelBufferRelease(slot->pixel_buffer);
        slot->pixel_buffer = NULL;
    }
    slot->state = 0;
    slot->frame_id = 0;
}

static void d2s_destroy_context(D2SCoreMLIO *ctx) {
    if (ctx == NULL) return;
    if (ctx->mutex_initialized) pthread_mutex_lock(&ctx->mutex);
    for (int i = 0; i < 3; ++i) {
        d2s_release_slot(&ctx->slots[i]);
        ctx->slots[i].packed_buffer = nil;
    }
    if (ctx->texture_cache != NULL) {
        CFRelease(ctx->texture_cache);
        ctx->texture_cache = NULL;
    }
    ctx->model = nil;
    ctx->device = nil;
    ctx->queue = nil;
    ctx->pack_queue = nil;
    ctx->preprocess_pipeline = nil;
    ctx->normalize_half_pipeline = nil;
    ctx->normalize_float_pipeline = nil;
    ctx->warp_pipeline = nil;
    ctx->input_name = nil;
    ctx->output_name = nil;
    ctx->input_shape = nil;
    ctx->output_shape = nil;
    if (ctx->mutex_initialized) {
        pthread_mutex_unlock(&ctx->mutex);
        pthread_mutex_destroy(&ctx->mutex);
        ctx->mutex_initialized = 0;
    }
    free(ctx);
}

static BOOL d2s_compile_pipelines(D2SCoreMLIO *ctx, NSError **error) {
    id<MTLLibrary> library = [ctx->device newLibraryWithSource:
                                                          [NSString stringWithUTF8String:D2SMetalSource]
                                                          options:nil
                                                            error:error];
    if (library == nil) return NO;
    id<MTLFunction> preprocess = [library newFunctionWithName:@"d2s_preprocess"];
    id<MTLFunction> norm_half = [library newFunctionWithName:@"d2s_normalize_half"];
    id<MTLFunction> norm_float = [library newFunctionWithName:@"d2s_normalize_float"];
    id<MTLFunction> warp = [library newFunctionWithName:@"d2s_warp_pack"];
    if (!preprocess || !norm_half || !norm_float || !warp) return NO;
    ctx->preprocess_pipeline = [ctx->device newComputePipelineStateWithFunction:preprocess error:error];
    ctx->normalize_half_pipeline = [ctx->device newComputePipelineStateWithFunction:norm_half error:error];
    ctx->normalize_float_pipeline = [ctx->device newComputePipelineStateWithFunction:norm_float error:error];
    ctx->warp_pipeline = [ctx->device newComputePipelineStateWithFunction:warp error:error];
    return ctx->preprocess_pipeline && ctx->normalize_half_pipeline &&
           ctx->normalize_float_pipeline && ctx->warp_pipeline;
}

static BOOL d2s_get_model_contract(D2SCoreMLIO *ctx, NSError **error) {
    MLModelDescription *description = ctx->model.modelDescription;
    NSDictionary<NSString *, MLFeatureDescription *> *inputs = description.inputDescriptionsByName;
    NSDictionary<NSString *, MLFeatureDescription *> *outputs = description.outputDescriptionsByName;
    if (inputs.count != 1 || outputs.count != 1) {
        d2s_set_error(ctx, @"native CoreML bridge requires exactly one input and one output");
        return NO;
    }
    ctx->input_name = inputs.allKeys.firstObject;
    ctx->output_name = outputs.allKeys.firstObject;
    MLFeatureDescription *input = inputs[ctx->input_name];
    MLFeatureDescription *output = outputs[ctx->output_name];
    if (input.type != MLFeatureTypeMultiArray || output.type != MLFeatureTypeMultiArray ||
        input.multiArrayConstraint.shape.count != 4 ||
        output.multiArrayConstraint.shape.count < 2) {
        d2s_set_error(ctx, @"native CoreML bridge requires array input/output features");
        return NO;
    }
    ctx->input_shape = input.multiArrayConstraint.shape;
    ctx->output_shape = output.multiArrayConstraint.shape;
    if (ctx->input_shape[0].intValue != 1 || ctx->input_shape[1].intValue != 3 ||
        ctx->input_shape[2].intValue != ctx->input_height ||
        ctx->input_shape[3].intValue != ctx->input_width) {
        d2s_set_error(ctx, @"CoreML input shape does not match the requested model resolution");
        return NO;
    }
    ctx->depth_height = ctx->output_shape[ctx->output_shape.count - 2].intValue;
    ctx->depth_width = ctx->output_shape.lastObject.intValue;
    ctx->depth_count = d2s_shape_count(ctx->output_shape);
    if (ctx->depth_count != (size_t)ctx->depth_height * (size_t)ctx->depth_width &&
        ctx->output_shape.count != 3) {
        d2s_set_error(ctx, @"unsupported CoreML depth output shape");
        return NO;
    }
    ctx->output_type = output.multiArrayConstraint.dataType;
    if (ctx->output_type != MLMultiArrayDataTypeFloat16 &&
        ctx->output_type != MLMultiArrayDataTypeFloat32) {
        d2s_set_error(ctx, @"native CoreML bridge supports Float16 or Float32 depth output only");
        return NO;
    }
    return YES;
}

void *d2s_coreml_io_create(const char *model_path, int32_t input_width,
                           int32_t input_height, int32_t compute_units,
                           char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        if (model_path == NULL || input_width <= 0 || input_height <= 0) {
            d2s_set_create_error(error_buffer, error_capacity, @"invalid native CoreML bridge arguments");
            return NULL;
        }
        D2SCoreMLIO *ctx = (D2SCoreMLIO *)calloc(1, sizeof(D2SCoreMLIO));
        if (ctx == NULL) {
            d2s_set_create_error(error_buffer, error_capacity, @"native CoreML bridge allocation failed");
            return NULL;
        }
        if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
            d2s_set_create_error(error_buffer, error_capacity, @"native CoreML bridge lock initialization failed");
            free(ctx);
            return NULL;
        }
        ctx->mutex_initialized = 1;
        ctx->input_width = input_width;
        ctx->input_height = input_height;
        ctx->device = MTLCreateSystemDefaultDevice();
        if (ctx->device == nil) {
            d2s_set_create_error(error_buffer, error_capacity, @"Metal device unavailable");
            d2s_destroy_context(ctx);
            return NULL;
        }
        ctx->queue = [ctx->device newCommandQueue];
        if (ctx->queue == nil) {
            d2s_set_create_error(error_buffer, error_capacity, @"Metal command queue unavailable");
            d2s_destroy_context(ctx);
            return NULL;
        }
        ctx->pack_queue = [ctx->device newCommandQueue];
        if (ctx->pack_queue == nil) {
            d2s_set_create_error(error_buffer, error_capacity, @"Metal pack command queue unavailable");
            d2s_destroy_context(ctx);
            return NULL;
        }
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:model_path]];
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = d2s_compute_units(compute_units);
        NSURL *load_url = url;
        if ([url.pathExtension.lowercaseString isEqualToString:@"mlpackage"]) {
            load_url = [MLModel compileModelAtURL:url error:&error];
        }
        if (load_url != nil) {
            ctx->model = [MLModel modelWithContentsOfURL:load_url
                                             configuration:configuration
                                                    error:&error];
        }
        if (ctx->model == nil) {
            d2s_set_create_error(error_buffer, error_capacity, error.localizedDescription ?: @"CoreML model load failed");
            d2s_destroy_context(ctx);
            return NULL;
        }
        if (!d2s_get_model_contract(ctx, &error)) {
            d2s_set_create_error(error_buffer, error_capacity, [NSString stringWithUTF8String:ctx->error]);
            d2s_destroy_context(ctx);
            return NULL;
        }
        if (!d2s_compile_pipelines(ctx, &error)) {
            d2s_set_create_error(error_buffer, error_capacity, error.localizedDescription ?: @"Metal pipeline compilation failed");
            d2s_destroy_context(ctx);
            return NULL;
        }
        CVReturn status = CVMetalTextureCacheCreate(kCFAllocatorDefault, NULL,
                                                     ctx->device, NULL,
                                                     &ctx->texture_cache);
        if (status != kCVReturnSuccess || ctx->texture_cache == NULL) {
            d2s_set_create_error(error_buffer, error_capacity, @"CVMetalTextureCache creation failed");
            d2s_destroy_context(ctx);
            return NULL;
        }
        for (int i = 0; i < 3; ++i) ctx->slots[i].state = 0;
        return ctx;
    }
}

static int32_t d2s_find_slot(D2SCoreMLIO *ctx) {
    for (int32_t i = 0; i < 3; ++i) {
        if (ctx->slots[i].state == 0) return i;
    }
    return -1;
}

static BOOL d2s_dispatch_preprocess(D2SCoreMLIO *ctx, D2SSlot *slot,
                                    uint32_t source_width, uint32_t source_height) {
    id<MTLCommandBuffer> command = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:ctx->preprocess_pipeline];
    D2SPreprocessParams params = {
        source_width, source_height,
        (uint32_t)ctx->input_width, (uint32_t)ctx->input_height,
        0.485f, 0.456f, 0.406f,
        0.229f, 0.224f, 0.225f,
    };
    [encoder setTexture:slot->color_texture atIndex:0];
    [encoder setBuffer:slot->input_buffer offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    NSUInteger w = 8;
    NSUInteger h = 8;
    MTLSize grid = MTLSizeMake(ctx->input_width, ctx->input_height, 1);
    MTLSize groups = MTLSizeMake(
        (grid.width + w - 1) / w, (grid.height + h - 1) / h, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted;
}

static void d2s_encode_normalize(D2SCoreMLIO *ctx, D2SSlot *slot,
                                 id<MTLComputeCommandEncoder> encoder,
                                 float lo, float hi) {
    uint32_t count = (uint32_t)ctx->depth_count;
    D2SNormalizeParams params = {count, lo, hi};
    *(uint32_t *)slot->status_buffer.contents = 0;
    id<MTLComputePipelineState> pipeline = slot->output_is_float16
        ? ctx->normalize_half_pipeline : ctx->normalize_float_pipeline;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:slot->raw_depth_buffer offset:0 atIndex:0];
    [encoder setBuffer:slot->normalized_depth_buffer offset:0 atIndex:1];
    [encoder setBuffer:slot->status_buffer offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    NSUInteger w = 64;
    MTLSize grid = MTLSizeMake(count, 1, 1);
    [encoder dispatchThreads:grid
      threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
}

static float d2s_half_to_float(uint16_t value) {
    uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)value >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) { mantissa <<= 1; --exponent; }
            mantissa &= 0x3ffu;
            bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int d2s_float_compare(const void *left, const void *right) {
    const float a = *(const float *)left;
    const float b = *(const float *)right;
    return (a > b) - (a < b);
}

static uint32_t d2s_depth_range(D2SCoreMLIO *ctx, D2SSlot *slot,
                                float *lo, float *hi) {
    // Percentile clipping is only used to reject sparse model outliers. A
    // bounded, evenly spaced sample preserves that behavior without sorting
    // the full model output on the runtime thread.
    const size_t sample_capacity = ctx->depth_count < 2048 ? ctx->depth_count : 2048;
    const size_t sample_stride = (ctx->depth_count + sample_capacity - 1) /
                                 sample_capacity;
    float *values = (float *)malloc(sample_capacity * sizeof(float));
    if (values == NULL) {
        *lo = 0.0f;
        *hi = 1.0f;
        return 0;
    }
    uint32_t finite = 0;
    size_t sampled = 0;
    uint32_t nonfinite = 0;
    if (slot->output_is_float16) {
        const uint16_t *source = (const uint16_t *)slot->raw_depth_buffer.contents;
        for (size_t i = 0; i < ctx->depth_count; ++i) {
            float value = d2s_half_to_float(source[i]);
            if (isfinite(value)) {
                ++finite;
                if ((i % sample_stride) == 0 && sampled < sample_capacity) {
                    values[sampled++] = value;
                }
            }
            else ++nonfinite;
        }
    } else {
        const float *source = (const float *)slot->raw_depth_buffer.contents;
        for (size_t i = 0; i < ctx->depth_count; ++i) {
            float value = source[i];
            if (isfinite(value)) {
                ++finite;
                if ((i % sample_stride) == 0 && sampled < sample_capacity) {
                    values[sampled++] = value;
                }
            }
            else ++nonfinite;
        }
    }
    if (finite == 0 || sampled == 0) {
        *lo = 0.0f;
        *hi = 1.0f;
        free(values);
        return nonfinite;
    }
    qsort(values, sampled, sizeof(float), d2s_float_compare);
    size_t lo_index = (size_t)floor(0.02 * (double)(sampled - 1));
    size_t hi_index = (size_t)floor(0.98 * (double)(sampled - 1));
    *lo = values[lo_index];
    *hi = values[hi_index];
    if (!isfinite(*lo) || !isfinite(*hi) || *hi <= *lo) {
        *lo = 0.0f;
        *hi = 1.0f;
    }
    free(values);
    return nonfinite;
}

int32_t d2s_coreml_io_predict(void *handle, void *pixel_buffer,
                              uint64_t frame_id, D2SCoreMLIOResult *result) {
    @autoreleasepool {
        D2SCoreMLIO *ctx = (D2SCoreMLIO *)handle;
        if (ctx == NULL || pixel_buffer == NULL || result == NULL) return D2S_COREML_ERROR;
        int32_t slot_index;
        {
            D2SLockGuard lock(&ctx->mutex);
            slot_index = d2s_find_slot(ctx);
            if (slot_index < 0) {
                d2s_set_error(ctx, @"native CoreML resource ring is full");
                return D2S_COREML_BUSY;
            }
            // Reserve before touching Metal/Core ML so another prediction can
            // use a different ring slot while this one is in flight.
            ctx->slots[slot_index].state = 2;
        }
        D2SSlot *slot = &ctx->slots[slot_index];
        CVPixelBufferRef buffer = (CVPixelBufferRef)pixel_buffer;
        size_t width = CVPixelBufferGetWidth(buffer);
        size_t height = CVPixelBufferGetHeight(buffer);
        CVMetalTextureRef cv_texture = NULL;
        CVReturn texture_status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, ctx->texture_cache, buffer, NULL,
            MTLPixelFormatBGRA8Unorm, width, height, 0, &cv_texture);
        if (texture_status != kCVReturnSuccess || cv_texture == NULL) {
            d2s_set_error(ctx, @"ScreenCaptureKit pixel buffer could not become a Metal texture");
            return D2S_COREML_ERROR;
        }
        slot->color_cv_texture = cv_texture;
        slot->color_texture = CVMetalTextureGetTexture(slot->color_cv_texture);
        slot->pixel_buffer = CVPixelBufferRetain(buffer);
        size_t input_bytes =
            (size_t)(3 * ctx->input_width * ctx->input_height * sizeof(float));
        if (slot->input_buffer == nil || slot->input_buffer.length < input_bytes) {
            slot->input_buffer = [ctx->device newBufferWithLength:(NSUInteger)input_bytes
                                                             options:MTLResourceStorageModeShared];
        }
        size_t output_bytes = ctx->depth_count *
            (ctx->output_type == MLMultiArrayDataTypeFloat16 ? sizeof(uint16_t) : sizeof(float));
        if (slot->raw_depth_buffer == nil || slot->raw_depth_buffer.length < output_bytes) {
            slot->raw_depth_buffer = [ctx->device newBufferWithLength:output_bytes
                                                                options:MTLResourceStorageModeShared];
        }
        size_t normalized_bytes = ctx->depth_count * sizeof(float);
        if (slot->normalized_depth_buffer == nil ||
            slot->normalized_depth_buffer.length < normalized_bytes) {
            slot->normalized_depth_buffer = [ctx->device newBufferWithLength:normalized_bytes
                                                                       options:MTLResourceStorageModeShared];
        }
        if (slot->status_buffer == nil || slot->status_buffer.length < sizeof(uint32_t)) {
            slot->status_buffer = [ctx->device newBufferWithLength:sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];
        }
        slot->output_is_float16 = ctx->output_type == MLMultiArrayDataTypeFloat16;
        if (!slot->color_texture || !slot->input_buffer || !slot->raw_depth_buffer ||
            !slot->normalized_depth_buffer || !slot->status_buffer) {
            d2s_set_error(ctx, @"native CoreML Metal buffer allocation failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        double preprocess_start = d2s_now_ms();
        if (!d2s_dispatch_preprocess(ctx, slot, (uint32_t)width, (uint32_t)height)) {
            d2s_set_error(ctx, @"Metal CoreML preprocessing command failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        double preprocess_ms = d2s_now_ms() - preprocess_start;
        NSArray<NSNumber *> *input_strides = @[
            @(3 * ctx->input_width * ctx->input_height),
            @(ctx->input_width * ctx->input_height),
            @(ctx->input_width), @1];
        NSError *error = nil;
        MLMultiArray *input_array = [[MLMultiArray alloc]
            initWithDataPointer:slot->input_buffer.contents
            shape:ctx->input_shape dataType:MLMultiArrayDataTypeFloat32
            strides:input_strides deallocator:nil error:&error];
        if (input_array == nil) {
            d2s_set_error(ctx, error.localizedDescription ?: @"CoreML input binding failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        slot->input_array = input_array;
        NSArray<NSNumber *> *output_strides = @[
            @(ctx->depth_width * ctx->depth_height), @(ctx->depth_width), @1];
        MLMultiArray *output_array = [[MLMultiArray alloc]
            initWithDataPointer:slot->raw_depth_buffer.contents
            shape:ctx->output_shape dataType:ctx->output_type
            strides:output_strides deallocator:nil error:&error];
        if (output_array == nil) {
            d2s_set_error(ctx, error.localizedDescription ?: @"CoreML output binding failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        slot->output_array = output_array;
        MLFeatureValue *input_value = [MLFeatureValue featureValueWithMultiArray:input_array];
        MLDictionaryFeatureProvider *features = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{ctx->input_name: input_value} error:&error];
        if (features == nil) {
            d2s_set_error(ctx, error.localizedDescription ?: @"CoreML feature provider creation failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        slot->features = features;
        MLPredictionOptions *options = [MLPredictionOptions new];
        options.outputBackings = @{ctx->output_name: output_array};
        slot->prediction_options = options;
        double model_start = d2s_now_ms();
        id<MLFeatureProvider> prediction = [ctx->model predictionFromFeatures:features
                                                                       options:options
                                                                        error:&error];
        double model_ms = d2s_now_ms() - model_start;
        if (prediction == nil) {
            d2s_set_error(ctx, error.localizedDescription ?: @"CoreML prediction failed");
            d2s_release_slot(slot);
            return D2S_COREML_ERROR;
        }
        slot->prediction = prediction;
        MLFeatureValue *output_value = [prediction featureValueForName:ctx->output_name];
        MLMultiArray *returned_array = output_value.multiArrayValue;
        if (returned_array == nil ||
            (returned_array != output_array &&
             returned_array.dataPointer != output_array.dataPointer)) {
            d2s_set_error(ctx, @"CoreML ignored the requested shared output backing");
            d2s_release_slot(slot);
            return D2S_COREML_OUTPUT_BACKING;
        }
        double postprocess_start = d2s_now_ms();
        float lo = 0.0f, hi = 1.0f;
        uint32_t nonfinite = d2s_depth_range(ctx, slot, &lo, &hi);
        slot->normalize_lo = lo;
        slot->normalize_hi = hi;
        double postprocess_ms = d2s_now_ms() - postprocess_start;
        {
            D2SLockGuard lock(&ctx->mutex);
            slot->state = 1;
            slot->frame_id = frame_id;
        }
        memset(result, 0, sizeof(*result));
        result->slot = slot_index;
        result->source_width = (int32_t)width;
        result->source_height = (int32_t)height;
        result->input_width = ctx->input_width;
        result->input_height = ctx->input_height;
        result->depth_width = ctx->depth_width;
        result->depth_height = ctx->depth_height;
        result->input_shared = 1;
        result->output_backing_used = 1;
        result->output_zero_copy = 0;
        // Normalization is encoded together with the final warp in pack().
        // This keeps the depth publish path from waiting on a separate Metal
        // command and still sanitizes every value before sampling.
        result->finite_depth = (nonfinite == 0);
        result->nonfinite_count = nonfinite;
        result->normalize_lo = lo;
        result->normalize_hi = hi;
        result->preprocess_ms = preprocess_ms;
        result->model_ms = model_ms;
        result->postprocess_ms = postprocess_ms;
        return D2S_COREML_OK;
    }
}

int32_t d2s_coreml_io_pack(void *handle, int32_t slot_index, void *destination,
                           size_t destination_size, int32_t output_width,
                           int32_t output_height, int32_t output_format,
                           float eye_offset, float depth_strength,
                           float convergence, float smooth_texels) {
    @autoreleasepool {
        D2SCoreMLIO *ctx = (D2SCoreMLIO *)handle;
        if (ctx == NULL || destination == NULL || slot_index < 0 || slot_index >= 3 ||
            output_width <= 0 || output_height <= 0 || output_format < 0 || output_format > 1) {
            return D2S_COREML_ERROR;
        }
        D2SSlot *slot = &ctx->slots[slot_index];
        {
            D2SLockGuard lock(&ctx->mutex);
            if (slot->state != 1 ||
                destination_size < (size_t)output_width * output_height * 4u ||
                output_width % 2 != 0) {
                d2s_set_error(ctx, @"native CoreML pack slot or destination is invalid");
                return D2S_COREML_ERROR;
            }
        }
        size_t output_bytes = (size_t)output_width * (size_t)output_height * 4u;
        if (slot->packed_buffer == nil || slot->packed_buffer.length < output_bytes) {
            slot->packed_buffer = [ctx->device newBufferWithLength:(NSUInteger)output_bytes
                                                               options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> destination_buffer = slot->packed_buffer;
        if (destination_buffer == nil) {
            d2s_set_error(ctx, @"Metal native warp output buffer allocation failed");
            return D2S_COREML_ERROR;
        }
        D2SWarpParams params = {
            (uint32_t)CVPixelBufferGetWidth(slot->pixel_buffer),
            (uint32_t)CVPixelBufferGetHeight(slot->pixel_buffer),
            (uint32_t)ctx->depth_width,
            (uint32_t)ctx->depth_height,
            (uint32_t)output_width,
            (uint32_t)output_height,
            eye_offset, depth_strength, convergence, smooth_texels,
        };
        id<MTLCommandBuffer> command = [ctx->pack_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        d2s_encode_normalize(ctx, slot, encoder, slot->normalize_lo, slot->normalize_hi);
        [encoder setComputePipelineState:ctx->warp_pipeline];
        [encoder setTexture:slot->color_texture atIndex:0];
        [encoder setBuffer:slot->normalized_depth_buffer offset:0 atIndex:0];
        [encoder setBuffer:destination_buffer offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        NSUInteger threads = 64;
        MTLSize grid = MTLSizeMake((NSUInteger)output_width * output_height, 1, 1);
        [encoder dispatchThreads:grid
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            d2s_set_error(ctx, @"Metal native warp command failed");
            return D2S_COREML_ERROR;
        }
        // Vulkan's mapped MoltenVK allocation is CPU-visible but is not a
        // valid Metal bytesNoCopy GPU resource. Keep the warp GPU-side, then
        // perform one explicit handoff into Vulkan's mapped transfer buffer.
        memcpy(destination, destination_buffer.contents, output_bytes);
        return D2S_COREML_OK;
    }
}

int32_t d2s_coreml_io_release(void *handle, int32_t slot_index) {
    D2SCoreMLIO *ctx = (D2SCoreMLIO *)handle;
    if (ctx == NULL || slot_index < 0 || slot_index >= 3) return D2S_COREML_ERROR;
    {
        D2SLockGuard lock(&ctx->mutex);
        d2s_release_slot(&ctx->slots[slot_index]);
    }
    return D2S_COREML_OK;
}

const char *d2s_coreml_io_last_error(void *handle) {
    D2SCoreMLIO *ctx = (D2SCoreMLIO *)handle;
    return ctx == NULL ? "native CoreML bridge is not initialized" : ctx->error;
}

void d2s_coreml_io_destroy(void *handle) {
    @autoreleasepool {
        D2SCoreMLIO *ctx = (D2SCoreMLIO *)handle;
        d2s_destroy_context(ctx);
    }
}
