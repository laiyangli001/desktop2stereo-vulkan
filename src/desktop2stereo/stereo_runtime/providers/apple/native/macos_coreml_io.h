#ifndef D2S_MACOS_COREML_IO_H
#define D2S_MACOS_COREML_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t slot;
    int32_t source_width;
    int32_t source_height;
    int32_t input_width;
    int32_t input_height;
    int32_t depth_width;
    int32_t depth_height;
    int32_t input_shared;
    int32_t output_backing_used;
    int32_t output_zero_copy;
    int32_t finite_depth;
    uint32_t nonfinite_count;
    float normalize_lo;
    float normalize_hi;
    double preprocess_ms;
    double model_ms;
    double postprocess_ms;
} D2SCoreMLIOResult;

void *d2s_coreml_io_create(const char *model_path, int32_t input_width,
                           int32_t input_height, int32_t compute_units,
                           char *error_buffer, size_t error_capacity);

int32_t d2s_coreml_io_predict(void *handle, void *pixel_buffer,
                             uint64_t frame_id, D2SCoreMLIOResult *result);

int32_t d2s_coreml_io_pack(void *handle, int32_t slot, void *destination,
                           size_t destination_size, int32_t output_width,
                           int32_t output_height, int32_t output_format,
                           float eye_offset, float depth_strength,
                           float convergence, float smooth_texels);

int32_t d2s_coreml_io_release(void *handle, int32_t slot);
const char *d2s_coreml_io_last_error(void *handle);
void d2s_coreml_io_destroy(void *handle);

#ifdef __cplusplus
}
#endif

#endif
