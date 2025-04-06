#include <cstddef>

struct RenderUtils {
    static void *temp_storage;
    static size_t temp_storage_size;
    static void sort_gaussians_gpu(float* keys_in, float* keys_out, int* val_in, int* val_out, int num);
};


