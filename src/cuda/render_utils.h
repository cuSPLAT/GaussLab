#include <cstddef>

struct RenderUtils {
    static void *temp_storage;
    static size_t temp_storage_size;
    static void sort_gaussians_gpu(int* keys_in, int* keys_out, float* val_in, float* val_out, int num);
};


