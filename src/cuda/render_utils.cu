#include <cub/cub.cuh>

#include "render_utils.h"

void* RenderUtils::temp_storage = nullptr;
size_t RenderUtils::temp_storage_size = 0;

void RenderUtils::sort_gaussians_gpu(int* keys_in, int* keys_out, float* val_in, float* val_out, int num) {
    using _this = RenderUtils;

    // THe first time we run this we have to know the size of the temp
    // storage needed for sorting
    if (_this::temp_storage == nullptr) {
        cub::DeviceRadixSort::SortPairs(
            nullptr, _this::temp_storage_size, keys_in, keys_out, val_in, val_out, num
        );
        cudaMalloc(&_this::temp_storage, _this::temp_storage_size);
    }

    cub::DeviceRadixSort::SortPairs(
        _this::temp_storage, _this::temp_storage_size, keys_in, keys_out, val_in, val_out, num
    );
}
