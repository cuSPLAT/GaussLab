#include <cstdio>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>

#include "render_utils.h"
#include "debug_utils.h"


void* RenderUtils::temp_storage = nullptr;
size_t RenderUtils::temp_storage_size = 0;

void RenderUtils::sort_gaussians_gpu(
    float* keys_in, float* keys_out, int* val_in, int* val_out, int num, bool& newScene
) {
    using _this = RenderUtils;

    // The first time we run this we have to know the size of the temp
    // storage needed for sorting
    if (newScene) {
        if (_this::temp_storage != nullptr) {
            cudaFree(_this::temp_storage);
            _this::temp_storage = nullptr;
        }
        cub::DeviceRadixSort::SortPairs(
            nullptr, _this::temp_storage_size, keys_in, keys_out, val_in, val_out, num
        );
        cudaMalloc(&_this::temp_storage, _this::temp_storage_size);
        newScene = false;

        std::cout << "New scene, needed storage of: " << _this::temp_storage_size << " bytes"
            << std::endl;
    }

    //int i = 4; printf("element number %d is equal to keys out%d\n",i,(int)*(dev_ptr_key+i));
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        _this::temp_storage, _this::temp_storage_size, keys_in, keys_out, val_in, val_out, num
    ), true)

    if (false) {
        static thrust::device_ptr<int> dev_ptr_sortedindex = thrust::device_pointer_cast(val_out);
        std::cout << "Lowest index: " << *dev_ptr_sortedindex << std::endl;
    }
}

void RenderUtils::cleanUp() {
    using _this = RenderUtils;

    cudaFree(_this::temp_storage);
    _this::temp_storage = nullptr;
}
