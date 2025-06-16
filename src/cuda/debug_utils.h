#pragma once

#define CHECK_CUDA(A, debug) \
{ \
    auto ret = A; \
    if (ret != cudaSuccess && debug) { \
        std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
        throw std::runtime_error(cudaGetErrorString(ret)); \
    } \
}

#define TIME_SANDWICH_START(id) \
auto static id = std::chrono::high_resolution_clock::now();

#define TIME_SANDWICH_END(id) \
auto now = std::chrono::high_resolution_clock::now(); \
std::chrono::duration<double, std::milli> duration = now - id; \
id = now; \
std::cout << "Time since last iteration: " << duration.count() << std::endl;

