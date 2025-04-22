#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>
#include <thread>

#include <glm/glm.hpp>

extern int8_t triangle_table[256][16];

class MarchingCubes {
    static std::vector<std::thread> threads;

public:
    // a static class
    MarchingCubes() = delete;

    static std::vector<float> OutputVertices;
    static std::atomic_flag marched;
    static std::atomic<uint8_t> finished;

    // I don't know if this will give the best performance but it is one of the ways
    static std::mutex OutVerticesMutex; 

    static void marching_cubes(
        float* buffer, int width, int length, int height, float threshold,
        glm::vec3& centroid, int step, int n_threads, int thread_idx
    );
    static void launchThreaded(
        float* buffer, int width, int length, int height, float threshold,
        glm::vec3& centroid, int step, int n_threads
    );

    static void cleanUp();
};
