#pragma once

#include <atomic>
#include <cstdint>
#include <chrono>
#include <mutex>
#include <vector>
#include <thread>

#include <glm/glm.hpp>

extern int8_t triangle_table[256][16];

struct Vertex {
    float x, y, z;
};

namespace MarchingCubes {

    extern std::vector<std::thread> threads;

    extern std::vector<Vertex> OutputVertices;
    extern std::atomic_flag marched;
    extern std::atomic<uint8_t> finished;
    extern std::mutex vertex_mutex;

    extern int num_threads;

    extern decltype(std::chrono::high_resolution_clock::now()) last_iter_timer;

    void marching_cubes(
        float* buffer, int width, int length, int height, float threshold,
        glm::vec3& centroid, int step, int n_threads, int thread_idx
    );
    void launchThreaded(
        float* buffer, int width, int length, int height, float threshold,
        glm::vec3& centroid, int step, int n_threads
    );

    void cleanUp();
};
