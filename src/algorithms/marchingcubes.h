#pragma once

#include <atomic>
#include <cstdint>
#include <chrono>
#include <mutex>
#include <vector>
#include <thread>

#include <glm/glm.hpp>

struct Vertex {
    float x, y, z;
};

struct MarchingCubesEngine {
    static int8_t triangle_table[256][16];

    std::vector<std::thread> threads;
    int num_threads = 0;

    std::atomic_flag marched;
    std::atomic<uint8_t> finished {0};
    std::mutex vertex_mutex;

    decltype(std::chrono::high_resolution_clock::now()) last_iter_timer;

    std::vector<Vertex> OutputVertices;

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
