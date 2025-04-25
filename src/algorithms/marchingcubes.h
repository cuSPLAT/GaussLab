#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <thread>

#include <glm/glm.hpp>

extern int8_t triangle_table[256][16];

struct Vertex {
    float x, y, z;
};

class MarchingCubes {

public:
    static std::vector<std::thread> threads;
    static std::unordered_map<int, std::vector<Vertex>> TemporaryBuffers;
    // a static class
    MarchingCubes() = delete;

    static std::vector<Vertex> OutputVertices;
    static std::atomic_flag marched;
    static std::atomic<uint8_t> finished;

    static int num_threads;

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
