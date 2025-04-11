#pragma once

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

struct DensityField {
    float* buffer;
    int width, length, height;
    int volume;
};

void marching_cubes(DensityField& field, float threshold, std::vector<float>& vertices, glm::vec3& centroid);

extern int8_t triangle_table[256][16];
