#pragma once

#define LOG(x) std::cout << "[LOG] " << x << std::endl;

#include <vector>

#include <algorithms/marchingcubes.h>

namespace DebugUtils {
    void exportObj(const std::string& file_name, std::vector<Vertex>& vertices);
};
