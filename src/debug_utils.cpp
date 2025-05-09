#include "debug_utils.h"

#include <fstream>

namespace DebugUtils {

void exportObj(const std::string& file_name, std::vector<Vertex> &vertices) {
    std::ofstream obj_file;
    obj_file.open(file_name, std::ios_base::out);
    for (size_t i = 0; i < vertices.size(); i += 2)
        obj_file << "v " << vertices[i].x << " " << vertices[i].z << " " << vertices[i].y << '\n';

    for (size_t i = 0; i < vertices.size(); i += 2) {
        if (i % 6 == 0) obj_file << "\nf ";
        obj_file << i / 2 + 1 << " ";
    }
    obj_file.close();
}

};
