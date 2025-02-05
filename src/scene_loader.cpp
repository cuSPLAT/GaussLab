#include <cstdint>
#include <glm/fwd.hpp>
#include <happly.h>

#include <string>

#include "scene_loader.h"

// All of implementation can be changed to a different ply backend
Scene* PLYLoader::loadPLy(const std::string &filename) {
    happly::PLYData scene(filename, true);
    happly::Element& points = scene.getElement("vertex");

    glm::vec3 centroid(0.0f, 0.0f, 0.0f);

    std::vector<float> x = points.getProperty<float>("x");
    std::vector<float> y = points.getProperty<float>("y");
    std::vector<float> z = points.getProperty<float>("z");
    
    std::vector<float> r = points.getProperty<float>("f_dc_0");
    std::vector<float> g = points.getProperty<float>("f_dc_1");
    std::vector<float> b = points.getProperty<float>("f_dc_2");
    
    // OpenGL's default camera looks toward the negative z axis so if
    // we have a positive system we reverse it
    int8_t modifier = 1;
    if (z[0] > 0)
       modifier = -1;

    std::vector<float> vertexPos;
    vertexPos.resize(x.size() * 3);
    for (size_t i = 0; i < x.size() * 3; i += 3) {
        vertexPos[i] = x[i / 3]; 
        centroid.x += vertexPos[i];

        vertexPos[i + 1] = y[i / 3]; 
        centroid.y += vertexPos[i + 1];

        //TODO: Understand the current coordinate system ? 
        // why the hell Z has values larger than 1 ?
        vertexPos[i + 2] = modifier * (z[i / 3] - 1); 
        centroid.z += vertexPos[i + 2];
    }
    centroid /= x.size();


    std::vector<float> vertexColors;
    vertexColors.resize(r.size() * 3);
    for (size_t i = 0; i < r.size() * 3; i += 3) {
        vertexColors[i] = r[i / 3]; 
        vertexColors[i + 1] = g[i / 3]; 
        vertexColors[i + 2] = b[i / 3]; 
    }

    //TODO: Find a better way and watch out for leaks
    Scene* scene_buffer = new Scene();
    scene_buffer->vertexPos = std::move(vertexPos);
    scene_buffer->vertexColor = std::move(vertexColors);
    scene_buffer->centroid = centroid;

    return scene_buffer;
}
