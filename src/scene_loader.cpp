#include <happly.h>

#include <string>

#include "scene_loader.h"

// All of implementation can be changed to a different ply backend
Scene* PLYLoader::loadPLy(const std::string &filename) {
    happly::PLYData scene(filename);
    happly::Element& points = scene.getElement("vertex");

    std::vector<float> x = points.getProperty<float>("x");
    std::vector<float> y = points.getProperty<float>("y");
    std::vector<float> z = points.getProperty<float>("z");
    
    std::vector<float> r = points.getProperty<float>("f_dc_0");
    std::vector<float> g = points.getProperty<float>("f_dc_1");
    std::vector<float> b = points.getProperty<float>("f_dc_2");

    std::vector<float> vertexPos;
    vertexPos.resize(x.size() * 3);
    for (size_t i = 0; i < x.size() * 3; i += 3) {
        vertexPos[i] = x[i / 3]; 
        vertexPos[i + 1] = y[i / 3]; 

        //TODO: Understand the current coordinate system ? 
        // why the hell Z has values larger than 1 ?
        vertexPos[i + 2] = -1 * (z[i / 3] - 1); 
    }


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

    return scene_buffer;
}
