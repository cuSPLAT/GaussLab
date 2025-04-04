#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/fwd.hpp>
#include <happly.h>

#include <memory>
#include <string>
#include <vector>

#include "scene_loader.h"

// All of implementation can be changed to a different ply backend
Scene* PLYLoader::loadPLy(const std::string &filename) {
    happly::PLYData scene(filename);
    happly::Element& points = scene.getElement("vertex");

    glm::vec3 centroid(0.0f, 0.0f, 0.0f);
    std::vector<float> x = points.getProperty<float>("x");
    std::vector<float> y = points.getProperty<float>("y");
    std::vector<float> z = points.getProperty<float>("z");
    
    std::vector<float> r = points.getProperty<float>("f_dc_0");
    std::vector<float> g = points.getProperty<float>("f_dc_1");
    std::vector<float> b = points.getProperty<float>("f_dc_2");

    // applying sigmoid to the opacities to transform them between 0 and 1
    // Kerbl et al (5.1)
    std::vector<float> opacity = points.getProperty<float>("opacity");
    std::transform(opacity.begin(), opacity.end(), opacity.begin(),
       [](float opacity) {
            return 1/(1 + exp(-opacity));
        }
    );

    std::vector<float> scale_0 = points.getProperty<float>("scale_0");
    std::vector<float> scale_1 = points.getProperty<float>("scale_1");
    std::vector<float> scale_2 = points.getProperty<float>("scale_2");
    
    std::vector<float> rot_0 = points.getProperty<float>("rot_0");
    std::vector<float> rot_1 = points.getProperty<float>("rot_1");
    std::vector<float> rot_2 = points.getProperty<float>("rot_2");
    std::vector<float> rot_3 = points.getProperty<float>("rot_3");

    size_t bufferSize = x.size() * 3 + r.size() * 3 + opacity.size() + scale_0.size() * 3 + rot_0.size() * 4;
    std::unique_ptr<float[]> flatDataBuffer = std::make_unique<float[]>(bufferSize);
    
    // OpenGL's default camera looks toward the negative z axis so if
    // we have a positive system we reverse it
    int8_t modifier = 1;
    if (z[0] > 0)
       modifier = -1;

    for (size_t i = 0; i < bufferSize; i += 14) {
        flatDataBuffer[i] = x[i / 14]; 
        centroid.x += flatDataBuffer[i];
        flatDataBuffer[i + 1] = y[i / 14]; 
        centroid.y += flatDataBuffer[i + 1];
        //TODO: Understand the current coordinate system ? 
        // why the hell Z has values larger than 1 ?
        flatDataBuffer[i + 2] = modifier * (z[i / 14] - 1); 
        centroid.z += flatDataBuffer[i + 2];

        flatDataBuffer[i + 3] = r[i / 14]; 
        flatDataBuffer[i + 4] = g[i / 14]; 
        flatDataBuffer[i + 5] = b[i / 14]; 
        flatDataBuffer[i + 6] = opacity[i /14];

        // According to the paper, scales are passed to an exponential function
        flatDataBuffer[i + 7] = exp(scale_0[i / 14]); 
        flatDataBuffer[i + 8] = exp(scale_1[i / 14]); 
        flatDataBuffer[i + 9] = exp(scale_2[i / 14]); 

        flatDataBuffer[i + 10] = rot_0[i / 14]; 
        flatDataBuffer[i + 11] = rot_1[i / 14]; 
        flatDataBuffer[i + 12] = rot_2[i / 14]; 
        flatDataBuffer[i + 13] = rot_2[i / 14]; 
    }
    centroid /= x.size();

    //TODO: Find a better way and watch out for leaks
    Scene* scene_buffer = new Scene();
    scene_buffer->sceneDataBuffer = std::move(flatDataBuffer);
    scene_buffer->verticesCount = x.size();
    scene_buffer->bufferSize = bufferSize * sizeof(float);
    scene_buffer->centroid = centroid;

    std::cout << "Scene loaded" << std::endl;

    return scene_buffer;
}
