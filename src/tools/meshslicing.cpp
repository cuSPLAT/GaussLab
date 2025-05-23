#include "meshslicing.h"

#include <core/renderer.h>
#include <core/camera.h>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_projection.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/matrix.hpp>
#include <iostream>

Camera* camera = nullptr;

std::vector<glm::vec3> world_mousePositions = {};

namespace Tools {

namespace MeshSlicing {

void captureMousePos(GLFWwindow* window) {
    double xpos, ypos;

    if (!camera) {
        Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
        camera = renderer->getCamera();
    }

    // 0, 0 as the bottom-left position is not always correct
    glm::vec4 viewport = {0.f, 0.f, camera->width, camera->height};

    glfwGetCursorPos(window, &xpos, &ypos);
    glm::vec3 worldSpaceMouse = glm::unProject(
        glm::vec3(xpos, ypos, 0.f), camera->view,
        camera->projection, viewport
    );
    world_mousePositions.push_back(worldSpaceMouse);

    std::cout << worldSpaceMouse[0] << " " << worldSpaceMouse[1] << " " << worldSpaceMouse[1] << std::endl;
    if (world_mousePositions.size() == 2)
        createPlane();
}

void createPlane() {
    const glm::vec3 negative_z = {0.f, 0.f, -1.f};
    glm::vec3 world_space_z = glm::inverse(glm::mat3(camera->view)) * negative_z;
    world_space_z = glm::normalize(world_space_z);

    // We do a cross product between the plane tangent and the into screen vector
    // to get the normal
    const glm::vec3 planeTangent = glm::normalize(world_mousePositions[1] - world_mousePositions[0]);
    glm::vec3 planeNormal = glm::cross(planeTangent, world_space_z);

    // solve the plane equation
    float d = glm::dot(world_mousePositions[0], planeNormal);
    
    world_mousePositions.clear();
}

};
};
