#include "meshslicing.h"

#include <core/renderer.h>
#include <core/camera.h>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_projection.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/matrix.hpp>
#include <iostream>

Camera* camera = nullptr;
Renderer* renderer = nullptr;

std::vector<glm::vec3> world_mousePositions = {};

namespace Tools {

namespace MeshSlicing {

void captureMousePos(GLFWwindow* window, int action) {
    double xpos, ypos;

    if (!camera) {
        renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
        camera = renderer->getCamera();
    }

    // 0, 0 as the bottom-left position is not always correct
    glm::vec4 viewport = {0.f, 0.f, camera->width, camera->height};

    glfwGetCursorPos(window, &xpos, &ypos);
    glm::vec3 worldSpaceMouse_tangent = glm::unProject(
        glm::vec3(xpos, ypos, 0.f), camera->view,
        camera->projection, viewport
    );
    world_mousePositions.push_back(worldSpaceMouse_tangent);
    if (world_mousePositions.size() == 2) {

        const glm::vec3 ray_near = glm::unProject(
            glm::vec3(xpos, ypos, 0.f), camera->view,
            camera->projection, viewport
        );
        const glm::vec3 ray_far = glm::unProject(
            glm::vec3(xpos, ypos, 1.f), camera->view,
            camera->projection, viewport
        );
        const glm::vec3 ray = ray_near - ray_far;

        // check if the differene is small ?
        createPlane(ray);
    }
}

void createPlane(const glm::vec3& ray) {
    // We do a cross product between the plane tangent and the into screen vector
    // to get the normal
    const glm::vec3 planeTangent = glm::normalize(world_mousePositions[1] - world_mousePositions[0]);
    glm::vec4 viewport = {0.f, 0.f, camera->width, camera->height};
    const glm::vec3 planeNormal = glm::normalize(glm::cross(ray, planeTangent));

    // solve the plane equation
    float d = -glm::dot(world_mousePositions[1], planeNormal);

    const GLuint optional = glGetUniformLocation(renderer->shaderProgram, "planeExists");
    const GLuint normal = glGetUniformLocation(renderer->shaderProgram, "planeData");
    glUniform1i(optional, true);
    glUniform4fv(normal, 1, glm::value_ptr(glm::vec4 {planeNormal, d})); 

    world_mousePositions.clear();
}

};
};
