#include "meshslicing.h"

#include <iostream>

#include <core/renderer.h>
#include <core/camera.h>

#include <interface/viewport.h>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_projection.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/matrix.hpp>

#define EPSILON 0.004f

std::vector<glm::vec3> world_mousePositions = {};

namespace Tools {

namespace MeshSlicing {

void captureMousePos(GLFWwindow* window, int action) {
    double xpos, ypos;
    const Viewport& this_viewport = Viewport::viewports[::globalState.selectedViewport];

    glfwGetCursorPos(window, &xpos, &ypos);
    glm::vec3 worldSpaceMouse_tangent = glm::unProject(
        glm::vec3(xpos, ypos, 0.f),
        this_viewport.view_camera->view,
        this_viewport.view_camera->projection,
        this_viewport.viewportPosData
    );
    world_mousePositions.push_back(worldSpaceMouse_tangent);
    if (world_mousePositions.size() == 2) {
        std::cout << glm::distance(world_mousePositions[0], world_mousePositions[1]) << std::endl;
        if (glm::distance(world_mousePositions[0], world_mousePositions[1]) < EPSILON) {
            world_mousePositions.clear();
            return;
        }

        const glm::vec3 ray_near = glm::unProject(
            glm::vec3(xpos, ypos, 0.f), 
            this_viewport.view_camera->view,
            this_viewport.view_camera->projection,
            this_viewport.viewportPosData
        );
        const glm::vec3 ray_far = glm::unProject(
            glm::vec3(xpos, ypos, 1.f),
            this_viewport.view_camera->view,
            this_viewport.view_camera->projection,
            this_viewport.viewportPosData
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
    const glm::vec3 planeNormal = glm::normalize(glm::cross(ray, planeTangent));

    // solve the plane equation
    float d = -glm::dot(world_mousePositions[1], planeNormal);

    const GLuint optional = glGetUniformLocation(::globalState.vertexProgram, "planeExists");
    const GLuint normal = glGetUniformLocation(::globalState.vertexProgram, "planeData");
    glUniform1i(optional, true);
    glUniform4fv(normal, 1, glm::value_ptr(glm::vec4 {planeNormal, d})); 

    world_mousePositions.clear();
}

};
};
