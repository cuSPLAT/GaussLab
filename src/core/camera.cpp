#include <cmath>
#include <iostream>
#include <glm/ext/matrix_clip_space.hpp>

#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "camera.h"
#include "core/renderer.h"

CameraView::CameraView(int width, int height): width(width), height(height),
    fov(45.0f), mouseData(0), model(1.0f), sceneCentroid(0.f)
{
    //TODO: changable from gui
    cameraPos = glm::vec3(0.0f, 0.0f, 0.0f);
    cameraTarget = glm::vec3(0.0f, 0.0f, -1.0f);
    cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    view = glm::lookAt(
        cameraPos, cameraPos + cameraTarget, cameraUp
    );

    projection = glm::perspective(glm::radians(45.0f), width / (float)height, 0.1f, 30.f);
}

CameraView::~CameraView() {}

void CameraView::updateViewport(float width, float height, int shader) {
    this->width = width;
    this->height = height;

    //NOTE: this is acutally slow, and it is better to have a different function
    //that uplaods on zoom and another one that uploads viewport change
    projection = glm::perspective(glm::radians(fov), width/height, 0.01f, 5.f);
    GLuint matrixLocation = glGetUniformLocation(shader, "projection");
    glUniformMatrix4fv(matrixLocation, 1, GL_FALSE, glm::value_ptr(projection));
};

void CameraView::registerModelView(GLuint shaderId) {
    GLuint matrixLocation = glGetUniformLocation(shaderId, "view");
    glUniformMatrix4fv(matrixLocation, 1, GL_FALSE, glm::value_ptr(view));
    matrixLocation = glGetUniformLocation(shaderId, "model");
    glUniformMatrix4fv(matrixLocation, 1, GL_FALSE, glm::value_ptr(model));

    //TODO: better is to do the constant location way or constant mapped memory
}

void Camera::updateView() {
    view = glm::lookAt(
        cameraPos, cameraPos + cameraTarget, cameraUp
    );
}

void Camera::lookAt(const glm::vec3& centroid) {
    sceneCentroid = centroid;
    cameraTarget = glm::normalize(sceneCentroid);
    cameraPos = sceneCentroid - 1.5f * cameraTarget;
    
    mouseData.yaw = glm::degrees(atan2(cameraTarget.z, cameraTarget.x));
    mouseData.pitch = glm::degrees(asin(cameraTarget.y));

    view = glm::lookAt(cameraPos, cameraPos + cameraTarget, cameraUp);
}

void CameraView::calculateDirection(GLFWwindow* window, double xpos, double ypos) {
    if (!(::globalState.in_view_mode))
        return;

    mouseData.xoffset = xpos - mouseData.lastX;
    mouseData.yoffset = mouseData.lastY - ypos;
    mouseData.lastX = xpos;
    mouseData.lastY = ypos;
    
    float sensitivity = 0.1f;
    mouseData.xoffset *= sensitivity;
    mouseData.yoffset *= sensitivity;

    if (!scene) {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            mouseData.objectModeYaw += mouseData.xoffset;
            mouseData.objectModePitch += mouseData.yoffset;

            model = glm::translate(glm::mat4(1.0f), glm::vec3(sceneCentroid));
            model = glm::rotate(model, glm::radians(mouseData.objectModeYaw), glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::rotate(model, glm::radians(mouseData.objectModePitch), glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::translate(model, -sceneCentroid);
        }
    } else {
        mouseData.yaw += mouseData.xoffset;
        mouseData.pitch += mouseData.yoffset;

        if(mouseData.pitch > 89.0f)
            mouseData.pitch = 89.0f;
        if(mouseData.pitch < -89.0f)
            mouseData.pitch = -89.0f;

        direction.x = cos(glm::radians(mouseData.yaw)) * cos(glm::radians(mouseData.pitch));
        direction.y = -1 * (sin(glm::radians(mouseData.pitch)));
        direction.z = sin(glm::radians(mouseData.yaw)) * cos(glm::radians(mouseData.pitch));
        cameraTarget = glm::normalize(direction);
    }
}

void CameraView::calculateZoom(double yoffset) {
    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 89.0f)
        fov = 89.0f; 

    //NOTE: Put the upload on zoom here, but for now all of it is done in
    //a single function at viewport update
}

void CameraView::uploadIntrinsics(GLuint program) {
    float htany = tan(glm::radians(fov / 2));
    float htanx = htany / height * width;
    float focal_z = height / (2 * htany);

    hfov_focal = glm::vec3(htanx, htany, focal_z);

    GLuint location = glGetUniformLocation(program, "hfov_focal");
    if (location != -1)
        glUniform3fv(location, 1, glm::value_ptr(hfov_focal));
}

void CameraView::handleInput(GLFWwindow* window) {
    static float deltaTime = 0.0f;
    static float lastFrameTime = 0.0f;

    float currentFrameTime = glfwGetTime();
    deltaTime = currentFrameTime - lastFrameTime;
    lastFrameTime = currentFrameTime;

    //TODO: make speed dynamic and adjustable from GUI
    const float cameraSpeed = deltaTime * 0.2f; 
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraTarget;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraTarget;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraTarget, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraTarget, cameraUp)) * cameraSpeed;
}

