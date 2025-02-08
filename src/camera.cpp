#include <cmath>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "camera.h"

Camera::Camera(int width, int height): width(width), height(height) {
    cameraPos = glm::vec3(0.0f, 0.0f, 0.0f);
    cameraTarget = glm::vec3(0.0f, 0.0f, -1.0f);
    cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    view = glm::lookAt(
        cameraPos, cameraPos + cameraTarget, cameraUp
    );

    yaw = -90.0f, pitch = 0.0f;

    // This is just a matrix to test with
    projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

    posVector = new GLfloat[16];
}

Camera::~Camera() {
    delete posVector;
}

void Camera::registerView(GLuint shaderId) {
    GLuint matrixLocation = glGetUniformLocation(shaderId, "view");
    glUniformMatrix4fv(matrixLocation, 1, GL_FALSE, glm::value_ptr(view));

    static bool first = true;
    if (!scene || first) {
        matrixLocation = glGetUniformLocation(shaderId, "projection");
        glUniformMatrix4fv(matrixLocation, 1, GL_FALSE, glm::value_ptr(projection));
        first = false;
    }
}

void Camera::getPositionFromShader(GLuint shaderId) {
    GLuint matrixLocation = glGetUniformLocation(shaderId, "view");
    glGetUniformfv(shaderId, matrixLocation, posVector);
}

void Camera::updateView() {
    if (!scene)
        return;

    view = glm::lookAt(
        cameraPos, cameraPos + cameraTarget, cameraUp
    );
}

void Camera::setCentroid(const glm::vec3& centroid) {
    sceneCentroid = centroid;
}

GLfloat* Camera::getVectorPtr() {
    return posVector;
}

void Camera::calculateDirection(GLFWwindow* window, double xpos, double ypos) {
    static float lastX = xpos, lastY = ypos;
    static float xoffset = xpos - lastX;
    static float yoffset = lastY - ypos;

    static float localYaw = yaw;
    static float localPitch = pitch;
    
    xoffset = xpos - lastX;
    yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;
    
    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;
    
    yaw += xoffset;
    pitch += yoffset;
    if(pitch > 89.0f)
        pitch = 89.0f;
    if(pitch < -89.0f)
        pitch = -89.0f;

    if (!scene && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        localYaw += xoffset;
        localPitch += yoffset;

        glm::mat4 translateToOrigin = glm::translate(glm::mat4(1.0f), sceneCentroid);
        glm::mat4 xRot = glm::rotate(translateToOrigin, glm::radians(localYaw), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 yRot = glm::rotate(xRot, glm::radians(localPitch), glm::vec3(1.0f, 0.0f, 0.0f));
        view = glm::translate(yRot, -1.0f * sceneCentroid);

        return;
    }

    direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    direction.y = -1 * (sin(glm::radians(pitch)));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraTarget = glm::normalize(direction);
}

void Camera::calculateZoom(double yoffset) {
    static float fov = 45.0f;

    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 89.0f)
        fov = 89.0f; 
    
    projection = glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 100.0f);
}

void Camera::handleInput(GLFWwindow* window) {
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

