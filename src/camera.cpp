#include <cmath>
#define GLFW_INCLUDE_NONE

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GLFW/glfw3.h>

#include "renderer.h"
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

    // This does not have to be called every frame btw
    matrixLocation = glGetUniformLocation(shaderId, "projection");
    glUniformMatrix4fv(matrixLocation, 1, GL_FALSE, glm::value_ptr(projection));
}

void Camera::getPositionFromShader(GLuint shaderId) {
    GLuint matrixLocation = glGetUniformLocation(shaderId, "view");
    glGetUniformfv(shaderId, matrixLocation, posVector);
}

void Camera::updateView() {
    view = glm::lookAt(
        cameraPos, cameraPos + cameraTarget, cameraUp
    );
}

GLfloat* Camera::getVectorPtr() {
    return posVector;
}

void Camera::calculateDirection(double xpos, double ypos) {
    static float lastX = xpos, lastY = ypos;
    static float xoffset = xpos - lastX;
    static float yoffset = lastY - ypos;
    
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

    direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    direction.y = sin(glm::radians(pitch));
    direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraTarget = glm::normalize(direction);
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

