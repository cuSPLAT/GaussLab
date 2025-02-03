#include <iostream>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <renderer.h>
#include "camera.h"

Camera::Camera() {
    cameraPos = glm::vec3(0.0f, 0.0f, 0.0f);
    cameraTarget = glm::vec3(0.0f, 0.0f, -1.0f);
    cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    view = glm::lookAt(
        cameraPos, cameraPos + cameraTarget, cameraUp
    );

    posVector = new GLfloat[16];
}

Camera::~Camera() {
    delete posVector;
}

void Camera::registerView(GLuint shaderId) {
    GLuint matrixLocation = glGetUniformLocation(shaderId, "view");
    glUniformMatrix4fv(matrixLocation, 1, GL_FALSE, glm::value_ptr(view));
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

void Camera::handleInput(GLFWwindow* window) {
    const float cameraSpeed = 0.05f; 
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        cameraPos += cameraSpeed * cameraTarget;
        std::cout << "key pressed" << std::endl;
        std::cout << cameraPos[0] << " " << cameraPos[1] << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraTarget;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraTarget, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraTarget, cameraUp)) * cameraSpeed;
}

