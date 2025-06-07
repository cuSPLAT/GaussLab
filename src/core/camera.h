#pragma once

#define GLFW_INCLUDE_NONE

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Camera {
    glm::vec3 direction;
    glm::vec3 sceneCentroid;

    glm::vec3 hfov_focal;

    float fov;
    float yaw, pitch;
    GLfloat* posVector;

public:
    glm::vec3 cameraPos, cameraUp, cameraTarget;
    // Probably it is better to make them public in the 
    // renderer, but leave it for later
    int width, height;
    bool scene = true;
    glm::mat4 view, projection;

public:
    Camera(int width, int height);
    ~Camera();
    void registerView(GLuint shaderId);
    void handleInput(GLFWwindow* window);
    void uploadIntrinsics(GLuint program);

    void updateViewport(float width, float height, int shader);

    void getPositionFromShader(GLuint shaderId);
    void calculateDirection(GLFWwindow* window, double xpos, double ypos);
    void calculateZoom(double yoffset);

    GLfloat* getVectorPtr();
    
    void setCentroid(const glm::vec3& centroid);
    void updateView();
};

