#pragma once

#define GLFW_INCLUDE_NONE

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class CameraView {
    glm::vec3 direction;
    glm::vec3 hfov_focal;

    float fov;
    struct MouseData {
        float yaw, pitch;
        float lastY, lastX;
        float xoffset, yoffset;
        float objectModeYaw, objectModePitch;
    };

    MouseData mouseData;

public:
    glm::vec3 cameraPos, cameraUp, cameraTarget;
    glm::vec3 sceneCentroid;
    // Probably it is better to make them public in the 
    // renderer, but leave it for later
    int width, height;
    bool scene = false;
    glm::mat4 model, view, projection;

public:
    CameraView(int width, int height);
    ~CameraView();
    void registerModelView(GLuint shaderId);
    void handleInput(GLFWwindow* window);
    void uploadIntrinsics(GLuint program);

    void updateViewport(float width, float height, int shader);

    void calculateDirection(GLFWwindow* window, double xpos, double ypos);
    void calculateZoom(double yoffset);

    void setCentroid(const glm::vec3& centroid);
    void updateView();
};

