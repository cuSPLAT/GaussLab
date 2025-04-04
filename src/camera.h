#ifndef CAMERA_H
#define CAMERA_H

#define GLFW_INCLUDE_NONE

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Camera {
    glm::mat4 view, projection;
    glm::vec3 cameraPos, cameraUp, cameraTarget;
    glm::vec3 direction;
    glm::vec3 sceneCentroid;

    glm::vec3 hfov_focal;

    float fov;
    float yaw, pitch;
    int width, height;
    GLfloat* posVector;

public:
    bool scene = true;

public:
    Camera(int width, int height);
    ~Camera();
    void registerView(GLuint shaderId);
    void handleInput(GLFWwindow* window);
    void uploadIntrinsics(GLuint program);

    void getPositionFromShader(GLuint shaderId);
    void calculateDirection(GLFWwindow* window, double xpos, double ypos);
    void calculateZoom(double yoffset);

    GLfloat* getVectorPtr();
    
    void setCentroid(const glm::vec3& centroid);
    void updateView();
};

#endif
