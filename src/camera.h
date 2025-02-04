#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Camera {
    glm::mat4 view, projection;
    glm::vec3 cameraPos;
    glm::vec3 cameraUp;
    glm::vec3 cameraTarget;
    glm::vec3 direction;

    float yaw, pitch;
    int width, height;
    GLfloat* posVector;

public:
    Camera(int width, int height);
    ~Camera();
    void registerView(GLuint shaderId);
    void handleInput(GLFWwindow* window);

    void getPositionFromShader(GLuint shaderId);
    void calculateDirection(double xpos, double ypos);

    GLfloat* getVectorPtr();
    
    void updateView();
};

#endif
