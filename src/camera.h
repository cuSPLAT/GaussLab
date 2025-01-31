#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Camera {
    glm::mat4 view;
    glm::vec3 cameraPos;
    glm::vec3 cameraUp;
    glm::vec3 cameraTarget;

    GLfloat* posVector;

public:
    Camera();
    ~Camera();
    void registerView(GLuint shaderId);
    void handleInput(GLFWwindow* window);

    void getPositionFromShader(GLuint shaderId);

    GLfloat* getVectorPtr();
    
    void updateView();
};

#endif
