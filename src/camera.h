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
    glm::vec3 sceneCentroid;


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

    void getPositionFromShader(GLuint shaderId);
    void calculateDirection(double xpos, double ypos);
    void calculateZoom(double yoffset);

    GLfloat* getVectorPtr();
    
    void setCentroid(const glm::vec3& centroid);
    void updateView();
};

#endif
