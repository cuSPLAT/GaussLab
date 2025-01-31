#ifndef RENDERER_H
#define RENDERER_H

#define GLFW_INCLUDE_NONE

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <vector>

#include "camera.h"

struct Scene {
    std::vector<float> vertexPos;
    std::vector<float> vertexColor;
};

class Renderer {
    GLuint VBO, VAO;
    GLuint frameBuffer;
    GLuint rendererBuffer;

    Camera camera;

    unsigned int width, height;

public:
    Renderer(int width, int height);
    void generateInitialBuffers();
    void initializeRendererBuffer();
    
    // I will do a getter later
    GLuint shaderProgram;

    void constructScene(Scene* scene);

    GLuint getRenderBuffer();

    Camera* getCamera();

    void render(GLFWwindow* window);
};

#endif
