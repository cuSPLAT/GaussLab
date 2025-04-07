#ifndef RENDERER_H
#define RENDERER_H

#include <imgui.h>
#include <memory>
#define GLFW_INCLUDE_NONE

#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <RadixSort.hpp>
#include <cuda_gl_interop.h>

#include "camera.h"

struct Scene {
    std::unique_ptr<float[]> sceneDataBuffer;
    size_t verticesCount;
    size_t bufferSize;
    glm::vec3 centroid;
};

class Renderer {
    static float quadVertices[8];
    static int quadIndices[6];

    enum class Mode {
        Splats,
        PCD
    };

    GLuint VBO, VAO;
    GLuint quadVBO, quadEBO;
    GLuint frameBuffer;
    GLuint rendererBuffer;

    // complete data of the Gaussians;
    GLuint gaussianDataBuffer;

    GLuint depthBuffer_gl, depthIndices_gl;
    GLuint sorted_depthBuffer_gl, sorted_depthIndices_gl;

    Mode renderingMode;

    cudaGraphicsResource_t depth_buffer, index_buffer;
    cudaGraphicsResource_t sorted_depth_buffer, sorted_index_buffer;
    cudaGraphicsResource_t cu_buffers[4];

    Camera camera;
    unsigned int width, height;


// for public variables, to make the code cleaner
public:
     unsigned int verticesCount;

public:
    Renderer(int width, int height);
    ~Renderer();

    void generateInitialBuffers();
    void initializeRendererBuffer();
    // I will do a getter later
    GLuint shaderProgram, veryRealComputeProgram;
    GLuint gaussRenProgram;

    void constructScene(Scene* scene);
    GLuint getRenderBuffer();

    Camera* getCamera();

    void render(GLFWwindow* window);
};

#endif
