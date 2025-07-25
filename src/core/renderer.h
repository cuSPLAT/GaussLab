#pragma once

#include <cstdint>
#include <imgui.h>
#include <memory>
#include <atomic>
#include <torch/serialize/input-archive.h>
#include <vector>
#define GLFW_INCLUDE_NONE

#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <cuda_gl_interop.h>

#include <core/camera.h>
#include <algorithms/marchingcubes.h>

struct Scene {
    std::unique_ptr<float[]> sceneDataBuffer;
    bool interleavedBuffer = true;
    size_t verticesCount = 0;
    size_t bufferSize;
    glm::vec3 centroid;
};

struct GPUScene {
    torch::Tensor means;
    torch::Tensor colors;
    torch::Tensor opacities;
    torch::Tensor scales;
    torch::Tensor quats;

    size_t bufferSize;
    glm::vec3 centroid;
};

class Renderer {
    static float quadVertices[8];
    static int quadIndices[6];

    GLuint VBO, VAO;
    GLuint quadVBO, quadEBO;

    // complete data of the Gaussians;
    GLuint gaussianDataBuffer;

    GLuint depthBuffer_gl, depthIndices_gl;
    GLuint sorted_depthBuffer_gl, sorted_depthIndices_gl;
    bool newScene;

    cudaGraphicsResource_t depth_buffer, index_buffer;
    cudaGraphicsResource_t sorted_depth_buffer, sorted_index_buffer;
    cudaGraphicsResource_t cu_buffers[4];
    bool gaussianSceneCreated = false;

    void allocateGaussianQuad();
    void allocateSortingBuffers();
    void processGaussianSplats(int i);

// for public variables, to make the code cleaner
public:
    unsigned int verticesCount = 0;
    unsigned int gaussiansCount = 0;

public:
    Renderer() = default;
    ~Renderer();

    bool initOpenGL(GLFWwindow* window);
    void generateInitialBuffers();
    // I will do a getter later
    GLuint shaderProgram, veryRealComputeProgram;
    GLuint gaussRenProgram;

    void constructMeshScene(std::vector<Vertex>& vertices);
    void constructSplatScene(Scene* scene);
    void constructSplatSceneFromGPU(GPUScene& scene);

    void render(GLFWwindow* window);
};

