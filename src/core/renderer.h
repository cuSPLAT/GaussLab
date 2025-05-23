#ifndef RENDERER_H
#define RENDERER_H

#include <cstdint>
#include <imgui.h>
#include <memory>
#include <atomic>
#include <vector>
#define GLFW_INCLUDE_NONE

#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <cuda_gl_interop.h>

#include <core/camera.h>
#include <algorithms/marchingcubes.h>

struct Scene {
    std::unique_ptr<float[]> sceneDataBuffer;
    size_t verticesCount = 0;
    size_t bufferSize;
    glm::vec3 centroid;
};

// A global state for the renderer, this will be used to pass data between classes
// instead of always having to call a function from one class with certain parameters
// which would get messy after time
struct GlobalState {
    enum class RenderMode {
        Splats,
        PCD
    };

    // write from GUI only
    RenderMode renderingMode = RenderMode::PCD;
    bool sortingEnabled = false;
    GLuint debugMode = GL_TRIANGLES;
};
extern GlobalState globalState;

class Renderer {
    static float quadVertices[8];
    static int quadIndices[6];

    GLuint VBO, VAO;
    GLuint quadVBO, quadEBO;
    GLuint frameBuffer;
    GLuint rendererBuffer;

    // complete data of the Gaussians;
    GLuint gaussianDataBuffer;

    GLuint depthBuffer_gl, depthIndices_gl;
    GLuint sorted_depthBuffer_gl, sorted_depthIndices_gl;
    bool newScene;

    cudaGraphicsResource_t depth_buffer, index_buffer;
    cudaGraphicsResource_t sorted_depth_buffer, sorted_index_buffer;
    cudaGraphicsResource_t cu_buffers[4];

    Camera camera;
    unsigned int width, height;


// for public variables, to make the code cleaner
public:
     unsigned int verticesCount = 0;

public:
    Renderer(int width, int height);
    ~Renderer();

    void generateInitialBuffers();
    void initializeRendererBuffer();
    // I will do a getter later
    GLuint shaderProgram, veryRealComputeProgram;
    GLuint gaussRenProgram;

    void constructScene(Scene* scene, std::vector<Vertex>& vertices);
    GLuint getRenderBuffer();

    Camera* getCamera();

    void render(GLFWwindow* window);
};

#endif
