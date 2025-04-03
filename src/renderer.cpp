#include "renderer.h"
#include "shaders.h"
#include "camera.h"
#include "cuda/render_utils.h"
#include "cuda/debug_utils.h"

#include <cstddef>
#include <iostream>
#include <chrono>

#include <RadixSort.hpp>

#include <cuda_gl_interop.h>

Renderer::Renderer(int width, int height):
    width(width), height(height), camera(width, height),
    cu_buffers(depth_buffer, index_buffer, sorted_depth_buffer, sorted_index_buffer) {}

void Renderer::initializeRendererBuffer() {
    glGenFramebuffers(1, &frameBuffer);

    // the texture that will act as a color buffer for rendering
    glGenTextures(1, &rendererBuffer);
    glBindTexture(GL_TEXTURE_2D, rendererBuffer);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
        GL_RGB, GL_UNSIGNED_BYTE, nullptr
    );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // attach the texture as a color buffer, the texture here acts just as
    // data buffer
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rendererBuffer, 0
    );
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
}

GLuint Renderer::getRenderBuffer() {
    return rendererBuffer;
}

void Renderer::generateInitialBuffers() {
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &colorBuffer);

    // For Cuda-GL Interop
    glGenBuffers(1, &depthBuffer_gl);
    glGenBuffers(1, &depthIndices_gl);
    glGenBuffers(1, &sorted_depthBuffer_gl);
    glGenBuffers(1, &sorted_depthIndices_gl);
    
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &Shaders::vertexShader, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &Shaders::fragmentShader, nullptr);
    glCompileShader(fragmentShader);

    GLuint veryRealComputeShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(veryRealComputeShader, 1, &Shaders::viewMatMulCompute, nullptr);
    glCompileShader(veryRealComputeShader);
    
    //TODO: move this to a macro or inline function
    //int  success;
    //char infoLog[512];
    //glGetShaderiv(veryRealComputeShader, GL_COMPILE_STATUS, &success);
    //if(!success)
    //{
    //    glGetShaderInfoLog(veryRealComputeShader, 512, NULL, infoLog);
    //    std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    //
    //}    

    // Create a shader storage buffer that will store the vertices after multiplying
    // by the view matrix. Later it will be better to use just one and modify it in place
    //NOTE: probably doing them in one step is much better, I am just testing

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    veryRealComputeProgram = glCreateProgram();
    glAttachShader(veryRealComputeProgram, veryRealComputeShader);
    glLinkProgram(veryRealComputeProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteShader(veryRealComputeShader);

}

void Renderer::constructScene(Scene* scene) {
    size_t verticesPosCount = scene->vertexPos.size();
    size_t color_count = scene->vertexColor.size();
    verticesCount = verticesPosCount / 3;
    camera.setCentroid(scene->centroid);

    camera.registerView(shaderProgram);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, verticesPosCount * sizeof(float), scene->vertexPos.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, color_count * sizeof(float), scene->vertexColor.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

    std::cout << "Current vertices count " << verticesCount << std::endl;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, depthBuffer_gl);
    glBufferData(GL_SHADER_STORAGE_BUFFER, verticesCount * sizeof(float), NULL, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, depthBuffer_gl);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, depthIndices_gl);
    glBufferData(GL_SHADER_STORAGE_BUFFER, verticesCount * sizeof(int), NULL, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, depthIndices_gl);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sorted_depthBuffer_gl);
    glBufferData(GL_SHADER_STORAGE_BUFFER, verticesCount * sizeof(float), NULL, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, sorted_depthBuffer_gl);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sorted_depthIndices_gl);
    glBufferData(GL_SHADER_STORAGE_BUFFER, verticesCount * sizeof(int), NULL, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, sorted_depthIndices_gl);

    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&depth_buffer, depthBuffer_gl, cudaGraphicsRegisterFlagsNone), true)
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&index_buffer, depthIndices_gl, cudaGraphicsRegisterFlagsNone), true)
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&sorted_depth_buffer, sorted_depthBuffer_gl, cudaGraphicsRegisterFlagsNone), true)
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&sorted_index_buffer, sorted_depthIndices_gl, cudaGraphicsRegisterFlagsNone),true)

}

Camera* Renderer::getCamera() {
    return &camera;
}

void Renderer::render(GLFWwindow* window) {
    camera.registerView(shaderProgram);
    camera.registerView(veryRealComputeProgram);
    camera.handleInput(window);

    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

    // We aren't drawing anything, just computing
    glEnable(GL_RASTERIZER_DISCARD);

    glUseProgram(veryRealComputeProgram);
    glDrawArrays(GL_POINTS, 0, verticesCount);

    glDisable(GL_RASTERIZER_DISCARD);

    TIME_SANDWICH_START(CUDA_INTEROP)

    CHECK_CUDA(cudaGraphicsMapResources(4, &depth_buffer), true)
    void *d_depth_ptr, *d_index_ptr, *d_sortedDepth_ptr, *d_sortedIndex_ptr;
    size_t depth_buffer_size, index_buffer_size, sorted_depth_size, sorted_index_size;
    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(&d_depth_ptr, &depth_buffer_size, depth_buffer), true)
    cudaGraphicsResourceGetMappedPointer(&d_index_ptr, &index_buffer_size, index_buffer);
    cudaGraphicsResourceGetMappedPointer(&d_sortedDepth_ptr, &sorted_depth_size, sorted_depth_buffer);
    cudaGraphicsResourceGetMappedPointer(&d_sortedIndex_ptr, &sorted_index_size, sorted_index_buffer);

    RenderUtils::sort_gaussians_gpu(
        (int*)d_index_ptr, (int*)d_sortedIndex_ptr,
        (float*)d_depth_ptr, (float*)d_sortedDepth_ptr, verticesCount
    );

    CHECK_CUDA(cudaGraphicsUnmapResources(4, &depth_buffer), true)
    TIME_SANDWICH_END(CUDA_INTEROP)

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    // drawing into the small window happens here
    // we only have one VAO and one shader that are always binded cudaGraphicsMapResources
    // no need to rebind them on every draw call
    glUseProgram(shaderProgram);
    glDrawArrays(GL_POINTS, 0, verticesCount);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    camera.updateView();
}

Renderer::~Renderer() {
    cudaGraphicsUnregisterResource(depth_buffer);
    cudaGraphicsUnregisterResource(index_buffer);
    cudaGraphicsUnregisterResource(sorted_depth_buffer);
    cudaGraphicsUnregisterResource(sorted_index_buffer);
}
