#include "algorithms/marchingcubes.h"
#include "core/camera.h"
#include <core/renderer.h>
#include <core/shaders.h>

#include <cuda/render_utils.h>
#include <cuda/debug_utils.h>

#include <driver_types.h>
#include <interface/viewport.h>

#include <cstddef>
#include <chrono>
#include <iostream>

#include <cuda_gl_interop.h>
#include <vector>

float Renderer::quadVertices[8] = {
    -1.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, -1.0f,
    -1.0f, -1.0f
};

int Renderer::quadIndices[6] = {
    0, 1, 2,
    0, 2, 3
};

GlobalState globalState;

Renderer::Renderer(int width, int height):
    width(width), height(height) {}

void Renderer::allocateGaussianQuad() {
    // the data of the rectangle that a Gaussian will occupy
    glGenBuffers(1, &quadVBO);
    glGenBuffers(1, &quadEBO);

    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);
}

void Renderer::generateInitialBuffers() {
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    // For Cuda-GL Interop
    glGenBuffers(1, &depthBuffer_gl);
    glGenBuffers(1, &depthIndices_gl);
    glGenBuffers(1, &sorted_depthBuffer_gl);
    glGenBuffers(1, &sorted_depthIndices_gl);

    glGenBuffers(1, &gaussianDataBuffer);
    allocateGaussianQuad();
    
    GLuint PCDVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(PCDVertexShader, 1, &Shaders::vertexShader, nullptr);
    glCompileShader(PCDVertexShader);

    GLuint PCDFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(PCDFragmentShader, 1, &Shaders::fragmentShader, nullptr);
    glCompileShader(PCDFragmentShader);

    GLuint veryRealComputeShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(veryRealComputeShader, 1, &Shaders::viewMatMulCompute, nullptr);
    glCompileShader(veryRealComputeShader);

    GLuint GaussianVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(GaussianVertexShader, 1, &Shaders::gaussianVertexShader, nullptr);
    glCompileShader(GaussianVertexShader);

    GLuint GaussianFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(GaussianFragmentShader, 1, &Shaders::gaussianFragmentShader, nullptr);
    glCompileShader(GaussianFragmentShader);
    
    //TODO: move this to a macro or inline function
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, PCDVertexShader);
    glAttachShader(shaderProgram, PCDFragmentShader);
    glLinkProgram(shaderProgram);

    veryRealComputeProgram = glCreateProgram();
    glAttachShader(veryRealComputeProgram, veryRealComputeShader);
    glLinkProgram(veryRealComputeProgram);

    gaussRenProgram = glCreateProgram();
    glAttachShader(gaussRenProgram, GaussianVertexShader);
    glAttachShader(gaussRenProgram, GaussianFragmentShader);
    glLinkProgram(gaussRenProgram);

    ::globalState.vertexProgram = shaderProgram;
    ::globalState.gaussianProgram = gaussRenProgram;

    glDeleteShader(PCDVertexShader);
    glDeleteShader(PCDFragmentShader);
    glDeleteShader(veryRealComputeShader);
    glDeleteShader(GaussianVertexShader);

}

void Renderer::allocateSortingBuffers() {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, depthBuffer_gl);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gaussiansCount * sizeof(float), NULL, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, depthBuffer_gl);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, depthIndices_gl);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gaussiansCount * sizeof(int), NULL, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, depthIndices_gl);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sorted_depthBuffer_gl);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gaussiansCount * sizeof(float), NULL, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, sorted_depthBuffer_gl);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sorted_depthIndices_gl);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gaussiansCount * sizeof(int), NULL, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, sorted_depthIndices_gl);

    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&depth_buffer, depthBuffer_gl, cudaGraphicsRegisterFlagsNone), true)
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&index_buffer, depthIndices_gl, cudaGraphicsRegisterFlagsNone), true)
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&sorted_depth_buffer, sorted_depthBuffer_gl, cudaGraphicsRegisterFlagsNone), true)
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&sorted_index_buffer, sorted_depthIndices_gl, cudaGraphicsRegisterFlagsNone),true)

    //*Ugly ahh code*
    cu_buffers[0] = depth_buffer;
    cu_buffers[1] = index_buffer;
    cu_buffers[2] = sorted_depth_buffer;
    cu_buffers[3] = sorted_index_buffer;
}

void Renderer::constructMeshScene(std::vector<Vertex>& vertices) {
    verticesCount = vertices.size() / 2;

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Temporary normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    const GLuint optional = glGetUniformLocation(shaderProgram, "planeExists");
    glUniform1i(optional, false);

    std::cout << "Current vertices count " << verticesCount << std::endl;
    std::cout << "Total buffer size in bytes " << vertices.size() * sizeof(Vertex) << std::endl;
}

void Renderer::constructSplatScene(Scene* scene) {
    for (int i = 0; i < Viewport::n_viewports; i++) {
        if (!Viewport::viewports[i].mesh)
            Viewport::viewports[i].view_camera->lookAt(scene->centroid);
    }
    gaussiansCount = scene->verticesCount;
    newScene = true;
    
    size_t stride = 14 * sizeof(float);
    size_t offset = 0;
    if (!scene->interleavedBuffer) {
        stride = 3 * sizeof(float);
        offset = scene->verticesCount * 3 * sizeof(float);
    }

    //this could be optimized to be done in a compute shader or a cuda kernel
    GLuint gaussianMeans;
    glGenBuffers(1, &gaussianMeans);
    glBindBuffer(GL_ARRAY_BUFFER, gaussianMeans);
    glBufferData(
        GL_ARRAY_BUFFER,
        scene->verticesCount * sizeof(Vertex), 
        scene->sceneDataBuffer.get(), GL_STATIC_DRAW
    );
    
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(3);
    //TODO: When you are sure everything works, don't use VBOs for Gaussian data anymore
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gaussianDataBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, scene->bufferSize, scene->sceneDataBuffer.get(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, gaussianDataBuffer);
    allocateSortingBuffers();

    //TODO: understand this
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    gaussianSceneCreated = true;
}

void Renderer::constructSplatSceneFromGPU(GPUScene& scene) {
    for (int i = 0; i < Viewport::n_viewports; i++) {
        Viewport::viewports[i].view_camera->lookAt(scene.centroid);
    }
    newScene = true;
    //NOTE: GPUScene cannot be used after this, all buffers are cleared to save memory
    gaussiansCount = scene.means.size(0);
    size_t meansBytes = gaussiansCount * 3 * sizeof(float);
    int floats_per_pt = 3  /*xyz*/ + 3 /*colors*/
                          + 1 /*opacity*/ + 3 /*scale*/;
    size_t wholeBufferSize = gaussiansCount * floats_per_pt * sizeof(float);

    void *gaussianMeans_glBuffer, *gaussianScene_glBuffer;
    size_t gaussianMeans_size, gaussianSceneBuffer_size;
    cudaGraphicsResource_t gaussianMeansResource, gaussianBufferResource;

    GLuint gaussianMeans;
    glGenBuffers(1, &gaussianMeans);
    glBindBuffer(GL_ARRAY_BUFFER, gaussianMeans);
    glBufferData(GL_ARRAY_BUFFER, meansBytes, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&gaussianMeansResource, gaussianMeans, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsMapResources(1, &gaussianMeansResource);

    cudaGraphicsResourceGetMappedPointer(&gaussianMeans_glBuffer, &gaussianMeans_size, gaussianMeansResource);
    cudaMemcpy(gaussianMeans_glBuffer, scene.means.const_data_ptr(), meansBytes, cudaMemcpyDeviceToDevice);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gaussianDataBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, wholeBufferSize, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&gaussianBufferResource, gaussianDataBuffer, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsMapResources(1, &gaussianBufferResource);
    cudaGraphicsResourceGetMappedPointer(&gaussianScene_glBuffer, &gaussianSceneBuffer_size, gaussianBufferResource);

    size_t offset = 0;
    cudaMemcpy(gaussianScene_glBuffer, scene.means.const_data_ptr(), meansBytes, cudaMemcpyDeviceToDevice);
    offset += scene.means.size(0) * 3;
    scene.means.reset();
    cudaMemcpy(
        (float*)gaussianScene_glBuffer + offset,
        scene.colors.const_data_ptr(),
        scene.colors.size(0) * 3 * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
    offset += scene.colors.size(0) * 3;
    scene.colors.reset();
    cudaMemcpy(
        (float*)gaussianScene_glBuffer + offset,
        scene.opacities.const_data_ptr(),
        scene.opacities.size(0) * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
    offset += scene.opacities.size(0);
    scene.opacities.reset();
    cudaMemcpy(
        (float*)gaussianScene_glBuffer + offset,
        scene.scales.const_data_ptr(),
        scene.scales.size(0) * 3 * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
    offset += scene.scales.size(0) * 3;
    scene.scales.reset();

    cudaGraphicsUnmapResources(1, &gaussianMeansResource);
    cudaGraphicsUnmapResources(1, &gaussianBufferResource);
    // Bind it to the shader position
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, gaussianDataBuffer);
    allocateSortingBuffers();

    glBindBuffer(GL_ARRAY_BUFFER, gaussianMeans);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(3);

    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    gaussianSceneCreated = true;
}

void Renderer::processGaussianSplats(int i) {
    // We aren't drawing anything, just computing
    glEnable(GL_RASTERIZER_DISCARD);

    glUseProgram(veryRealComputeProgram);
    Viewport::viewports[i].view_camera->registerModelView(veryRealComputeProgram);
    glDrawArrays(GL_POINTS, 0, gaussiansCount);

    glDisable(GL_RASTERIZER_DISCARD);

    if (::globalState.sortingEnabled) {
        //TIME_SANDWICH_START(CUDA_INTEROP)
        cudaGraphicsMapResources(4, cu_buffers);

        void *d_depth_ptr, *d_index_ptr, *d_sortedDepth_ptr, *d_sortedIndex_ptr;
        size_t depth_buffer_size, index_buffer_size, sorted_depth_size, sorted_index_size;
        CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(&d_depth_ptr, &depth_buffer_size, depth_buffer), true)
        cudaGraphicsResourceGetMappedPointer(&d_index_ptr, &index_buffer_size, index_buffer);
        cudaGraphicsResourceGetMappedPointer(&d_sortedDepth_ptr, &sorted_depth_size, sorted_depth_buffer);
        cudaGraphicsResourceGetMappedPointer(&d_sortedIndex_ptr, &sorted_index_size, sorted_index_buffer);

        RenderUtils::sort_gaussians_gpu(
            (float*)d_depth_ptr, (float*)d_sortedDepth_ptr,
            (int*)d_index_ptr, (int*)d_sortedIndex_ptr, gaussiansCount, newScene
        );

        cudaGraphicsUnmapResources(4, cu_buffers);
        //TIME_SANDWICH_END(CUDA_INTEROP)
    }


}

void Renderer::render(GLFWwindow* window) {
    Viewport::viewports[::globalState.selectedViewport].view_camera->handleInput(window);

    //TODO: use UBOs instead of normal uniforms
    for (int i = 0; i < Viewport::n_viewports; i++) {
        Viewport::renderOnViewport(i);

        if (Viewport::viewports[i].mesh) {
            glEnable(GL_DEPTH_TEST);

            glUseProgram(shaderProgram);
            Viewport::viewports[i].view_camera->registerModelView(shaderProgram);
            //TODO: pls dont forget -- i forgot
            glDrawArrays(::globalState.debugMode, 0, verticesCount);
        } else if (gaussianSceneCreated) {
            if (::globalState.renderingMode == GlobalState::RenderMode::PCD) {
                glUseProgram(shaderProgram);
                glDrawArrays(GL_POINTS, 0, gaussiansCount);
                continue;
            }
            glDisable(GL_DEPTH_TEST);
            processGaussianSplats(i);

            glUseProgram(gaussRenProgram);
            Viewport::viewports[i].view_camera->registerModelView(gaussRenProgram);
            Viewport::viewports[i].view_camera->uploadIntrinsics(gaussRenProgram, gaussiansCount);
            glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)0, gaussiansCount);
        }
        Viewport::viewports[i].view_camera->updateView();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Renderer::~Renderer() {
    if (gaussianSceneCreated) {
        cudaGraphicsUnregisterResource(depth_buffer);
        cudaGraphicsUnregisterResource(index_buffer);
        cudaGraphicsUnregisterResource(sorted_depth_buffer);
        cudaGraphicsUnregisterResource(sorted_index_buffer);
        RenderUtils::cleanUp();
    }
}
