#include <core/renderer.h>
#include <core/shaders.h>

#include <cuda/render_utils.h>
#include <cuda/debug_utils.h>

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
    width(width), height(height), camera(width, height) {}

void Renderer::newRenderBuffer() {
    glGenFramebuffers(1, &frameBuffers[n_created]);

    // the texture that will act as a color buffer for rendering
    glGenTextures(1, &rendererBuffers[n_created]);
    glBindTexture(GL_TEXTURE_2D, rendererBuffers[n_created]);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
        GL_RGB, GL_UNSIGNED_BYTE, nullptr
    );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // attach the texture as a color buffer, the texture here acts just as
    // data buffer
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffers[n_created]);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rendererBuffers[n_created], 0
    );

    GLuint depth;
    glGenTextures(1, &depth);
    glBindTexture(GL_TEXTURE_2D, depth);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_DEPTH32F_STENCIL8, width, height, 0, 
        GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, nullptr
    );
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depth, 0);


    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    n_created++;

}

GLuint Renderer::getRenderBuffer(int id) {
    return rendererBuffers[id];
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

    glGenBuffers(1, &quadVBO);
    glGenBuffers(1, &quadEBO);

    glGenBuffers(1, &gaussianDataBuffer);
    
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
    int  success;
    char infoLog[512];
    glGetShaderiv(PCDFragmentShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(PCDFragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    
    }    

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

    glDeleteShader(PCDVertexShader);
    glDeleteShader(PCDFragmentShader);
    glDeleteShader(veryRealComputeShader);
    glDeleteShader(GaussianVertexShader);

}

void Renderer::constructMeshScene(Scene* scene, std::vector<Vertex>& vertices) {
    verticesCount = vertices.size() / 2;
    camera.setCentroid(scene->centroid);
    camera.registerView(shaderProgram);
    newScene = true;

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Temporary normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    const GLuint optional = glGetUniformLocation(shaderProgram, "planeExists");
    glUniform1i(optional, false);
    //glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(3 * sizeof(float)));
    //glEnableVertexAttribArray(1);

    std::cout << "Current vertices count " << verticesCount << std::endl;
    std::cout << "Total buffer size in bytes " << vertices.size() * 3 << std::endl;

   glEnable(GL_DEPTH_TEST);
    //TODO: understand this
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Renderer::constructSplatScene(Scene* scene) {
    // the data of the rectangle that a Gaussian will occupy
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);
    // -----------------------------------------------
    
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

    //*Ugly ahh code*
    cu_buffers[0] = depth_buffer;
    cu_buffers[1] = index_buffer;
    cu_buffers[2] = sorted_depth_buffer;
    cu_buffers[3] = sorted_index_buffer;

    //TODO: When you are sure everything works, don't use VBOs for Gaussian data anymore
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gaussianDataBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, scene->bufferSize, scene->sceneDataBuffer.get(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, gaussianDataBuffer);
}

Camera* Renderer::getCamera() {
    return &camera;
}

void Renderer::selectFrameBuffer(int id) {
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffers[id]);
}

void Renderer::render(GLFWwindow* window) {
    camera.handleInput(window);

    for (int i = 0; i < n_created; i++) {
        selectFrameBuffer(i);
    }

    selectFrameBuffer(0);
    // We aren't drawing anything, just computing
    glEnable(GL_RASTERIZER_DISCARD);
    glUseProgram(veryRealComputeProgram);
    camera.registerView(veryRealComputeProgram);
    glDrawArrays(GL_POINTS, 0, verticesCount);

    glDisable(GL_RASTERIZER_DISCARD);

    if (::globalState.sortingEnabled) {
    //TIME_SANDWICH_START(CUDA_INTEROP)
        //cudaGraphicsMapResources(4, cu_buffers);

        //void *d_depth_ptr, *d_index_ptr, *d_sortedDepth_ptr, *d_sortedIndex_ptr;
        //size_t depth_buffer_size, index_buffer_size, sorted_depth_size, sorted_index_size;
        //CHECK_CUDA(cudaGraphicsResourceGetMappedPointer(&d_depth_ptr, &depth_buffer_size, depth_buffer), true)
        //cudaGraphicsResourceGetMappedPointer(&d_index_ptr, &index_buffer_size, index_buffer);
        //cudaGraphicsResourceGetMappedPointer(&d_sortedDepth_ptr, &sorted_depth_size, sorted_depth_buffer);
        //cudaGraphicsResourceGetMappedPointer(&d_sortedIndex_ptr, &sorted_index_size, sorted_index_buffer);

        //RenderUtils::sort_gaussians_gpu(
        //    (float*)d_depth_ptr, (float*)d_sortedDepth_ptr,
        //    (int*)d_index_ptr, (int*)d_sortedIndex_ptr, verticesCount, newScene
        //);

        //cudaGraphicsUnmapResources(4, cu_buffers);
        //TIME_SANDWICH_END(CUDA_INTEROP)
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // drawing into the small window happens here
    // we only have one VAO and one shader that are always binded
    // no need to rebind them on every draw call
    //if (globalState.renderingMode == GlobalState::RenderMode::PCD) {
    glUseProgram(shaderProgram);
    camera.registerView(shaderProgram);
    //TODO: pls dont forget
    glDrawArrays(::globalState.debugMode, 0, verticesCount);
    //} else {
    //    glUseProgram(gaussRenProgram);
    //    camera.registerView(gaussRenProgram);
    //    camera.uploadIntrinsics(gaussRenProgram);
    //    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)0, verticesCount);
    //}

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    camera.updateView();
}

Renderer::~Renderer() {
    //cudaGraphicsUnregisterResource(depth_buffer);
    //cudaGraphicsUnregisterResource(index_buffer);
    //cudaGraphicsUnregisterResource(sorted_depth_buffer);
    //cudaGraphicsUnregisterResource(sorted_index_buffer);
    //RenderUtils::cleanUp();
}
