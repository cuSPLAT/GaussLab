#include "renderer.h"
#include "shaders.h"
#include <GL/gl.h>

Renderer::Renderer(int width, int height): width(width), height(height) {}

void Renderer::initializeRendererBuffer() {
    glGenFramebuffers(1, &frameBuffer);

    // the texture that will act as a color buffer for rendering
    glGenTextures(1, &rendererBuffer);
    glBindTexture(GL_TEXTURE_2D, rendererBuffer);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RG8, width, height, 0,
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
}

GLuint Renderer::getRenderBuffer() {
    return rendererBuffer;
}

void Renderer::generateInitialBuffers() {
    // dummy vertices for now
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.0f,  0.5f, 0.0f
    };

    // if we suddenly have manu VBOs move the generation to another method
    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    // we will only bind the VBO during initialization, after that we will
    // only bind the VAO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &Shaders::vertexShader, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &Shaders::fragmentShader, nullptr);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // should be moved to another function if we have multiple shaders
    glUseProgram(shaderProgram);
}

void Renderer::render() {
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

    // drawing into the small window happens here
    // we only have one VAO and one shader that are always binded so
    // no need to always rebind them
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
