#include "renderer.h"
#include "shaders.h"
#include <GL/gl.h>

Renderer::Renderer() = default;

void Renderer::generateInitialBuffers() {
    // dummy vertices for now
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.0f,  0.5f, 0.0f
    };

    // if we suddenly have manu VBOs move the generation to another method
    glGenBuffers(1, &VBO);
    glGenFramebuffers(1, &frameBuffer);
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
}
