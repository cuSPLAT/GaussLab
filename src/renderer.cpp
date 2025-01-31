#include "renderer.h"
#include "shaders.h"
#include "camera.h"

#include <iostream>

Renderer::Renderer(int width, int height): width(width), height(height) {}

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
}

GLuint Renderer::getRenderBuffer() {
    return rendererBuffer;
}

void Renderer::generateInitialBuffers() {
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &Shaders::vertexShader, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &Shaders::fragmentShader, nullptr);
    glCompileShader(fragmentShader);
    
    //TODO: move this to a macro or inline function
    int  success;
    char infoLog[512];
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // should be moved to another function if we have multiple shaders
    glUseProgram(shaderProgram);
}

void Renderer::constructScene(Scene* scene) {
    size_t vertices_count = scene->vertexPos.size();
    size_t color_count = scene->vertexColor.size();

    camera.registerView(shaderProgram);
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices_count * sizeof(float), scene->vertexPos.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    GLuint colorBuffer;
    glGenBuffers(1, &colorBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, color_count * sizeof(float), scene->vertexColor.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);

}

Camera* Renderer::getCamera() {
    return &camera;
}

void Renderer::render(GLFWwindow* window) {
    camera.registerView(shaderProgram);
    camera.handleInput(window);

    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    // drawing into the small window happens here
    // we only have one VAO and one shader that are always binded so
    // no need to always rebind them
    camera.updateView();
    glDrawArrays(GL_POINTS, 0, 140000);


    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
