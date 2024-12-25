#include <iostream>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3native.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "main_interface.h"

// Define Gaussian
struct GaussianPoint {
    float x, y, z;       // Position
    float size;          // Point size
    float r, g, b, a;    // Color
};

// Function prototypes
void renderGaussianSplatting(const std::vector<GaussianPoint>& points);
GLuint createPlaceholderTexture();

int main() {

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    Interface interface;
    interface.setupWindow();
    interface.initOpengl();
    interface.setupImgui();
    interface.setupRenderer();
    interface.startMainLoop();

    // Sample Gaussian points
    std::vector<GaussianPoint> points = {
        {0.0f, 0.0f, 0.0f, 0.1f, 1.0f, 0.0f, 0.0f, 1.0f},
        {0.5f, 0.5f, 0.5f, 0.1f, 0.0f, 1.0f, 0.0f, 1.0f},
        {-0.5f, -0.5f, -0.5f, 0.1f, 0.0f, 0.0f, 1.0f, 1.0f}
    };

    // Create a placeholder texture
    GLuint texture = createPlaceholderTexture();

    return 0;
}

void renderGaussianSplatting(const std::vector<GaussianPoint>& points) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for (const auto& point : points) {
        // to-do: render Gaussian splatting
    }
}

GLuint createPlaceholderTexture() {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Create a simple 2x2 colored texture
    unsigned char data[4 * 4] = {
        255, 255, 255, 255,
        255, 0, 0, 255,
        0, 255, 0, 255,
        0, 0, 255, 255
    };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    return texture;
}
