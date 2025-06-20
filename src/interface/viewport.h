#pragma once
#include <glm/fwd.hpp>
#define MAX_VIEWPORTS 5

#include <memory>

#include <core/camera.h>
#include <core/renderer.h>

extern const char viewport_ids[5][11];

struct Viewport {
    static Viewport viewports[5];
    static unsigned int n_viewports;
    
    static void drawViewports_ImGui(Renderer* renderer);
    static void newViewport(int width, int height);

    static void renderOnViewport(int i);
    static void lookAtScene_all(const glm::vec3& centroid);

    const char* viewport_id;
    std::unique_ptr<CameraView> view_camera;
    bool mesh = true;

    glm::vec4 viewportPosData;

    int m_width, m_height;
    GLuint frameBuffer, frameBufferTexture;

private:
    Viewport() = default;

    Viewport(int width, int height);
    void allocateFrameBuffer();
};
