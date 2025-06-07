#pragma once
#define MAX_VIEWPORTS 5

#include <memory>

#include <core/camera.h>
#include <core/renderer.h>

extern const char viewport_ids[5][11];

struct Viewport {
    static Viewport viewports[5];
    static unsigned int n_viewports;
    
    static void drawViewports_ImGui(Renderer* renderer);
    static void newViewport();

    const char* viewport_id;
    std::unique_ptr<Camera> view_camera;
    bool mesh = true;

    GLuint frameBuffer, frameBufferTexture;

private:
    Viewport();
};
