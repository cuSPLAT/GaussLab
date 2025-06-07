#include "viewport.h"
#include <iostream>
#include <core/renderer.h>

#include <imgui.h>
#include <memory>

// This is the fastest way if we are going to limit to only
// 5 viewports, else it will be have to be done dynamically
const char viewport_ids[5][11] = {
    "Viewport 1",
    "Viewport 2",
    "Viewport 3",
    "Viewport 4",
    "Viewport 5"
};
unsigned int Viewport::n_viewports = 0;
Viewport Viewport::viewports[5] = {};

Viewport::Viewport(int width, int height){
    viewport_id = viewport_ids[Viewport::n_viewports];
    view_camera = std::make_unique<Camera>(width, height);
    mesh = true;

    // also for now
    m_width = width;
    m_height = height;

    allocateFrameBuffer();
}

void Viewport::allocateFrameBuffer() {
    //TODO: Exceptions
    glGenFramebuffers(1, &frameBuffer);

    // the texture that will act as a color buffer for rendering
    glGenTextures(1, &frameBufferTexture);
    glBindTexture(GL_TEXTURE_2D, frameBufferTexture);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0,
        GL_RGB, GL_UNSIGNED_BYTE, nullptr
    );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // attach the texture as a color buffer, the texture here acts just as
    // data buffer
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frameBufferTexture, 0
    );

    GLuint depth;
    glGenTextures(1, &depth);
    glBindTexture(GL_TEXTURE_2D, depth);
    glTexImage2D(
    GL_TEXTURE_2D, 0, GL_DEPTH32F_STENCIL8, m_width, m_height, 0, 
    GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, nullptr
    );
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depth, 0);


    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Viewport::renderOnViewport(int i) {
    glBindFramebuffer(GL_FRAMEBUFFER, viewports[i].frameBuffer);

    // clear all bufferes before rendering
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Viewport::newViewport(int width, int height) {
    Viewport::viewports[Viewport::n_viewports] = Viewport(width, height);
    Viewport::n_viewports++;
}

// TODO: remove the renderer pointer. it is literally useless here
void Viewport::drawViewports_ImGui(Renderer* renderer) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    for (int i = 0; i < Viewport::n_viewports; i++) {
        // TODO: ONly allow a limited number of viewports
        if (ImGui::Begin(viewport_ids[i])) {
            if(ImGui::BeginChild("Render")) {
                if (ImGui::IsWindowHovered())
                    ::globalState.windowHovered = true;
                else ::globalState.windowHovered = false;

                if (ImGui::IsWindowFocused() && ::globalState.selectedViewport != i)
                    ::globalState.selectedViewport = i;

                ImVec2 size = ImGui::GetWindowSize();
                ImVec2 pos  = ImGui::GetWindowPos();
                ImVec2 window_pos = ImGui::GetCursorScreenPos();
                viewports[i].viewportPosData = {pos.x, pos.y, size.x, size.y};

                //TODO: don't always update, just update when window size changes
                viewports[i].view_camera->updateViewport(
                    size.x, size.y, viewports[i].mesh ? renderer->shaderProgram : renderer->gaussRenProgram
                );
                ImGui::Image((ImTextureID)viewports[i].frameBufferTexture, size);

                // ----------------------- Top-Toolbar -----------------------
                ImGui::SetCursorScreenPos(ImVec2(10 + window_pos.x, 10 + window_pos.y)); 
                ImGui::BeginGroup();
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.2f, 0.6f));

                if (ImGui::Button("Mesh"))
                    Viewport::viewports[i].mesh = true;
                ImGui::SameLine();
                if (ImGui::Button("Gaussian"))
                    Viewport::viewports[i].mesh = false;

                ImGui::PopStyleColor();
                ImGui::EndGroup();
                // ----------------------------------------------------------
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }
    ImGui::PopStyleVar();
}
