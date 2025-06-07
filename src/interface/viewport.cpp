#include "viewport.h"

#include <iostream>
#include <imgui.h>

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

Viewport::Viewport() {
    viewport_id = viewport_ids[Viewport::n_viewports];
    view_camera = nullptr; // for now
    mesh = true;
}

void Viewport::newViewport() {
    Viewport::viewports[Viewport::n_viewports] = Viewport();
    Viewport::n_viewports++;
}

// TODO: remove the renderer pointer. it is literally useless here
void Viewport::drawViewports_ImGui(Renderer* renderer) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    for (int i = 0; i < Viewport::n_viewports; i++) {
        // TODO: ONly allow a limited number of viewports
        if (ImGui::Begin(viewport_ids[i])) {
            if(ImGui::BeginChild("Render")) {
                if (ImGui::IsWindowHovered()) ::globalState.windowHovered = true;
                else ::globalState.windowHovered = false;

                ImVec2 viewSize = ImGui::GetWindowSize();
                ImVec2 window_pos = ImGui::GetCursorScreenPos();

                //TODO: don't always update, just update when window size changes
                //viewports[i].view_camera->updateViewport(viewSize.x, viewSize.y, renderer->shaderProgram);
                ImGui::Image((ImTextureID)viewports[i].frameBufferTexture, viewSize);

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
