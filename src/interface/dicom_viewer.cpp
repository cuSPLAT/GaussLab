#include "dicom_viewer.h"
#include <imgui.h>
#include <sstream>
#include <iomanip>

std::vector<DicomEntry> loadDicomTags(const std::string& path) {
    std::vector<DicomEntry> entries;
    DcmFileFormat fileFormat;
    OFCondition status = fileFormat.loadFile(path.c_str());

    if (status.good()) {
        DcmDataset* dataset = fileFormat.getDataset();
        DcmStack stack;

        while (dataset->nextObject(stack, true).good()) {
            DcmObject* obj = stack.top();
            DcmTag tag = obj->getTag();

            DcmElement* elem = dynamic_cast<DcmElement*>(obj);
            if (!elem) continue;

            OFString valueStr;
            std::string value = "<Could not read>";
            if (tag == DCM_PixelData) {
                value = "<Pixel Data not shown>";
            } else if (elem->getOFString(valueStr, 0).good()) {
                value = valueStr.c_str();
            }

            entries.push_back({
                tag,
                tag.getTagName(),
                value
            });
        }
    }
    return entries;
}

void ShowDicomViewer(std::vector<DicomEntry>& entries, DcmDataset* dataset) {
    static char groupInput[5] = "";
    static char elementInput[5] = "";
    static std::string searchResult = "";
    static bool shouldScrollTagList = false;
    static bool shouldScrollSearchResult = false;

    ImGui::Begin("DICOM Viewer", nullptr, ImGuiWindowFlags_NoCollapse); // Allow no collapse

    // --- Search Tag Input Fields & Button ---
    ImGui::Text("Search Tag:");
    
    // Input fields will take full available width, stacked vertically for clear visibility
    float inputFieldWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
    
    // Explicitly set width for each input field
    ImGui::SetNextItemWidth(inputFieldWidth);
    ImGui::InputTextWithHint("##Group", "Group (hex)", groupInput, sizeof(groupInput), ImGuiInputTextFlags_CharsHexadecimal);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(inputFieldWidth);
    ImGui::InputTextWithHint("##Element", "Element (hex)", elementInput, sizeof(elementInput), ImGuiInputTextFlags_CharsHexadecimal);

    // Search Button - always visible below inputs, takes full width
    if (ImGui::Button("Search", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
        uint16_t group = 0, element = 0;
        std::stringstream ss1, ss2;
        ss1 << std::hex << groupInput;
        ss1 >> group;
        ss2 << std::hex << elementInput;
        ss2 >> element;

        DcmTagKey key(group, element);
        OFString val;
        DcmTag tag(key);

        if (dataset->findAndGetOFString(key, val).good()) {
            std::string tagStr = tag.toString().c_str();
            std::string tagName = tag.getTagName();
            std::string valStr = val.c_str();
            searchResult = "(" + tagStr + ") | " + tagName + ": " + valStr;
        } else {
            std::string tagStr = tag.toString().c_str();
            std::string tagName = tag.getTagName();
            searchResult = "(" + tagStr + ") | " + tagName + ": Not Found";
        }
        shouldScrollSearchResult = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // --- Search Result Area (fixed height, always visible) ---
    ImGui::Text("Search Result:");
    // Fixed height to ensure it's always on screen, enough for 3-4 lines of text
    float searchResultFixedH = ImGui::GetTextLineHeightWithSpacing() * 2 + ImGui::GetStyle().ItemSpacing.y * 2;
    ImGui::BeginChild("searchResultChild", ImVec2(0, searchResultFixedH), true, ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::TextWrapped("%s", searchResult.c_str());
    if (shouldScrollSearchResult) {
        ImGui::SetScrollHereY(1.0f);
        shouldScrollSearchResult = false;
    }
    ImGui::EndChild();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // --- Tag List Area (takes remaining available space) ---
    ImGui::Text("Tags:");
    // ImVec2(0,0) makes this child take up the rest of the available height in the parent window.
    ImGui::BeginChild("tagListChild", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::Spacing();
    for (size_t i = 0; i < std::min<size_t>(20, entries.size()); ++i) {
        const auto& entry = entries[i];
        ImGui::TextWrapped("(%04X,%04X) | %-30s: %s",
            entry.tag.getGTag(), entry.tag.getETag(),
            entry.tagName.c_str(), entry.value.c_str());
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }
    if (shouldScrollTagList) {
        ImGui::SetScrollHereY(1.0f);
        shouldScrollTagList = false;
    }
    ImGui::EndChild();

    ImGui::End();
} 