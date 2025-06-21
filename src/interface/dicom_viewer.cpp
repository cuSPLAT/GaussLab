#include "dicom_viewer.h"
#include <imgui.h>
#include <sstream>
#include <iomanip>
#include <vega/dicom/file.h>
#include <vega/dictionary/dictionary.h>
#include <vega/dicom/data_set.h>
#include <vega/dicom/data_element.h>
#include <fstream>
#include "data_reader/dicom_reader.h"

std::vector<DicomEntry> loadDicomTags(const std::string& path) {
    std::vector<DicomEntry> entries;
    try {
        // Check if file exists and is readable
        std::ifstream fileCheck(path, std::ios::binary);
        if (!fileCheck.good()) {
            std::cerr << "Error: Cannot open file: " << path << std::endl;
            return entries;
        }
        
        fileCheck.seekg(0, std::ios::end);
        if (fileCheck.tellg() == 0) {
            std::cerr << "Error: DICOM file is empty: " << path << std::endl;
            return entries;
        }
        fileCheck.close();

        vega::dicom::File file(path);
        auto dataset = file.data_set();
        
        if (!dataset) {
            std::cerr << "Error: Could not read dataset from file: " << path << std::endl;
            return entries;
        }

        for (auto it = dataset->begin(); it != dataset->end(); ++it) {
            try {
                auto element = *it;
                if (!element) continue;
                
                vega::Tag tag = element->tag();
                std::string tagName;
                auto page = vega::dictionary::Dictionary::instance().page_for(tag);
                if (page) {
                    tagName = page->name();
                } else {
                    tagName = "Unknown";
                }
                std::string value = "<Could not read>";
                if (tag == vega::Tag::PIXEL_DATA) {
                    value = "<Pixel Data not shown>";
                } else {
                    try {
                        value = element->str();
                    } catch (const std::exception& e) {
                        value = "<Could not read>";
                    }
                }
                entries.push_back({tag, tagName, value});
            } catch (const std::exception& e) {
                std::cerr << "Warning: Error reading element: " << e.what() << std::endl;
                continue;
            }
        }
    } catch (const vega::Exception& e) {
        std::cerr << "Vega Exception reading DICOM file " << path << ": " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception reading DICOM file " << path << ": " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error reading DICOM file: " << path << std::endl;
    }
    return entries;
}

void ShowDicomViewer(DicomReader& dcmReader) {
    // Only reload DICOM tag entries if the file changes
    static std::string lastDicomPath;
    static std::vector<DicomEntry> entries;
    const auto& dicomFilePaths = dcmReader.getDicomFilePaths();

    if (!dicomFilePaths.empty()) {
        if (dicomFilePaths[0] != lastDicomPath) {
            entries = loadDicomTags(dicomFilePaths[0]);
            lastDicomPath = dicomFilePaths[0];
        }
    }

    static char groupInput[5] = "";
    static char elementInput[5] = "";
    static std::string searchResult = "";
    static bool shouldScrollTagList = false;
    static bool shouldScrollSearchResult = false;

    ImGui::Begin("DICOM Viewer", nullptr, ImGuiWindowFlags_NoCollapse); // Allow no collapse

    // --- Search Tag Input Fields & Button ---
    ImGui::Text("Search Tag:");
    float inputFieldWidth = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) / 2;
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

        bool found = false;
        for (const auto& entry : entries) {
            if (entry.tag.group() == group && entry.tag.element() == element) {
                std::stringstream tagStr;
                tagStr << std::hex << std::uppercase << std::setfill('0')
                       << "(" << std::setw(4) << group << "," << std::setw(4) << element << ")";
                searchResult = tagStr.str() + " | " + entry.tagName + ": " + entry.value;
                found = true;
                break;
            }
        }
        if (!found) {
            std::stringstream tagStr;
            tagStr << std::hex << std::uppercase << std::setfill('0')
                   << "(" << std::setw(4) << group << "," << std::setw(4) << element << ")";
            searchResult = tagStr.str() + ": Not Found";
        }
        shouldScrollSearchResult = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // --- Search Result Area (fixed height, always visible) ---
    ImGui::Text("Search Result:");
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
            entry.tag.group(), entry.tag.element(),
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