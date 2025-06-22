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
                        // Check VR type to determine if we can safely convert to string
                        const auto& vr = element->vr();
                        
                        // Handle different VR types appropriately
                        if (vr == vega::vr::SQ) {
                            // Sequence - show number of items
                            if (element->is_sequence()) {
                                size_t count = 0;
                                for (auto it = element->begin(); it != element->end(); ++it) {
                                    count++;
                                }
                                value = "<Sequence with " + std::to_string(count) + " items>";
                            } else {
                                value = "<Sequence>";
                            }
                        } else if (vr == vega::vr::OB || vr == vega::vr::OW || vr == vega::vr::UN) {
                            // Binary data - show size
                            value = "<Binary data, " + std::to_string(element->length()) + " bytes>";
                        } else if (vr == vega::vr::AT) {
                            // Attribute tag - special handling
                            value = element->str();
                        } else {
                            // Try to convert to string for other types
                            value = element->str();
                            if (value.empty()) {
                                value = "<Empty>";
                            }
                        }
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

