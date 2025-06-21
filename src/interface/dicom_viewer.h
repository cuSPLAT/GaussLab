#pragma once

#include <vega/tag.h>
#include <string>
#include <vector>

class DicomReader; // Forward declaration

struct DicomEntry {
    vega::Tag tag;
    std::string tagName;
    std::string value;
};

std::vector<DicomEntry> loadDicomTags(const std::string& path);
void ShowDicomViewer(DicomReader& dcmReader); 