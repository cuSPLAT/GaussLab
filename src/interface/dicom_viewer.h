#pragma once

#include <dcmtk/dcmdata/dctk.h>
#include <string>
#include <vector>

struct DicomEntry {
    DcmTag tag;
    std::string tagName;
    std::string value;
};

std::vector<DicomEntry> loadDicomTags(const std::string& path);
void ShowDicomViewer(std::vector<DicomEntry>& entries, DcmDataset* dataset); 