// dcmcore/tag_table.cpp
#include "dcmcore/tag_table.h" // Includes DataEntry and VR (via data_entry.h -> defs.h and vr.h)
#include "dcmcore/vr.h"        // Specifically for VR::Type enum values

namespace dcmcore {

// Define your essential DataEntry items here
// Ensure they are sorted by tag value for binary search in DataDictionary::FindEntry
const DataEntry g_tag_table[] = {
    // Group 0x0020
    { 0x00200032, VR::DS, "ImagePositionPatient",        "Image Position (Patient)" },
    { 0x00200037, VR::DS, "ImageOrientationPatient",     "Image Orientation (Patient)" },
    // Group 0x0028
    { 0x00280010, VR::US, "Rows",                        "Rows" },
    { 0x00280011, VR::US, "Columns",                     "Columns" },
    { 0x00280030, VR::DS, "PixelSpacing",                "Pixel Spacing" },
    // Group 0x7FE0
    { 0x7FE00010, VR::OB, "PixelData",                   "Pixel Data" } // Or VR::OW depending on context, OB is safer for raw
};

// Update TAG_COUNT to reflect the actual number of entries above
const std::size_t TAG_COUNT = sizeof(g_tag_table) / sizeof(DataEntry);

} // namespace dcmcore