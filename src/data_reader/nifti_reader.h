#ifndef NIFTI_READER_H_
#define NIFTI_READER_H_

#include <cstdint>
#include <string>
#include <nifti1_io.h>

#include <glm/glm.hpp>

class SegmentationMask {
    nifti_image* m_niftiImage = nullptr;

    int m_length, m_width, m_height;

public:

    SegmentationMask(const std::string& filename);
    ~SegmentationMask();

    // yes I am using glm for indexing
    uint8_t operator[] (const glm::vec<3, int>& index) const;

    const uint8_t* data() const {
        // I am assuming that all outputs in total segmentator are single byte numbers
        // This is subject to change if anything else is found
        return static_cast<const uint8_t*>(m_niftiImage->data);
    }

};

#endif
