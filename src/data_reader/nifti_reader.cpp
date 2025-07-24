#include "nifti_reader.h"
#include <nifti1_io.h>
#include <iostream>
#include <stdexcept>

#define TRUE 1

SegmentationMask::SegmentationMask(const std::string& filename) {
    m_niftiImage = nifti_image_read(filename.c_str(), TRUE);

    if (!m_niftiImage)
        throw std::runtime_error("Failed to read NIFTI");

    m_width = m_niftiImage->nx;
    m_length = m_niftiImage->ny;
    m_height = m_niftiImage->nz;

    std::cout << m_width << " " << m_length << " " << m_height << std::endl;
}

SegmentationMask::~SegmentationMask() {
    if (m_niftiImage != nullptr)
        nifti_image_free(m_niftiImage);
}

uint8_t SegmentationMask::operator[] (const glm::vec<3, int>& index) const {
    const int column = index.x;
    const int row = index.y;
    const int slice = index.z;

    if (column >= m_width || row >= m_length || slice >= m_height)
        throw std::runtime_error("Invalid Index");

    const int slice_area = m_width * m_length;

    // The nifti mask is flipped from some reason, this undoes the flip
    return data()[column + (m_length - 1 - row) * m_width + (m_height - slice - 1) * slice_area];
}

