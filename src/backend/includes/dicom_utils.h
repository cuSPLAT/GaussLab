// FILE: dicom_utils.h
#ifndef DICOM_UTILS_H
#define DICOM_UTILS_H

#include <string>
#include <vector>
#include <sstream>
#include <array>
#include <optional>
#include <algorithm> // std::copy_n, std::find_if
#include <stdexcept> // std::stod exceptions

#include "dcmcore/data_set.h" // DataElement, Tag, VR, Buffer, Defs
#include "dcmcore/tag.h"

namespace DicomTags_NG
{
    constexpr dcmcore::Tag PixelData(0x7FE0, 0x0010);
    constexpr dcmcore::Tag ImagePositionPatient(0x0020, 0x0032);
    constexpr dcmcore::Tag ImageOrientationPatient(0x0020, 0x0037);
    constexpr dcmcore::Tag PixelSpacing(0x0028, 0x0030);

    constexpr dcmcore::Tag Rows(0x0028, 0x0010);
    constexpr dcmcore::Tag Columns(0x0028, 0x0011);

    constexpr dcmcore::Tag RescaleSlope(0x0028, 0x1053);
    constexpr dcmcore::Tag RescaleIntercept(0x0028, 0x1052);
}

namespace dcm
{
    // check for tag presence.
    inline bool hasAttr(dcmcore::DataSet& dataSet, const dcmcore::Tag& tag)
    {
        const dcmcore::DataElement* elem = dataSet.GetElement(tag);
        if (elem)
        {
            if (elem->length() > 0) {
                return true;
            } else {
                // Element exists but has 0 length
                if (tag == DicomTags_NG::PixelData) {
                    return false;
                }
                // For other tags, original logic considered 0-length as "present"
                return true;
            }
        }
        return false;
    }

    // Get US (Unsigned Short) tag value.
    inline unsigned short getUSTagValue(dcmcore::DataSet& dataSet, const dcmcore::Tag& tag, unsigned short defaultValue = 0)
    {
        uint16_t value; // dcmcore uses standard integer types
        if (dataSet.Get<uint16_t>(tag, value))
        {
            return value;
        }
        return defaultValue;
    }

    // Trim leading/trailing whitespace from a string.
    inline std::string trim_whitespace(const std::string& str)
    {
        const std::string whitespace = " \t\n\r\f\v";
        size_t first = str.find_first_not_of(whitespace);
        if (std::string::npos == first)
            return ""; // Return empty string if only whitespace
        size_t last = str.find_last_not_of(whitespace);
        return str.substr(first, (last - first + 1));
    }

    // Parse DICOM Decimal String (DS) into a vector of doubles.
    inline std::vector<double> getDSTagValues(dcmcore::DataSet& dataSet, const dcmcore::Tag& tag)
    {
        std::vector<double> values;
        std::string ds_string;
        if (dataSet.GetString(tag, ds_string))
        {
            // dcmcore::DataElement::GetString already removes a single trailing space if present.
            // trim_whitespace here will handle multiple spaces or other whitespace chars if they occur.
            std::stringstream ss(ds_string);
            std::string item;
            while (std::getline(ss, item, '\\')) // DICOM DS values are separated by backslash
            {
                std::string trimmed_item = trim_whitespace(item); // trim each part
                if (!trimmed_item.empty())
                {
                    values.push_back(std::stod(trimmed_item));
                }
            }
        }
        return values;
    }

    inline std::optional<std::array<double, 3>> getImagePositionPatient(dcmcore::DataSet& dataSet)
    {
        std::vector<double> values = getDSTagValues(dataSet, DicomTags_NG::ImagePositionPatient);
        if (values.size() == 3) {
            std::array<double, 3> result;
            std::copy_n(values.begin(), 3, result.begin());
            return result;
        }
        return std::nullopt;
    }

    inline std::optional<std::array<double, 6>> getImageOrientationPatient(dcmcore::DataSet& dataSet)
    {
        std::vector<double> values = getDSTagValues(dataSet, DicomTags_NG::ImageOrientationPatient);
        if (values.size() == 6) {
            std::array<double, 6> result;
            std::copy_n(values.begin(), 6, result.begin());
            return result;
        }
        return std::nullopt;
    }

    inline std::optional<std::array<double, 2>> getPixelSpacing(dcmcore::DataSet& dataSet)
    {
        std::vector<double> values = getDSTagValues(dataSet, DicomTags_NG::PixelSpacing);
        if (values.size() == 2)
        {
            std::array<double, 2> result;
            std::copy_n(values.begin(), 2, result.begin());
            return result;
        }
        return std::nullopt;
    }

    inline unsigned short getRows(dcmcore::DataSet& dataSet, unsigned short defaultValue = 0)
    { return getUSTagValue(dataSet, DicomTags_NG::Rows, defaultValue); }

    inline unsigned short getColumns(dcmcore::DataSet& dataSet, unsigned short defaultValue = 0)
    { return getUSTagValue(dataSet, DicomTags_NG::Columns, defaultValue); }

    inline double getRescaleSlope(dcmcore::DataSet& dataSet, double defaultValue = 1.0)
    {
        std::vector<double> values = getDSTagValues(dataSet, DicomTags_NG::RescaleSlope);
        return values.empty() ? defaultValue : values[0];
    }
    
    inline double getRescaleIntercept(dcmcore::DataSet& dataSet, double defaultValue = 0.0)
    {
        std::vector<double> values = getDSTagValues(dataSet, DicomTags_NG::RescaleIntercept);
        return values.empty() ? defaultValue : values[0];
    }

} // namespace dcm

#endif
