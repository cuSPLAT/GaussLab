#define DEFAULT_DICOM_DICTIONARY "/home/Abdelrahman/projects/gradproject/GausStudio/dictionary.txt"

#define getDataManipulator(file, tag) file.data_set()->element<tag>()->manipulator()

#include <string>

#include <vega/dicom/file.h>

class DicomReader {
    std::vector<vega::dicom::File> dicomFiles;

public:
    DicomReader(const std::string& dictionary = "");
    
    std::tuple<float*, int, int, int> readDirectory(const std::string& path);
};
