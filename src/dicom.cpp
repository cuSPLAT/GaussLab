#include <iostream>
#include <dcmtk/dcmdata/dctk.h>

void extractCleanMetadata(const std::string& filePath) {
    DcmFileFormat fileFormat;
    OFCondition status = fileFormat.loadFile(filePath.c_str());

    if (status.good()) {
        std::cout << "DICOM Metadata:\n" << std::endl;
        DcmDataset* dataset = fileFormat.getDataset();
        DcmStack stack;

        while (dataset->nextObject(stack, true).good()) {
            DcmObject* obj = stack.top();
            DcmTag tag = obj->getTag();

            // Try to cast to DcmElement for value extraction
            DcmElement* elem = dynamic_cast<DcmElement*>(obj);
            if (!elem) continue;

            OFString value;
            OFCondition cond;

            if (tag == DCM_PixelData) {
                std::cout << tag << " | Pixel Data: <Pixel Data not shown>" << std::endl;
            } else {
                cond = elem->getOFString(value, 0);
                if (cond.good()) {
                    std::cout << tag << " | " << tag.getTagName() << ": " << value << std::endl;
                } else {
                    std::cout << tag << " | " << tag.getTagName() << ": <Could not read value>" << std::endl;
                }
            }
        }
    } else {
        std::cerr << "Error: Cannot read DICOM file (" << status.text() << ")" << std::endl;
    }
}
void printTagValue(DcmDataset* dataset, const DcmTagKey& tagKey) {
    OFString value;
    OFCondition status = dataset->findAndGetOFString(tagKey, value);

    DcmTag tag(tagKey);
    const char* tagName = tag.getTagName();  // Get tag name

    if (status.good()) {
        std::cout << tagKey << " | " << tagName << ": " << value << std::endl;
    } else {
        std::cout << tagKey << " | " << tagName << ": None" << std::endl;
    }
}

int main() {
    std::string path = "/home/zain/Downloads/fef7e115-c8c1c523-1e67bffb-13f9223d-1fec008f/55f205a41f404a069013d1728567bc8f Anonymized460/Unknown Study/CT/CT000000.dcm";
    
    // Load the file
    DcmFileFormat fileFormat;
    OFCondition status = fileFormat.loadFile(path.c_str());

    if (status.good()) {
        DcmDataset* dataset = fileFormat.getDataset();

        // Extract and print all metadata
        extractCleanMetadata(path);

        // Print specific tags
        std::cout << "\nSelected Tags:\n" << std::endl;
        printTagValue(dataset, DCM_PatientName);
        printTagValue(dataset, DCM_StudyDate);
        printTagValue(dataset, DCM_Modality);
        printTagValue(dataset, DcmTagKey(0x0010, 0x0010));
    } else {
        std::cerr << "Error loading DICOM file: " << status.text() << std::endl;
    }

    // Wait for user input to keep console open
    std::cout << "\nPress Enter to exit..." << std::endl;
    std::cin.get();
    return 0;
}
