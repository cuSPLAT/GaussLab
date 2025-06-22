#include <iostream>
#include <string>
#include <fstream>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <iomanip>
#include <vector>
#include "stb_image_write.h"



using json = nlohmann::json;
std::string apiKey = "AIzaSyB_hCvu3NXM6Db0yUS2SoQ_EcnZ7m46U50";
// Base64 encoding function (minimal implementation)
std::string GetAxialBase64PNG(const DicomReader::DicomData& dicom, int axialSlice);
std::string base64_encode(const std::vector<unsigned char>& bytes) {
    static const char* base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    int val = 0, valb = -6;
    for (unsigned char c : bytes) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            result.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) result.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (result.size() % 4) result.push_back('=');
    return result;
}

// Curl write callback
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t totalSize = size * nmemb;
    output->append((char*)contents, totalSize);
    return totalSize;
}

// Function to send text + image to Gemini
std::string getGeminiResponseWithImage(const std::string& prompt, const DicomReader::DicomData& dicom, int axialSlice) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    std::string url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + apiKey;

    // Encode image to base64
    std::string imageBytes = GetAxialBase64PNG(dicom, axialSlice);

    // JSON request with image and prompt
    json payload = {
        {"contents", {{
            {"parts", {
                {{"text", prompt}},
                {{
                    "inline_data", {
                        {"mime_type", "image/png"},
                        {"data", imageBytes}
                    }
                }}
            }}
        }}}
    };

    std::string jsonData = payload.dump();

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            try {
                // std::cout << "Response received: " << readBuffer << std::endl;
                json responseJson = json::parse(readBuffer);
                
                // Check if the response has the expected structure
                if (responseJson.contains("candidates") && 
                    !responseJson["candidates"].empty() && 
                    responseJson["candidates"][0].contains("content") &&
                    responseJson["candidates"][0]["content"].contains("parts") &&
                    !responseJson["candidates"][0]["content"]["parts"].empty() &&
                    responseJson["candidates"][0]["content"]["parts"][0].contains("text")) {
                    
                    return responseJson["candidates"][0]["content"]["parts"][0]["text"];
                } else {
                    std::cerr << "Unexpected response structure: " << responseJson.dump(2) << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "JSON parse error: " << e.what() << std::endl;
                std::cerr << "Raw response: " << readBuffer << std::endl;
            }
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    curl_global_cleanup();
    return "Error: Could not get response from Gemini API";
}
std::string GetAxialBase64PNG(const DicomReader::DicomData& dicom, int axialSlice) {
    int width = dicom.width;
    int height = dicom.length;  // axial is XY plane → width × length
    const float* buffer = dicom.buffer.get();

    // Step 1: Extract axial slice as grayscale 8-bit
    std::vector<unsigned char> grayscale(width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int idx = axialSlice * width * height + y * width + x;
            grayscale[y * width + x] = static_cast<unsigned char>(std::clamp(buffer[idx], 0.0f, 1.0f) * 255);
        }

    // Step 2: Encode to PNG in memory
    std::vector<unsigned char> pngData;
    auto pngWriteCallback = [](void* context, void* data, int size) {
        std::vector<unsigned char>* output = reinterpret_cast<std::vector<unsigned char>*>(context);
        unsigned char* bytes = reinterpret_cast<unsigned char*>(data);
        output->insert(output->end(), bytes, bytes + size);
    };

    stbi_write_png_to_func(pngWriteCallback, &pngData, width, height, 1, grayscale.data(), width);

    // Step 3: Encode PNG binary to base64
    return base64_encode(pngData);  // Return only the base64 data without the MIME type prefix
}


// int main() {
//     std::string apiKey = "AIzaSyB_hCvu3NXM6Db0yUS2SoQ_EcnZ7m46U50";
//     std::string prompt = "explain what at image";
//     std::string imagePath ="/home/zain/GS/GausStudio/image.png";
//     std::string result = getGeminiResponseWithImage(prompt, imagePath, apiKey);
//     if (!result.empty()) {
//         std::cout << "Extracted Text:\n" << result << std::endl;
//     } else {
//         std::cerr << "Failed to extract text from response." << std::endl;
//     }

//     return 0;
// }

