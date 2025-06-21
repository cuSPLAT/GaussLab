#include "request.hpp"
#include "data_reader/dicom_reader.h"
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

std::string GetAxialBase64PNG(const DicomReader::DicomData& dicom, int axialSlice) {
    int width = dicom.width;
    int height = dicom.length;  // axial is XY plane → width × length
    const float* buffer = dicom.buffer.get();

    // Step 1: Find min and max in the slice for windowing
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int idx = axialSlice * width * height + y * width + x;
            float val = buffer[idx];
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
    // If min==max, avoid division by zero
    if (minVal == maxVal) maxVal = minVal + 1.0f;

    // Step 2: Extract axial slice as windowed 8-bit grayscale
    std::vector<unsigned char> grayscale(width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int idx = axialSlice * width * height + y * width + x;
            float val = buffer[idx];
            float norm = (val - minVal) / (maxVal - minVal); // [0,1]
            grayscale[y * width + x] = static_cast<unsigned char>(std::clamp(norm, 0.0f, 1.0f) * 255);
        }

    // Step 3: Encode to PNG in memory
    std::vector<unsigned char> pngData;
    auto pngWriteCallback = [](void* context, void* data, int size) {
        std::vector<unsigned char>* output = reinterpret_cast<std::vector<unsigned char>*>(context);
        unsigned char* bytes = reinterpret_cast<unsigned char*>(data);
        output->insert(output->end(), bytes, bytes + size);
    };

    stbi_write_png_to_func(pngWriteCallback, &pngData, width, height, 1, grayscale.data(), width);

    // Step 4: Encode PNG binary to base64
    return base64_encode(pngData);  // Return only the base64 data without the MIME type prefix
}

std::string getGeminiResponseWithImage(const std::string& prompt, DicomReader::DicomData& dicom, int axialSlice) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return "Error: Failed to initialize CURL";
    }

    // Encode image to base64
    std::string base64Image = GetAxialBase64PNG(dicom, axialSlice);

    CURLcode res;
    std::string readBuffer;

    std::string url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + apiKey;

    // JSON request with image and prompt
    json payload = {
        {"contents", {{
            {"parts", {
                {{"text", prompt}},
                {{
                    "inline_data", {
                        {"mime_type", "image/png"},
                        {"data", base64Image}
                    }
                }}
            }}
        }}}
    };

    std::string jsonData = payload.dump();
    // std::cout << "Sending request to Gemini API..." << std::endl;

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

    return "Error: Could not get response from Gemini API";
} 