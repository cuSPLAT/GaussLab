#ifndef SHADER_LOADER_H
#define SHADER_LOADER_H

#include <filesystem>
#include <string>
#include <unordered_map>
#include <glad/glad.h>

class ShaderLoader {
    std::unordered_map<std::string, GLuint> shaders;
    std::filesystem::path shaderFolder;

public:
    ShaderLoader(const std::filesystem::path& shaderFolder);
    void readAndCompile(const std::string& filepath, GLenum shadertype);
};

#endif
