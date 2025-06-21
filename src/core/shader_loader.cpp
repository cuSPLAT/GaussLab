#include "shader_loader.h"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>

ShaderLoader::ShaderLoader(const std::filesystem::path& shaderFolder) : shaderFolder(shaderFolder) {}

void ShaderLoader::readAndCompile(const std::string& filename, GLenum shadertype) {
    std::ifstream file(shaderFolder / filename);
    std::stringstream shaderbuffer;
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        std::cerr << "Make sure it is in the shader directory" << std::endl;
        return;
    }
    shaderbuffer << file.rdbuf();
    std::string shader_string = shaderbuffer.str();
    const char* shader_string_c = shader_string.c_str();

    GLuint shader = glCreateShader(shadertype);
    glShaderSource(shader, 1, &shader_string_c, nullptr);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\nTYPE: "
            << (shadertype == GL_VERTEX_SHADER ? "VERTEX\n" : "FRAGMENT\n") 
            << infoLog << std::endl;
        return;
    }

    shaders["nameplaceholder"] = shader;
}
