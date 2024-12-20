#include "shaders.h"

// defining the shaders in variables just to prevent the
// headache of bundling the shader files with the binary

const char* Shaders::vertexShader = "#version 330 core"
    "layout (location = 0) in vec3 aPos;\n"
    "void main() {\n"
    "gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0)\n"
    "}\0";

const char* Shaders::fragmentShader = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main() {\n"
    "FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\0";
