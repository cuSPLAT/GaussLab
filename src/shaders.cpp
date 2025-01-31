#include "shaders.h"

// defining the shaders in variables just to prevent the
// headache of bundling the shader files with the binary

// Why do we multiply by C0 ?
const char* Shaders::vertexShader = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec3 inColor;\n"
    "out vec4 vertexColor;\n"
    "uniform mat4 view;\n"
    "void main() {\n"
    "gl_Position = view * vec4(aPos, 1.0);\n"
    "vertexColor = vec4(inColor * 0.282 + 0.5, 1.0);"
    "}\0";

const char* Shaders::fragmentShader = "#version 330 core\n"
    "in vec4 vertexColor;"
    "out vec4 FragColor;\n"
    "void main() {\n"
    "FragColor = vertexColor;\n"
    "}\0";
