#include <glm/fwd.hpp>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace Tools {
namespace MeshSlicing {

void captureMousePos(GLFWwindow* window, int action);

void createPlane(const glm::vec3& ray);

};
};
