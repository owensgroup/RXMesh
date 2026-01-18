#include <fstream>
#include <sstream>
#include <string>


#include <glm/vec3.hpp>

enum class Scenario : int
{
    Animated   = 0,
    MarkerFile = 1
};


inline bool read_markers(const std::string&                  path,
                         std::unordered_map<int, glm::vec3>& markers)
{
    std::ifstream in(path);
    if (!in.good()) {
        return false;
    }

    int K = 0;
    in >> K;
    if (!in.good() || K < 0) {
        return false;
    }

    markers.clear();
    markers.reserve(static_cast<size_t>(K));

    for (int k = 0; k < K; ++k) {
        float tx, ty, tz;
        int   vid;
        in >> tx >> ty >> tz >> vid;
        if (!in.good()) {
            return false;
        }

        markers[vid] = glm::vec3(tx, ty, tz);
    }

    return true;
}