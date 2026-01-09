#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

/**
 * Physical parameters for each mesh object in the scene
 */
template <typename T>
struct MeshConfig {
    std::string filepath;

    // Material properties
    T density        = 1000.0;    // kg/m^3
    T young_mod      = 1e5;       // Pa (Young's modulus)
    T poisson_ratio  = 0.4;       // Dimensionless (nu)
    T bending_stiff  = 1e8;       // Bending stiffness

    // Instance generation (3D grid)
    int num_instances_x = 1;
    int num_instances_y = 1;
    int num_instances_z = 1;
    T   spacing_x = 2.0;
    T   spacing_y = 2.0;
    T   spacing_z = 2.0;
    T   offset_x  = 0.0;
    T   offset_y  = 0.0;
    T   offset_z  = 0.0;

    // Scale (applied to each instance)
    T scale = 1.0;

    // Initial velocity (shared by all instances)
    T velocity_x = 0.0;
    T velocity_y = 0.0;
    T velocity_z = 0.0;

    // Constraints (true = fixed in that direction)
    bool fixed_x = false;
    bool fixed_y = false;
    bool fixed_z = false;
};

/**
 * Global simulation parameters
 */
template <typename T>
struct SimulationConfig {
    T time_step      = 0.01;      // Time step (h)
    T fricition_coef = 0.11;      // Friction coefficient (mu)
    T stiffness_coef = 4e4;       // Stiffness coefficient
    T tol            = 0.01;      // Solver tolerance
    T dhat           = 0.1;       // Contact distance threshold
    T kappa          = 1e5;       // Contact barrier stiffness
    int num_steps    = 5;         // Number of simulation steps

    // Ground plane
    T ground_x = 0.0;
    T ground_y = -1.0;
    T ground_z = 0.0;
    T ground_nx = 0.0;
    T ground_ny = 1.0;
    T ground_nz = 0.0;
};

/**
 * Complete scene configuration
 */
template <typename T>
struct SceneConfig {
    SimulationConfig<T> simulation;
    std::vector<MeshConfig<T>> meshes;
};

/**
 * Simple JSON-like config parser
 * Format:
 * simulation {
 *   time_step 0.01
 *   friction_coef 0.11
 *   ...
 * }
 * mesh {
 *   filepath "path/to/mesh.obj"
 *   density 1000
 *   young_mod 1e5
 *   translate_y 2.5
 *   ...
 * }
 * mesh {
 *   ...
 * }
 */
template <typename T>
class SceneConfigParser {
public:
    static bool parse(const std::string& filename, SceneConfig<T>& config) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            fprintf(stderr, "Error: Could not open config file: %s\n", filename.c_str());
            return false;
        }

        std::string line;
        std::string current_section;
        MeshConfig<T> current_mesh;
        bool in_mesh = false;

        while (std::getline(file, line)) {
            // Remove comments
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }

            // Trim whitespace
            line = trim(line);
            if (line.empty()) continue;

            // Check for section headers
            if (line == "simulation {") {
                current_section = "simulation";
                continue;
            } else if (line == "mesh {") {
                current_section = "mesh";
                in_mesh = true;
                current_mesh = MeshConfig<T>();  // Reset
                continue;
            } else if (line == "}") {
                if (in_mesh) {
                    config.meshes.push_back(current_mesh);
                    in_mesh = false;
                }
                current_section = "";
                continue;
            }

            // Parse key-value pairs
            std::istringstream iss(line);
            std::string key;
            iss >> key;

            if (current_section == "simulation") {
                parse_simulation_param(key, iss, config.simulation);
            } else if (current_section == "mesh") {
                parse_mesh_param(key, iss, current_mesh);
            }
        }

        return !config.meshes.empty();
    }

private:
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }

    static void parse_simulation_param(const std::string& key, std::istringstream& iss, SimulationConfig<T>& sim) {
        if (key == "time_step") iss >> sim.time_step;
        else if (key == "friction_coef") iss >> sim.fricition_coef;
        else if (key == "stiffness_coef") iss >> sim.stiffness_coef;
        else if (key == "tol") iss >> sim.tol;
        else if (key == "dhat") iss >> sim.dhat;
        else if (key == "kappa") iss >> sim.kappa;
        else if (key == "num_steps") iss >> sim.num_steps;
        else if (key == "ground_x") iss >> sim.ground_x;
        else if (key == "ground_y") iss >> sim.ground_y;
        else if (key == "ground_z") iss >> sim.ground_z;
        else if (key == "ground_nx") iss >> sim.ground_nx;
        else if (key == "ground_ny") iss >> sim.ground_ny;
        else if (key == "ground_nz") iss >> sim.ground_nz;
    }

    static void parse_mesh_param(const std::string& key, std::istringstream& iss, MeshConfig<T>& mesh) {
        if (key == "filepath") {
            std::string path;
            iss >> std::ws;  // Skip whitespace
            std::getline(iss, path);
            // Remove quotes if present
            if (!path.empty() && path[0] == '"') path = path.substr(1);
            if (!path.empty() && path[path.size()-1] == '"') path = path.substr(0, path.size()-1);
            mesh.filepath = path;
        }
        else if (key == "density") iss >> mesh.density;
        else if (key == "young_mod") iss >> mesh.young_mod;
        else if (key == "poisson_ratio") iss >> mesh.poisson_ratio;
        else if (key == "bending_stiff") iss >> mesh.bending_stiff;
        else if (key == "num_instances_x") iss >> mesh.num_instances_x;
        else if (key == "num_instances_y") iss >> mesh.num_instances_y;
        else if (key == "num_instances_z") iss >> mesh.num_instances_z;
        else if (key == "spacing_x") iss >> mesh.spacing_x;
        else if (key == "spacing_y") iss >> mesh.spacing_y;
        else if (key == "spacing_z") iss >> mesh.spacing_z;
        else if (key == "offset_x") iss >> mesh.offset_x;
        else if (key == "offset_y") iss >> mesh.offset_y;
        else if (key == "offset_z") iss >> mesh.offset_z;
        else if (key == "scale") iss >> mesh.scale;
        else if (key == "velocity_x") iss >> mesh.velocity_x;
        else if (key == "velocity_y") iss >> mesh.velocity_y;
        else if (key == "velocity_z") iss >> mesh.velocity_z;
        else if (key == "fixed_x") { std::string val; iss >> val; mesh.fixed_x = (val == "true" || val == "1"); }
        else if (key == "fixed_y") { std::string val; iss >> val; mesh.fixed_y = (val == "true" || val == "1"); }
        else if (key == "fixed_z") { std::string val; iss >> val; mesh.fixed_z = (val == "true" || val == "1"); }
    }
};
