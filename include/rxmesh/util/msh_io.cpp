#include "rxmesh/util/msh_io.h"

#include <mshio/mshio.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rxmesh/util/log.h"

namespace rxmesh {

namespace {

constexpr int TRIANGLE_ELEMENT = 2;
constexpr int TET_ELEMENT      = 4;

void msh_error(const char*        operation,
               const std::string& filename,
               const std::string& message)
{
    RXMESH_ERROR("{}() failed for {}: {}", operation, filename, message);
    exit(EXIT_FAILURE);
}

}  // namespace

MeshKind load_msh(const std::string&                    filename,
                  std::vector<std::vector<rx_coord_t>>& vertices,
                  std::vector<std::vector<uint32_t>>&   simplices,
                  bool                                  append)
{
    RXMESH_INFO("Reading {}", filename);

    try {
        const mshio::MshSpec spec = mshio::load_msh(filename);

        int mesh_dimension = -1;
        for (const mshio::ElementBlock& block : spec.elements.entity_blocks) {
            if (block.num_elements_in_block != 0) {
                mesh_dimension = std::max(
                    mesh_dimension, mshio::get_element_dim(block.element_type));
            }
        }

        MeshKind    kind;
        int         element_type;
        std::size_t vertices_per_simplex;

        if (mesh_dimension == 2) {
            kind                 = MeshKind::Triangle;
            element_type         = TRIANGLE_ELEMENT;
            vertices_per_simplex = 3;
        } else if (mesh_dimension == 3) {
            kind                 = MeshKind::Tet;
            element_type         = TET_ELEMENT;
            vertices_per_simplex = 4;
        } else {
            msh_error("load_msh",
                      filename,
                      "the highest element dimension must be two or three");
        }

        std::vector<std::vector<std::size_t>> simplex_node_tags;
        std::unordered_set<std::size_t>       used_node_tags;
        simplex_node_tags.reserve(spec.elements.num_elements);

        for (const mshio::ElementBlock& block : spec.elements.entity_blocks) {
            if (block.num_elements_in_block == 0 ||
                mshio::get_element_dim(block.element_type) != mesh_dimension) {
                continue;
            }

            if (block.element_type != element_type) {
                msh_error("load_msh",
                          filename,
                          "unsupported or mixed top-dimensional element types");
            }

            const std::size_t entries_per_element = vertices_per_simplex + 1;
            if (block.data.size() !=
                block.num_elements_in_block * entries_per_element) {
                msh_error("load_msh",
                          filename,
                          "inconsistent top-dimensional element data");
            }

            for (std::size_t e = 0; e < block.num_elements_in_block; ++e) {
                const std::size_t offset = e * entries_per_element + 1;
                simplex_node_tags.emplace_back(vertices_per_simplex);

                for (std::size_t v = 0; v < vertices_per_simplex; ++v) {
                    const std::size_t tag       = block.data[offset + v];
                    simplex_node_tags.back()[v] = tag;
                    used_node_tags.insert(tag);
                }
            }
        }

        if (simplex_node_tags.empty()) {
            msh_error("load_msh",
                      filename,
                      "the file contains no supported top-dimensional cells");
        }

        std::vector<std::vector<rx_coord_t>>         new_vertices;
        std::unordered_map<std::size_t, std::size_t> node_tag_to_local;
        new_vertices.reserve(used_node_tags.size());
        node_tag_to_local.reserve(used_node_tags.size());

        for (const mshio::NodeBlock& block : spec.nodes.entity_blocks) {
            if (block.parametric != 0 && block.entity_dim < 0) {
                msh_error("load_msh",
                          filename,
                          "invalid parametric node block dimension");
            }

            const std::size_t entries_per_node =
                3 + (block.parametric == 0 ?
                         0 :
                         static_cast<std::size_t>(block.entity_dim));

            if (block.tags.size() != block.num_nodes_in_block ||
                block.data.size() !=
                    block.num_nodes_in_block * entries_per_node) {
                msh_error("load_msh", filename, "inconsistent node block data");
            }

            for (std::size_t n = 0; n < block.num_nodes_in_block; ++n) {
                const std::size_t tag = block.tags[n];
                if (used_node_tags.find(tag) == used_node_tags.end()) {
                    continue;
                }

                const auto result =
                    node_tag_to_local.emplace(tag, new_vertices.size());
                if (!result.second) {
                    msh_error("load_msh",
                              filename,
                              "a referenced node tag appears more than once");
                }

                const std::size_t offset = n * entries_per_node;
                new_vertices.push_back(
                    {static_cast<rx_coord_t>(block.data[offset]),
                     static_cast<rx_coord_t>(block.data[offset + 1]),
                     static_cast<rx_coord_t>(block.data[offset + 2])});
            }
        }

        if (node_tag_to_local.size() != used_node_tags.size()) {
            msh_error("load_msh",
                      filename,
                      "an element references a missing node tag");
        }

        const std::size_t vertex_offset = append ? vertices.size() : 0;
        std::vector<std::vector<uint32_t>> new_simplices(
            simplex_node_tags.size(),
            std::vector<uint32_t>(vertices_per_simplex));

        for (std::size_t s = 0; s < simplex_node_tags.size(); ++s) {
            for (std::size_t v = 0; v < vertices_per_simplex; ++v) {
                const std::size_t index =
                    vertex_offset +
                    node_tag_to_local.at(simplex_node_tags[s][v]);

                if (index > std::numeric_limits<uint32_t>::max()) {
                    msh_error("load_msh",
                              filename,
                              "the compact vertex index exceeds uint32_t");
                }
                new_simplices[s][v] = static_cast<uint32_t>(index);
            }
        }

        if (append) {
            vertices.reserve(vertices.size() + new_vertices.size());
            simplices.reserve(simplices.size() + new_simplices.size());
            vertices.insert(vertices.end(),
                            std::make_move_iterator(new_vertices.begin()),
                            std::make_move_iterator(new_vertices.end()));
            simplices.insert(simplices.end(),
                             std::make_move_iterator(new_simplices.begin()),
                             std::make_move_iterator(new_simplices.end()));
        } else {
            vertices  = std::move(new_vertices);
            simplices = std::move(new_simplices);
        }

        RXMESH_INFO("load_msh() #vertices= {} ", vertices.size());
        if (kind == MeshKind::Triangle) {
            RXMESH_INFO("load_msh() #faces= {} ", simplices.size());
        } else {
            RXMESH_INFO("load_msh() #tets= {} ", simplices.size());
        }

        return kind;
    } catch (const std::exception& e) {
        msh_error("load_msh", filename, e.what());
    }
}

void save_msh(const std::string&                          filename,
              const std::vector<std::vector<rx_coord_t>>& vertices,
              const std::vector<std::vector<uint32_t>>&   simplices,
              MeshKind                                    kind,
              bool                                        binary)
{
    RXMESH_INFO("Writing {}", filename);

    try {
        if (vertices.empty() || simplices.empty()) {
            msh_error("save_msh",
                      filename,
                      "the mesh must contain vertices and cells");
        }

        const int dimension = kind == MeshKind::Tet ? 3 : 2;
        const int element_type =
            kind == MeshKind::Tet ? TET_ELEMENT : TRIANGLE_ELEMENT;
        const std::size_t vertices_per_simplex = kind == MeshKind::Tet ? 4 : 3;

        mshio::MshSpec spec;
        spec.mesh_format.version   = "2.2";
        spec.mesh_format.file_type = binary ? 1 : 0;
        spec.mesh_format.data_size = sizeof(double);

        mshio::NodeBlock node_block;
        node_block.entity_dim         = dimension;
        node_block.entity_tag         = 1;
        node_block.parametric         = 0;
        node_block.num_nodes_in_block = vertices.size();
        node_block.tags.reserve(vertices.size());
        node_block.data.reserve(3 * vertices.size());

        for (std::size_t v = 0; v < vertices.size(); ++v) {
            if (vertices[v].size() < 3) {
                msh_error("save_msh",
                          filename,
                          "every vertex must have three coordinates");
            }

            node_block.tags.push_back(v + 1);
            node_block.data.push_back(vertices[v][0]);
            node_block.data.push_back(vertices[v][1]);
            node_block.data.push_back(vertices[v][2]);
        }

        spec.nodes.num_entity_blocks = 1;
        spec.nodes.num_nodes         = vertices.size();
        spec.nodes.min_node_tag      = 1;
        spec.nodes.max_node_tag      = vertices.size();
        spec.nodes.entity_blocks.push_back(std::move(node_block));

        mshio::ElementBlock element_block;
        element_block.entity_dim            = dimension;
        element_block.entity_tag            = 1;
        element_block.element_type          = element_type;
        element_block.num_elements_in_block = simplices.size();
        element_block.data.reserve((vertices_per_simplex + 1) *
                                   simplices.size());

        for (std::size_t s = 0; s < simplices.size(); ++s) {
            if (simplices[s].size() != vertices_per_simplex) {
                msh_error("save_msh",
                          filename,
                          "a cell has the wrong number of vertices");
            }

            element_block.data.push_back(s + 1);
            for (std::size_t v = 0; v < vertices_per_simplex; ++v) {
                if (simplices[s][v] >= vertices.size()) {
                    msh_error(
                        "save_msh", filename, "a cell index is out of bounds");
                }
                element_block.data.push_back(simplices[s][v] + 1);
            }
        }

        spec.elements.num_entity_blocks = 1;
        spec.elements.num_elements      = simplices.size();
        spec.elements.min_element_tag   = 1;
        spec.elements.max_element_tag   = simplices.size();
        spec.elements.entity_blocks.push_back(std::move(element_block));

        mshio::validate_spec(spec);
        mshio::save_msh(filename, spec);

        RXMESH_INFO("save_msh() #vertices= {} ", vertices.size());
        if (kind == MeshKind::Triangle) {
            RXMESH_INFO("save_msh() #faces= {} ", simplices.size());
        } else {
            RXMESH_INFO("save_msh() #tets= {} ", simplices.size());
        }
    } catch (const std::exception& e) {
        msh_error("save_msh", filename, e.what());
    }
}

}  // namespace rxmesh
