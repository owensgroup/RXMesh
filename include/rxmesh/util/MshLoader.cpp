// based on MSH reader from PyMesh

// Copyright (c) 2015 Qingnan Zhou <qzhou@adobe.com>
// Copyright (C) 2020 Vladimir Fonov <vladimir.fonov@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "rxmesh/util/MshLoader.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace rxmesh {
namespace {
void eat_text_white_space(std::ifstream& fin)
{
    char next = fin.peek();
    while (next == '\n' || next == ' ' || next == '\t' || next == '\r') {
        fin.get();
        next = fin.peek();
    }
}

void consume_binary_line_end(std::ifstream& fin)
{
    int next = fin.get();
    while (next == ' ' || next == '\t') {
        next = fin.get();
    }
    if (next == '\r') {
        next = fin.get();
    }
    if (next != '\n') {
        throw std::runtime_error("Expected a line ending before binary data");
    }
}

template <typename T>
T read_binary(std::ifstream& fin)
{
    T value;
    if (!fin.read(reinterpret_cast<char*>(&value), sizeof(T))) {
        throw std::runtime_error("Unexpected end of binary .msh data");
    }
    return value;
}
}  // namespace

MshLoader::MshLoader(const std::string& filename)
{
    std::ifstream fin(filename, std::ios::in | std::ios::binary);

    if (!fin.is_open()) {
        std::stringstream err_msg;
        err_msg << "failed to open file \"" << filename << "\"";
        throw std::ios_base::failure(err_msg.str());
    }
    // Parse header
    std::string buf;
    double      version;
    int         type;
    fin >> buf;
    if (buf != "$MeshFormat") {
        throw std::runtime_error("Unexpected .msh format");
    }

    fin >> version >> type >> m_data_size;
    if (type != 0 && type != 1) {
        throw std::runtime_error("Unsupported .msh encoding");
    }
    m_binary = (type == 1);
    if (version != 2.0 && version != 2.1 && version != 2.2) {
        // probably unsupported version
        std::stringstream err_msg;
        err_msg << "Error: Unsupported file version:" << version << std::endl;
        throw std::runtime_error(err_msg.str());
    }
    // Some sanity check.
    if (m_data_size != 8) {
        std::stringstream err_msg;
        err_msg << "Error: data size must be 8 bytes." << std::endl;
        throw std::runtime_error(err_msg.str());
    }
    if (sizeof(int) != 4) {
        std::stringstream err_msg;
        err_msg << "Error: code must be compiled with int size 4 bytes."
                << std::endl;
        throw std::runtime_error(err_msg.str());
    }

    // Read in extra info from binary header.
    if (m_binary) {
        consume_binary_line_end(fin);
        const int one = read_binary<int>(fin);
        if (one != 1) {
            std::stringstream err_msg;
            err_msg << "Binary msh file " << filename
                    << " is saved with different endianness than this machine."
                    << std::endl;
            throw std::runtime_error(err_msg.str());
        }
    }

    fin >> buf;
    if (buf != "$EndMeshFormat") {
        std::stringstream err_msg;
        err_msg << "Unexpected contents in the file header." << std::endl;
        throw std::runtime_error(err_msg.str());
    }

    while (!fin.eof()) {
        buf.clear();
        fin >> buf;
        if (buf == "$Nodes") {
            parse_nodes(fin);
            fin >> buf;
            if (buf != "$EndNodes") {
                throw std::runtime_error("Unexpected tag");
            }
        } else if (buf == "$Elements") {
            parse_elements(fin);
            fin >> buf;
            if (buf != "$EndElements") {
                throw std::runtime_error("Unexpected tag");
            }
        } else if (buf == "$NodeData") {
            parse_node_field(fin);
            fin >> buf;
            if (buf != "$EndNodeData") {
                throw std::runtime_error("Unexpected tag");
            }
        } else if (buf == "$ElementData") {
            parse_element_field(fin);
            fin >> buf;
            if (buf != "$EndElementData") {
                throw std::runtime_error("Unexpected tag");
            }
        } else if (fin.eof()) {
            break;
        } else {
            parse_unknown_field(fin, buf);
        }
    }
    fin.close();
}

void MshLoader::parse_nodes(std::ifstream& fin)
{
    size_t num_nodes;
    if (!(fin >> num_nodes)) {
        throw std::runtime_error("Invalid .msh node count");
    }
    if (num_nodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Too many .msh nodes");
    }
    m_nodes.reserve(num_nodes * 3);

    auto add_node =
        [&](const int tag, const double x, const double y, const double z) {
            if (tag <= 0 ||
                !m_node_map.emplace(tag, static_cast<int>(m_node_map.size()))
                     .second) {
                throw std::runtime_error("Invalid or duplicate .msh node tag");
            }
            m_nodes.push_back(static_cast<Float>(x));
            m_nodes.push_back(static_cast<Float>(y));
            m_nodes.push_back(static_cast<Float>(z));
        };

    if (m_binary) {
        consume_binary_line_end(fin);
        for (size_t i = 0; i < num_nodes; i++) {
            const int    tag = read_binary<int>(fin);
            const double x   = read_binary<double>(fin);
            const double y   = read_binary<double>(fin);
            const double z   = read_binary<double>(fin);
            add_node(tag, x, y, z);
        }
    } else {
        for (size_t i = 0; i < num_nodes; i++) {
            int    tag;
            double x, y, z;
            if (!(fin >> tag >> x >> y >> z)) {
                throw std::runtime_error("Invalid .msh node");
            }
            add_node(tag, x, y, z);
        }
    }
}

void MshLoader::parse_elements(std::ifstream& fin)
{
    m_elements_tags.resize(2);  // hardcoded to have 2 tags
    size_t num_elements;
    if (!(fin >> num_elements)) {
        throw std::runtime_error("Invalid .msh element count");
    }
    if (num_elements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Too many .msh elements");
    }

    auto add_element = [&](const int elem_id, const int elem_type) {
        if (m_elements.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Too much .msh element connectivity");
        }
        if (elem_id <= 0 ||
            !m_element_map
                 .emplace(elem_id, static_cast<int>(m_element_map.size()))
                 .second) {
            throw std::runtime_error("Invalid or duplicate .msh element tag");
        }
        m_elements_ids.push_back(elem_id - 1);
        m_elements_types.push_back(elem_type);
        m_elements_lengths.push_back(num_nodes_per_elem_type(elem_type));
        m_elements_nodes_idx.push_back(static_cast<int>(m_elements.size()));
    };

    auto add_node = [&](const int node_tag) {
        const auto iter = m_node_map.find(node_tag);
        if (iter == m_node_map.end()) {
            throw std::runtime_error("Element references a missing node tag");
        }
        m_elements.push_back(iter->second);
    };

    if (m_binary) {
        consume_binary_line_end(fin);
        size_t elem_read = 0;
        while (elem_read < num_elements) {
            // Parse element header.
            const int elem_type = read_binary<int>(fin);
            const int num_elems = read_binary<int>(fin);
            const int num_tags  = read_binary<int>(fin);
            if (num_elems <= 0 || num_tags < 0 ||
                static_cast<size_t>(num_elems) > num_elements - elem_read) {
                throw std::runtime_error("Invalid binary .msh element block");
            }
            const int nodes_per_element = num_nodes_per_elem_type(elem_type);

            // store node info
            for (int i = 0; i < num_elems; i++) {
                add_element(read_binary<int>(fin), elem_type);

                // read first two tags
                for (int j = 0; j < num_tags; j++) {
                    const int tag = read_binary<int>(fin);
                    if (j < 2)
                        m_elements_tags[j].push_back(tag);
                }

                for (int j = num_tags; j < 2; j++)
                    m_elements_tags[j].push_back(
                        -1);  // fill up tags if less then 2

                // Element values.
                for (int j = 0; j < nodes_per_element; j++) {
                    add_node(read_binary<int>(fin));
                }
            }
            elem_read += num_elems;
        }
    } else {
        for (size_t i = 0; i < num_elements; i++) {
            // Parse per element header
            int elem_num, elem_type, num_tags;
            if (!(fin >> elem_num >> elem_type >> num_tags) || num_tags < 0) {
                throw std::runtime_error("Invalid .msh element");
            }
            add_element(elem_num, elem_type);

            // read tags.
            for (int j = 0; j < num_tags; j++) {
                int tag;
                if (!(fin >> tag)) {
                    throw std::runtime_error("Invalid .msh element tag");
                }
                if (j < 2)
                    m_elements_tags[j].push_back(tag);
            }
            for (int j = num_tags; j < 2; j++)
                m_elements_tags[j].push_back(
                    -1);  // fill up tags if less then 2

            // Parse node idx.
            for (int j = 0; j < m_elements_lengths.back(); j++) {
                int idx;
                if (!(fin >> idx)) {
                    throw std::runtime_error("Invalid .msh element node tag");
                }
                add_node(idx);
            }
        }
    }
    // debug
    assert(m_elements_types.size() == m_elements_ids.size());
    assert(m_elements_tags[0].size() == m_elements_ids.size());
    assert(m_elements_tags[1].size() == m_elements_ids.size());
    assert(m_elements_lengths.size() == m_elements_ids.size());
}

void MshLoader::parse_node_field(std::ifstream& fin)
{
    size_t num_string_tags;
    size_t num_real_tags;
    size_t num_int_tags;

    fin >> num_string_tags;
    std::vector<std::string> str_tags(num_string_tags);

    for (size_t i = 0; i < num_string_tags; i++) {
        eat_text_white_space(fin);
        if (fin.peek() == '\"') {
            // Handle field name between quotes.
            fin.get();  // remove the quote at the beginning.
            std::getline(fin, str_tags[i], '"');
        } else {
            fin >> str_tags[i];
        }
    }

    fin >> num_real_tags;
    std::vector<Float> real_tags(num_real_tags);
    for (size_t i = 0; i < num_real_tags; i++)
        fin >> real_tags[i];

    fin >> num_int_tags;
    std::vector<int> int_tags(num_int_tags);
    for (size_t i = 0; i < num_int_tags; i++)
        fin >> int_tags[i];

    if (num_string_tags <= 0 || num_int_tags <= 2) {
        throw std::runtime_error("Unexpected number of field tags");
    }
    std::string fieldname      = str_tags[0];
    int         num_components = int_tags[1];
    int         num_entries    = int_tags[2];

    if (num_components <= 0 || num_entries < 0) {
        throw std::runtime_error("Invalid node field size");
    }
    std::vector<Float> field(m_node_map.size() * num_components);

    if (m_binary) {
        consume_binary_line_end(fin);
    }
    for (int i = 0; i < num_entries; i++) {
        int node_tag;
        if (m_binary) {
            node_tag = read_binary<int>(fin);
        } else if (!(fin >> node_tag)) {
            throw std::runtime_error("Invalid node field entry");
        }
        const auto node = m_node_map.find(node_tag);
        if (node == m_node_map.end()) {
            throw std::runtime_error("Node field references a missing node");
        }
        for (int j = 0; j < num_components; j++) {
            double value;
            if (m_binary) {
                value = read_binary<double>(fin);
            } else if (!(fin >> value)) {
                throw std::runtime_error("Invalid node field value");
            }
            field[node->second * num_components + j] =
                static_cast<Float>(value);
        }
    }

    m_node_fields_names.push_back(fieldname);
    m_node_fields.push_back(field);
    m_node_fields_components.push_back(num_components);
}

void MshLoader::parse_element_field(std::ifstream& fin)
{
    size_t num_string_tags;
    size_t num_real_tags;
    size_t num_int_tags;

    fin >> num_string_tags;
    std::vector<std::string> str_tags(num_string_tags);
    for (size_t i = 0; i < num_string_tags; i++) {
        eat_text_white_space(fin);
        if (fin.peek() == '\"') {
            // Handle field name between quoates.
            fin.get();  // remove the quote at the beginning.
            std::getline(fin, str_tags[i], '"');
        } else {
            fin >> str_tags[i];
        }
    }

    fin >> num_real_tags;
    std::vector<Float> real_tags(num_real_tags);
    for (size_t i = 0; i < num_real_tags; i++)
        fin >> real_tags[i];

    fin >> num_int_tags;
    std::vector<int> int_tags(num_int_tags);
    for (size_t i = 0; i < num_int_tags; i++)
        fin >> int_tags[i];

    if (num_string_tags <= 0 || num_int_tags <= 2) {
        throw std::runtime_error("Invalid file format");
    }
    std::string fieldname      = str_tags[0];
    int         num_components = int_tags[1];
    int         num_entries    = int_tags[2];
    if (num_components <= 0 || num_entries < 0) {
        throw std::runtime_error("Invalid element field size");
    }
    std::vector<Float> field(m_element_map.size() * num_components);

    if (m_binary) {
        consume_binary_line_end(fin);
    }
    for (int i = 0; i < num_entries; i++) {
        int elem_tag;
        if (m_binary) {
            elem_tag = read_binary<int>(fin);
        } else if (!(fin >> elem_tag)) {
            throw std::runtime_error("Invalid element field entry");
        }
        const auto element = m_element_map.find(elem_tag);
        if (element == m_element_map.end()) {
            throw std::runtime_error(
                "Element field references a missing element");
        }
        for (int j = 0; j < num_components; j++) {
            double value;
            if (m_binary) {
                value = read_binary<double>(fin);
            } else if (!(fin >> value)) {
                throw std::runtime_error("Invalid element field value");
            }
            field[element->second * num_components + j] =
                static_cast<Float>(value);
        }
    }
    m_element_fields_names.push_back(fieldname);
    m_element_fields.push_back(field);
    m_element_fields_components.push_back(num_components);
}

void MshLoader::parse_unknown_field(std::ifstream&     fin,
                                    const std::string& fieldname)
{
    std::cerr << "Warning: \"" << fieldname << "\" not supported yet.  Ignored."
              << std::endl;
    std::string endmark = fieldname.substr(0, 1) + "End" +
                          fieldname.substr(1, fieldname.size() - 1);

    std::string buf("");
    while (buf != endmark && !fin.eof()) {
        fin >> buf;
    }
}

int MshLoader::num_nodes_per_elem_type(int elem_type)
{
    int nodes_per_element = 0;
    switch (elem_type) {
        case ELEMENT_LINE:  // 2-node line
            nodes_per_element = 2;
            break;
        case ELEMENT_TRI:
            nodes_per_element = 3;  // 3-node triangle
            break;
        case ELEMENT_QUAD:
            nodes_per_element = 4;  // 5-node quad
            break;
        case ELEMENT_TET:
            nodes_per_element = 4;  // 4-node tetrahedra
            break;
        case ELEMENT_HEX:  // 8-node hexahedron
            nodes_per_element = 8;
            break;
        case ELEMENT_PRISM:  // 6-node prism
            nodes_per_element = 6;
            break;
        case ELEMENT_PYRAMID:  // 5-node pyramid
            nodes_per_element = 5;
            break;
        case ELEMENT_LINE_2ND_ORDER:
            nodes_per_element = 3;
            break;
        case ELEMENT_TRI_2ND_ORDER:
            nodes_per_element = 6;
            break;
        case ELEMENT_QUAD_2ND_ORDER:
            nodes_per_element = 9;
            break;
        case ELEMENT_TET_2ND_ORDER:
            nodes_per_element = 10;
            break;
        case ELEMENT_HEX_2ND_ORDER:
            nodes_per_element = 27;
            break;
        case ELEMENT_PRISM_2ND_ORDER:
            nodes_per_element = 18;
            break;
        case ELEMENT_PYRAMID_2ND_ORDER:
            nodes_per_element = 14;
            break;
        case ELEMENT_POINT:  // 1-node point
            nodes_per_element = 1;
            break;
        default:
            std::stringstream err_msg;
            err_msg << "Element type (" << elem_type
                    << ") is not supported yet." << std::endl;
            throw std::runtime_error(err_msg.str());
    }
    return nodes_per_element;
}


bool MshLoader::is_element_map_identity() const
{
    for (int i = 0; i < m_elements_ids.size(); i++) {
        int id = m_elements_ids[i];
        if (id != i)
            return false;
    }
    return true;
}


void MshLoader::index_structures(int tag_column)
{
    // cleanup
    m_structure_index.clear();
    m_structures.clear();
    m_structure_length.clear();

    // index structure tags
    for (auto i = 0; i != m_elements_tags[tag_column].size(); ++i) {
        m_structure_index.insert(std::pair<msh_struct, int>(
            msh_struct(m_elements_tags[tag_column][i], m_elements_types[i]),
            i));
    }

    // identify unique structures
    std::vector<StructIndex::value_type> _unique_structs;
    std::unique_copy(
        std::begin(m_structure_index),
        std::end(m_structure_index),
        std::back_inserter(_unique_structs),
        [](const StructIndex::value_type& c1,
           const StructIndex::value_type& c2) { return c1.first == c2.first; });

    std::for_each(_unique_structs.begin(),
                  _unique_structs.end(),
                  [this](const StructIndex::value_type& n) {
                      this->m_structures.push_back(n.first);
                  });

    for (auto t = m_structures.begin(); t != m_structures.end(); ++t) {
        // identify all elements corresponding to this tag
        auto structure_range = m_structure_index.equal_range(*t);
        int  cnt             = 0;

        for (auto i = structure_range.first; i != structure_range.second; i++)
            cnt++;

        m_structure_length.insert(std::pair<msh_struct, int>(*t, cnt));
    }
}

MeshKind load_msh(const std::string&                    filename,
                  std::vector<std::vector<rx_coord_t>>& vertices,
                  std::vector<std::vector<uint32_t>>&   simplices,
                  bool                                  append)
{
    const MshLoader loader(filename);
    const auto&     nodes    = loader.get_nodes();
    const auto&     elements = loader.get_elements();
    const auto&     starts   = loader.get_elements_nodes_idx();
    const auto&     types    = loader.get_elements_types();
    const auto&     lengths  = loader.get_elements_lengths();

    auto dimension = [](const int type) {
        switch (type) {
            case MshLoader::ELEMENT_POINT:
                return 0;
            case MshLoader::ELEMENT_LINE:
            case MshLoader::ELEMENT_LINE_2ND_ORDER:
                return 1;
            case MshLoader::ELEMENT_TRI:
            case MshLoader::ELEMENT_QUAD:
            case MshLoader::ELEMENT_TRI_2ND_ORDER:
            case MshLoader::ELEMENT_QUAD_2ND_ORDER:
                return 2;
            default:
                return 3;
        }
    };

    int top_dimension = 0;
    for (const int type : types) {
        top_dimension = std::max(top_dimension, dimension(type));
    }

    MeshKind kind;
    int      selected_type;
    if (top_dimension == 2) {
        kind          = MeshKind::Triangle;
        selected_type = MshLoader::ELEMENT_TRI;
    } else if (top_dimension == 3) {
        kind          = MeshKind::Tet;
        selected_type = MshLoader::ELEMENT_TET;
    } else {
        throw std::runtime_error("The .msh file has no triangles or tets");
    }

    for (const int type : types) {
        if (dimension(type) == top_dimension && type != selected_type) {
            throw std::runtime_error(
                "Unsupported top-dimensional .msh element type");
        }
    }

    const size_t num_nodes = nodes.size() / 3;
    if (nodes.size() % 3 != 0 ||
        num_nodes > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Invalid .msh node storage");
    }

    std::vector<bool> used(num_nodes, false);
    size_t            num_simplices     = 0;
    size_t            num_used_vertices = 0;
    for (size_t i = 0; i < types.size(); i++) {
        if (types[i] != selected_type) {
            continue;
        }
        const size_t start = static_cast<size_t>(starts[i]);
        if (starts[i] < 0 || lengths[i] != top_dimension + 1 ||
            start > elements.size() ||
            static_cast<size_t>(lengths[i]) > elements.size() - start) {
            throw std::runtime_error("Invalid .msh element storage");
        }
        for (int j = 0; j < lengths[i]; j++) {
            const int vertex = elements[start + j];
            if (vertex < 0 || static_cast<size_t>(vertex) >= num_nodes) {
                throw std::runtime_error("Invalid .msh vertex index");
            }
            if (!used[vertex]) {
                used[vertex] = true;
                num_used_vertices++;
            }
        }
        num_simplices++;
    }

    const uint32_t        invalid = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> old_to_new(num_nodes, invalid);
    const size_t          vertex_offset = append ? vertices.size() : 0;
    const size_t          max_vertices =
        static_cast<size_t>(std::numeric_limits<uint32_t>::max());
    if (num_used_vertices > max_vertices ||
        vertex_offset > max_vertices - num_used_vertices) {
        throw std::runtime_error("Too many combined mesh vertices");
    }

    if (!append) {
        vertices.clear();
        simplices.clear();
    }
    vertices.reserve(vertex_offset + num_used_vertices);
    for (size_t i = 0; i < num_nodes; i++) {
        if (used[i]) {
            old_to_new[i] = static_cast<uint32_t>(vertices.size());
            vertices.push_back({static_cast<rx_coord_t>(nodes[3 * i]),
                                static_cast<rx_coord_t>(nodes[3 * i + 1]),
                                static_cast<rx_coord_t>(nodes[3 * i + 2])});
        }
    }

    simplices.reserve(simplices.size() + num_simplices);
    for (size_t i = 0; i < types.size(); i++) {
        if (types[i] != selected_type) {
            continue;
        }
        const size_t          start = static_cast<size_t>(starts[i]);
        std::vector<uint32_t> simplex;
        simplex.reserve(lengths[i]);
        for (int j = 0; j < lengths[i]; j++) {
            simplex.push_back(old_to_new[elements[start + j]]);
        }
        simplices.push_back(std::move(simplex));
    }
    return kind;
}
}  // namespace rxmesh
