#pragma once

#include <assert.h>
#include <iomanip>
#include <vector>
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_context.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/util.h"


class RXMeshTest
{
   private:
    bool                               m_quite;
    std::vector<std::vector<uint32_t>> m_h_FE;

   public:
    RXMeshTest(const RXMeshTest&) = delete;
    RXMeshTest(const rxmesh::RXMeshStatic&               rxmesh,
               const std::vector<std::vector<uint32_t>>& fv,
               bool                                      quite = true)
        : m_quite(quite)
    {
        assert(rxmesh.m_edges_map.size() != 0);

        for (uint32_t f = 0; f < rxmesh.m_num_faces; ++f) {
            uint32_t i = f;

            std::vector<uint32_t> fe(3);

            for (uint32_t j = 0; j < 3; ++j) {

                uint32_t v0 = fv[i][j];
                uint32_t v1 = fv[i][(j + 1) % 3];

                std::pair<uint32_t, uint32_t> my_edge =
                    rxmesh::detail::edge_key(v0, v1);
                uint32_t edge_id = rxmesh.get_edge_id(my_edge);
                fe[j]            = edge_id;
            }
            m_h_FE.push_back(fe);
        }
    }

    bool run_query_verifier(
        const rxmesh::RXMeshStatic&               rxmesh,
        const std::vector<std::vector<uint32_t>>& fv,
        const rxmesh::Op                          op,
        const rxmesh::RXMeshAttribute<uint32_t>&  input_container,
        const rxmesh::RXMeshAttribute<uint32_t>&  output_container)
    {

        // run test on specific query operation on an instance of rxmesh. this
        // does not account for patching so works only on big matrices data-
        // structure

        switch (op) {
            case rxmesh::Op::VV:
                return test_VV(rxmesh, input_container, output_container);
                break;

            case rxmesh::Op::VE:
                return test_VE(rxmesh, input_container, output_container);
                break;

            case rxmesh::Op::VF:
                return test_VF(rxmesh, fv, input_container, output_container);
                break;

            case rxmesh::Op::FV:
                return test_FV(rxmesh, fv, input_container, output_container);
                break;

            case rxmesh::Op::FE:
                return test_FE(rxmesh, input_container, output_container);
                break;

            case rxmesh::Op::FF:
                return test_FF(rxmesh, input_container, output_container);
                break;

            case rxmesh::Op::EV:
                return test_EV(rxmesh, input_container, output_container);
                break;
            case rxmesh::Op::EF:
                return test_EF(rxmesh, input_container, output_container);
                break;

            default:
                RXMESH_ERROR("RXMeshTest::run_test() Op is not supported!!");
                break;
        }
        return false;
    }


    bool run_ltog_mapping_test(const rxmesh::RXMesh&                     rxmesh,
                               const std::vector<std::vector<uint32_t>>& fv)
    {
        // check if the mapping created for each patch is consistent
        // i.e., what you have in the local index space represents the global
        // space
        for (uint32_t p = 0; p < rxmesh.m_num_patches; ++p) {
            bool edges_ok(true), faces_ok(true);
            check_mapping(rxmesh, p, edges_ok, faces_ok);
            if (!edges_ok || !faces_ok) {
                return false;
            }
        }
        return true;
    }

    bool test_VVV(const rxmesh::RXMeshStatic&              rxmesh,
                  const rxmesh::RXMeshAttribute<uint32_t>& input_container,
                  const rxmesh::RXMeshAttribute<uint32_t>& output_container)
    {

        // construct VV
        std::vector<std::vector<uint32_t>> v_v(rxmesh.m_num_vertices,
                                               std::vector<uint32_t>(0));

        auto e_it  = rxmesh.m_edges_map.begin();
        auto e_end = rxmesh.m_edges_map.end();

        for (; e_it != e_end; e_it++) {
            std::pair<uint32_t, uint32_t> vertices = e_it->first;

            v_v[vertices.first].push_back(vertices.second);
            v_v[vertices.second].push_back(vertices.first);
        }

        // use VV to construct VVV
        std::vector<std::vector<uint32_t>> v_v_v = v_v;
        for (uint32_t v = 0; v < v_v_v.size(); ++v) {

            // loop over the v_v list of the vertex v
            for (uint32_t i = 0; i < v_v[v].size(); ++i) {

                // this is a vertex in the 1-ring (v_v) of v
                uint32_t n = v_v_v[v][i];

                // loop over the v_v list (1-ring) of n
                for (uint32_t j = 0; j < v_v[n].size(); ++j) {

                    // a candidate to be added to the 2-ring of v
                    uint32_t candid = v_v[n][j];

                    // but we need to check first if it is not duplicate and
                    // it is not v itself
                    if (candid != v &&
                        rxmesh::find_index(candid, v_v_v[v]) ==
                            std::numeric_limits<uint32_t>::max()) {

                        v_v_v[v].push_back(candid);
                    }
                }
            }
        }


        // two-way verification
        return verifier(rxmesh.get_patcher()->get_vertex_patch().data(),
                        v_v_v,
                        input_container,
                        output_container);
    }

    /**
     * @brief verify VV query
     */
    bool run_test(
        const rxmesh::RXMeshStatic&                                rxmesh,
        const rxmesh::RXMeshVertexAttribute<rxmesh::VertexHandle>& input,
        const rxmesh::RXMeshVertexAttribute<rxmesh::VertexHandle>& output)
    {
        std::vector<std::vector<uint32_t>> v_v(rxmesh.get_num_vertices(),
                                               std::vector<uint32_t>(0));

        auto e_it  = rxmesh.m_edges_map.begin();
        auto e_end = rxmesh.m_edges_map.end();

        for (; e_it != e_end; e_it++) {
            std::pair<uint32_t, uint32_t> vertices = e_it->first;

            v_v[vertices.first].push_back(vertices.second);
            v_v[vertices.second].push_back(vertices.first);
        }

        bool res = true;
        for (uint32_t p = 0; p < rxmesh.get_num_patches(); ++p) {
            for (uint32_t v = 0; v < rxmesh.m_h_num_owned_v[p]; ++v) {
                rxmesh::VertexHandle vh(p, v);
                if (input(vh) != vh) {
                    res = false;
                    break;
                }

                uint32_t v_global = rxmesh.m_h_patches_ltog_v[p][v];

                uint32_t num_vv = 0;
                for (uint32_t i = 0; i < output.get_num_attributes(); ++i) {
                    rxmesh::VertexHandle vvh = output(vh, i);
                    if (vvh.is_valid()) {
                        num_vv++;

                        // extract local id from vvh's unique id
                        uint64_t uid       = vvh.unique_id();
                        uint16_t lid       = uid & ((1 << 16) - 1);
                        uint32_t vv_global = rxmesh.m_h_patches_ltog_v[p][lid];

                        uint32_t id =
                            rxmesh::find_index(vv_global, v_v[v_global]);

                        if (id == std::numeric_limits<uint32_t>::max()) {
                            res = false;
                            break;
                        }
                    }
                }

                if (!res) {
                    break;
                }
                if (num_vv != v_v[v_global].size()) {
                    res = false;
                    break;
                }
            }
            if (!res) {
                break;
            }
        }

        return res;
    }

    /**
     * @brief verify VE query
     */
    bool run_test(
        const rxmesh::RXMeshStatic&                                rxmesh,
        const rxmesh::RXMeshVertexAttribute<rxmesh::VertexHandle>& input,
        const rxmesh::RXMeshVertexAttribute<rxmesh::EdgeHandle>&   output)
    {
        return false;
    }

    /**
     * @brief verify VF query
     */
    bool run_test(
        const rxmesh::RXMeshStatic&                                rxmesh,
        const rxmesh::RXMeshVertexAttribute<rxmesh::VertexHandle>& input,
        const rxmesh::RXMeshVertexAttribute<rxmesh::FaceHandle>&   output)
    {
        return false;
    }

    /**
     * @brief verify EV query
     */
    bool run_test(
        const rxmesh::RXMeshStatic&                              rxmesh,
        const rxmesh::RXMeshEdgeAttribute<rxmesh::EdgeHandle>&   input,
        const rxmesh::RXMeshEdgeAttribute<rxmesh::VertexHandle>& output)
    {
        return false;
    }

    /**
     * @brief verify EF query
     */
    bool run_test(const rxmesh::RXMeshStatic&                            rxmesh,
                  const rxmesh::RXMeshEdgeAttribute<rxmesh::EdgeHandle>& input,
                  const rxmesh::RXMeshEdgeAttribute<rxmesh::FaceHandle>& output)
    {
        return false;
    }

    /**
     * @brief verify FV query
     */
    bool run_test(
        const rxmesh::RXMeshStatic&                              rxmesh,
        const rxmesh::RXMeshFaceAttribute<rxmesh::FaceHandle>&   input,
        const rxmesh::RXMeshFaceAttribute<rxmesh::VertexHandle>& output)
    {
        return false;
    }

    /**
     * @brief verify FE query
     */
    bool run_test(const rxmesh::RXMeshStatic&                            rxmesh,
                  const rxmesh::RXMeshFaceAttribute<rxmesh::FaceHandle>& input,
                  const rxmesh::RXMeshFaceAttribute<rxmesh::EdgeHandle>& output)
    {
        return false;
    }

    /**
     * @brief verify FF query
     */
    bool run_test(const rxmesh::RXMeshStatic&                            rxmesh,
                  const rxmesh::RXMeshFaceAttribute<rxmesh::FaceHandle>& input,
                  const rxmesh::RXMeshFaceAttribute<rxmesh::FaceHandle>& output)
    {
        return false;
    }

    bool test_VV(const rxmesh::RXMeshStatic&              rxmesh,
                 const rxmesh::RXMeshAttribute<uint32_t>& input_container,
                 const rxmesh::RXMeshAttribute<uint32_t>& output_container)
    {

        // construct VV

        std::vector<std::vector<uint32_t>> v_v(rxmesh.m_num_vertices,
                                               std::vector<uint32_t>(0));

        auto e_it  = rxmesh.m_edges_map.begin();
        auto e_end = rxmesh.m_edges_map.end();

        for (; e_it != e_end; e_it++) {
            std::pair<uint32_t, uint32_t> vertices = e_it->first;

            v_v[vertices.first].push_back(vertices.second);
            v_v[vertices.second].push_back(vertices.first);
        }

        // two-way verification
        return verifier(rxmesh.get_patcher()->get_vertex_patch().data(),
                        v_v,
                        input_container,
                        output_container);
    }

    bool test_VE(const rxmesh::RXMeshStatic&              rxmesh,
                 const rxmesh::RXMeshAttribute<uint32_t>& input_container,
                 const rxmesh::RXMeshAttribute<uint32_t>& output_container)
    {

        // construct VE

        std::vector<std::vector<uint32_t>> v_e(rxmesh.m_num_vertices,
                                               std::vector<uint32_t>(0));

        auto e_it  = rxmesh.m_edges_map.begin();
        auto e_end = rxmesh.m_edges_map.end();

        for (; e_it != e_end; e_it++) {
            std::pair<uint32_t, uint32_t> vertices = e_it->first;
            uint32_t                      edge     = e_it->second;
            v_e[vertices.first].push_back(edge);
            v_e[vertices.second].push_back(edge);
        }

        // two-way verification
        return verifier(rxmesh.get_patcher()->get_vertex_patch().data(),
                        v_e,
                        input_container,
                        output_container);
    }

    bool test_VF(const rxmesh::RXMeshStatic&               rxmesh,
                 const std::vector<std::vector<uint32_t>>& fv,
                 const rxmesh::RXMeshAttribute<uint32_t>&  input_container,
                 const rxmesh::RXMeshAttribute<uint32_t>&  output_container)
    {

        // construct FV
        if (rxmesh.m_num_faces != fv.size()) {
            return false;
        }
        std::vector<std::vector<uint32_t>> v_f(rxmesh.m_num_vertices,
                                               std::vector<uint32_t>(0));

        for (uint32_t f = 0; f < fv.size(); f++) {
            for (uint32_t v = 0; v < 3; v++) {
                uint32_t vert = fv[f][v];
                v_f[vert].push_back(f);
            }
        }

        // two-way verification
        return verifier(rxmesh.get_patcher()->get_vertex_patch().data(),
                        v_f,
                        input_container,
                        output_container);
    }

    bool test_FV(const rxmesh::RXMeshStatic&               rxmesh,
                 const std::vector<std::vector<uint32_t>>& fv,
                 const rxmesh::RXMeshAttribute<uint32_t>&  input_container,
                 const rxmesh::RXMeshAttribute<uint32_t>&  output_container)
    {
        // two-way verification
        return verifier(rxmesh.get_patcher()->get_face_patch().data(),
                        fv,
                        input_container,
                        output_container);
    }

    bool test_FE(const rxmesh::RXMeshStatic&              rxmesh,
                 const rxmesh::RXMeshAttribute<uint32_t>& input_container,
                 const rxmesh::RXMeshAttribute<uint32_t>& output_container)
    {

        // construct FE

        std::vector<std::vector<uint32_t>> f_e(rxmesh.m_num_faces,
                                               std::vector<uint32_t>(0));

        for (uint32_t f = 0; f < rxmesh.m_num_faces; f++) {
            f_e[f].reserve(3);
        }

        for (uint32_t f = 0; f < rxmesh.m_num_faces; f++) {
            uint32_t e0 = m_h_FE[f][0];
            uint32_t e1 = m_h_FE[f][1];
            uint32_t e2 = m_h_FE[f][2];

            f_e[f].push_back(e0);
            f_e[f].push_back(e1);
            f_e[f].push_back(e2);
        }


        // two-way verification
        return verifier(rxmesh.get_patcher()->get_face_patch().data(),
                        f_e,
                        input_container,
                        output_container);
    }

    bool test_FF(const rxmesh::RXMeshStatic&              rxmesh,
                 const rxmesh::RXMeshAttribute<uint32_t>& input_container,
                 const rxmesh::RXMeshAttribute<uint32_t>& output_container)
    {

        // construct FF

        std::vector<std::vector<uint32_t>> f_f(rxmesh.m_num_faces,
                                               std::vector<uint32_t>(0));

        std::vector<std::vector<uint32_t>> e_f;
        for (uint32_t e = 0; e < rxmesh.m_num_edges; e++) {
            std::vector<uint32_t> eee;
            e_f.push_back(eee);
        }

        for (uint32_t f = 0; f < rxmesh.m_num_faces; f++) {
            // every face throw itself to one edge and then edge aggregates
            // the faces
            uint32_t e0 = m_h_FE[f][0];
            uint32_t e1 = m_h_FE[f][1];
            uint32_t e2 = m_h_FE[f][2];

            e_f[e0].push_back(f);
            e_f[e1].push_back(f);
            e_f[e2].push_back(f);
        }

        for (uint32_t e = 0; e < rxmesh.m_num_edges; e++) {
            for (uint32_t f = 0; f < e_f[e].size() - 1; f++) {
                for (uint32_t ff = f + 1; ff < e_f[e].size(); ff++) {
                    uint32_t f0 = e_f[e][f];
                    uint32_t f1 = e_f[e][ff];

                    f_f[f0].push_back(f1);
                    f_f[f1].push_back(f0);
                }
            }
        }

        // two-way verification
        return verifier(rxmesh.get_patcher()->get_face_patch().data(),
                        f_f,
                        input_container,
                        output_container);
    }

    bool test_EV(const rxmesh::RXMeshStatic&              rxmesh,
                 const rxmesh::RXMeshAttribute<uint32_t>& input_container,
                 const rxmesh::RXMeshAttribute<uint32_t>& output_container)
    {

        // construct EV

        std::vector<std::vector<uint32_t>> e_v(rxmesh.m_num_edges,
                                               std::vector<uint32_t>(2));

        auto e_it = rxmesh.m_edges_map.begin();
        while (e_it != rxmesh.m_edges_map.end()) {
            e_v[e_it->second][0] = (e_it->first).first;
            e_v[e_it->second][1] = (e_it->first).second;
            e_it++;
        }


        // two-way verification
        return verifier(rxmesh.get_patcher()->get_edge_patch().data(),
                        e_v,
                        input_container,
                        output_container);
    }

    bool test_EF(const rxmesh::RXMeshStatic&              rxmesh,
                 const rxmesh::RXMeshAttribute<uint32_t>& input_container,
                 const rxmesh::RXMeshAttribute<uint32_t>& output_container)
    {

        // construct EF

        std::vector<std::vector<uint32_t>> e_f(rxmesh.m_num_edges,
                                               std::vector<uint32_t>(0));

        for (uint32_t f = 0; f < rxmesh.m_num_faces; f++) {
            for (uint32_t e = 0; e < 3; e++) {
                uint32_t edge = m_h_FE[f][e];
                e_f[edge].push_back(f);
            }
        }

        // two-way verification
        return verifier(rxmesh.get_patcher()->get_edge_patch().data(),
                        e_f,
                        input_container,
                        output_container);
    }

    bool verifier(const uint32_t*                           element_patch,
                  const std::vector<std::vector<uint32_t>>& mesh_ele,
                  const rxmesh::RXMeshAttribute<uint32_t>&  input_container,
                  const rxmesh::RXMeshAttribute<uint32_t>&  output_container)
    {

        bool results = true;

        const uint32_t input_size = input_container.get_num_mesh_elements();

        assert(input_size == output_container.get_num_mesh_elements());

        for (uint32_t v = 0; v < input_size; v++) {

            const uint32_t src_ele = input_container(v);

            if (src_ele == INVALID32) {
                // means it is isolated element so don't bother
                continue;
            }
            // check for correctness (e.g, all edges in h_output are actually
            // edges incident to the vertex v)
            for (uint32_t i = 1; i <= output_container(v, 0); ++i) {

                uint32_t id = rxmesh::find_index(output_container(v, i),
                                                 mesh_ele[src_ele]);

                if (id == std::numeric_limits<uint32_t>::max()) {
                    if (!m_quite) {
                        RXMESH_ERROR(
                            "RXMeshTest::verifier() element {} is not incident "
                            "to {}",
                            output_container(v, i),
                            src_ele);
                    }
                    results = false;
                }
            }

            // check for completeness (e.g, that all edges incident to the
            // vertex v are actually returned in output_container)
            for (uint32_t i = 0; i < mesh_ele[src_ele].size(); i++) {
                uint32_t e     = mesh_ele[src_ele][i];
                bool     found = false;
                for (uint32_t j = 1; j <= output_container(v, 0); j++) {
                    if (output_container(v, j) == e) {
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    if (!m_quite) {
                        RXMESH_ERROR(
                            "RXMeshTest::verifier() element {} is not incident "
                            "to {}",
                            e,
                            src_ele);
                    }
                    results = false;
                }
            }
        }

        return results;
    }

    void check_mapping(const rxmesh::RXMesh& rxmesh,
                       const uint32_t        patch_id,
                       bool&                 is_edges_ok,
                       bool&                 is_faces_ok)
    {
        // check if the mapping is consistent i.e., going from local to
        // global gives the same results as from global to local

        // Number of edges and faces in this patch
        uint32_t num_p_edges = rxmesh.m_h_ad_size[patch_id].y >> 1;
        uint32_t num_p_faces = static_cast<uint32_t>(
            static_cast<float>(rxmesh.m_h_ad_size[patch_id].w) / 3.0f);

        assert(num_p_edges <= std::numeric_limits<uint16_t>::max());
        assert(num_p_faces <= std::numeric_limits<uint16_t>::max());

        is_edges_ok = check_mapping_edges(rxmesh, patch_id, num_p_edges);

        is_faces_ok = check_mapping_faces(rxmesh, patch_id, num_p_faces);
    }

    bool check_mapping_edges(const rxmesh::RXMesh& rxmesh,
                             const uint32_t        patch_id,
                             const uint32_t        num_p_edges)
    {
        // 1) For each local edge in the patch, get its global id using the
        // mapping (using m_h_patches_ltog_e)

        // 2) get the local edge's local vertices (using m_h_patches_ev)

        // 3) map the local vertices to their global id (using
        // m_h_patches_ltog_v)

        // 4) use the converted vertices to get their global edge id
        //(using m_edges_map)

        // 5) check if the resulting global edge id in 4) matches that
        // obtained in 1)

        for (uint16_t e_l = 0; e_l < num_p_edges; ++e_l) {

            // 1)
            // convert the local edge to global one
            uint32_t e_ltog = rxmesh.m_h_patches_ltog_e.at(patch_id).at(e_l);

            // 2)
            // get the local vertices
            uint16_t v0_l = rxmesh.m_h_patches_ev.at(patch_id).at(e_l * 2);
            uint16_t v1_l = rxmesh.m_h_patches_ev.at(patch_id).at(e_l * 2 + 1);

            // 3)
            // convert the local vertices to global
            uint32_t v0_ltog = rxmesh.m_h_patches_ltog_v.at(patch_id).at(v0_l);
            uint32_t v1_ltog = rxmesh.m_h_patches_ltog_v.at(patch_id).at(v1_l);


            // 4)
            // use the convered vertices to look for the edge global id
            auto my_edge = rxmesh::detail::edge_key(v0_ltog, v1_ltog);

            uint32_t e_g;
            try {
                e_g = rxmesh.m_edges_map.at(my_edge);
            } catch (const std::out_of_range&) {
                if (!m_quite) {
                    RXMESH_ERROR(
                        "RXMeshTest::check_mapping_edges() can not "
                        "find the corresponding edge between global vertices "
                        "{} and {} with local id {} and in patch {} of "
                        "converted to global vertices",
                        v0_ltog,
                        v1_ltog,
                        v0_l,
                        v1_l,
                        patch_id);
                }
                return false;
            }


            // 5)
            // check if the edge obtain from the converted (to global)
            // vertices matche the edge obtain from just mapping the local
            // edge to global one
            if (e_g != e_ltog) {
                if (!m_quite) {
                    RXMESH_ERROR(
                        "RXMeshTest::check_mapping_edges() Edge mapping "
                        "results do not match. Output summary: patch id = "
                        "{}, local edge id = {}, mapped to = {}, local "
                        "vertices id = ({}, {}) mapped to= ({}, {}), global "
                        "edge connecting the mapped global vertices = {}",
                        patch_id,
                        e_l,
                        e_ltog,
                        v0_l,
                        v1_l,
                        v0_ltog,
                        v1_ltog,
                        e_g);
                }
                return false;
            }
        }
        return true;
    }

    bool check_mapping_faces(const rxmesh::RXMesh& rxmesh,
                             const uint32_t        patch_id,
                             const uint32_t        num_p_faces)
    {
        using namespace rxmesh;
        // 1) for each local face in the patch, get its global id using the
        // mapping (using m_h_patches_ltog_f)

        // 2) get the local face's local edges (using m_h_patches_fe)

        // 3) map the local edges to their global id
        //(using m_h_patches_ltog_v)

        // 4) use the converted edges to get their global face id (using
        // m_h_patches_fe)


        // 5) check if the resulting global face id in 4) matches that
        // obtained in 1)
        std::vector<uint16_t> e_l(3);
        std::vector<uint16_t> e_g(3);
        std::vector<uint16_t> e_ltog(3);

        for (uint16_t f_l = 0; f_l < num_p_faces; ++f_l) {
            // 1)
            // convert the local face to global one
            uint32_t f_ltog = rxmesh.m_h_patches_ltog_f.at(patch_id).at(f_l);

            // 2)
            // get the local edges
            for (uint32_t i = 0; i < 3; ++i) {
                e_l[i] = rxmesh.m_h_patches_fe.at(patch_id).at(f_l * 3 + i);
                // shift right because the first bit is reserved for edge
                // direction
                flag_t dir(0);
                RXMeshContext::unpack_edge_dir(e_l[i], e_l[i], dir);
            }

            // 3)
            // convert the local edges to global
            for (uint32_t i = 0; i < 3; ++i) {
                e_ltog[i] = rxmesh.m_h_patches_ltog_e.at(patch_id).at(e_l[i]);
            }

            // 4)
            // from the mapped face (f_ltog) get its global edges
            for (uint32_t i = 0; i < 3; ++i) {
                e_g[i] = m_h_FE[f_ltog][i];
            }

            // 5)
            // check if the global edges matches the mapping edges
            for (uint32_t i = 0; i < 3; ++i) {
                if (e_g[i] != e_ltog[i]) {
                    if (!m_quite) {
                        RXMESH_ERROR(
                            "RXMeshTest::check_mapping_faces() Face mapping "
                            "results does not match. Output summary: patch id "
                            "= {}, local edge id = {}, mapped to = {}, local "
                            "edges id = ({}, {}, {}), mapped to = ({}, {}, "
                            "{}), global edges obtained from the mapped global "
                            "face= ({}, {}, {})",
                            patch_id,
                            f_l,
                            f_ltog,
                            e_l[0],
                            e_l[1],
                            e_l[2],
                            e_ltog[0],
                            e_ltog[1],
                            e_ltog[2],
                            e_ltog[0],
                            e_ltog[1],
                            e_ltog[2]);
                    }
                    return false;
                }
            }
        }


        return true;
    }
};