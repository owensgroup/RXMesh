#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/MshLoader.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/report.h"
#include "rxmesh_test.h"

#include "query_kernel.cuh"

template <rxmesh::Op op,
          typename InputHandleT,
          typename OutputHandleT,
          typename InputAttributeT,
          typename OutputAttributeT>
void launcher(const std::vector<std::vector<uint32_t>>& Faces,
              rxmesh::RXMeshStatic&                     rx,
              InputAttributeT&                          input,
              OutputAttributeT&                         output,
              RXMeshTest&                               tester,
              rxmesh::Report&                           report,
              bool                                      oriented)
{
    using namespace rxmesh;

    // launch box
    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({op},
                          launch_box,
                          (void*)query_kernel<blockThreads,
                                              op,
                                              InputHandleT,
                                              OutputHandleT,
                                              InputAttributeT,
                                              OutputAttributeT>,
                          oriented);

    // test data
    TestData td;
    td.test_name   = op_to_string(op);
    td.num_threads = launch_box.num_threads;
    td.num_blocks  = launch_box.blocks;
    td.dyn_smem    = launch_box.smem_bytes_dyn;
    td.static_smem = launch_box.smem_bytes_static;
    td.num_reg     = launch_box.num_registers_per_thread;

    float total_time = 0;


    // Reset input/output
    input.reset(InputHandleT(), rxmesh::DEVICE);
    output.reset(OutputHandleT(), rxmesh::DEVICE);
    CUDA_ERROR(cudaDeviceSynchronize());

    CUDA_ERROR(cudaProfilerStart());

    int num_run = 1000;

    for (int i = 0; i < num_run; ++i) {
        GPUTimer timer;
        timer.start();
        query_kernel<blockThreads, op, InputHandleT, OutputHandleT>
            <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
                rx.get_context(), input, output, oriented);

        timer.stop();
        CUDA_ERROR(cudaDeviceSynchronize());
        CUDA_ERROR(cudaGetLastError());
        CUDA_ERROR(cudaProfilerStop());

        total_time += timer.elapsed_millis();
    }
    td.time_ms.push_back(total_time / float(num_run));


    // move containers to the CPU for testing
    output.move(rxmesh::DEVICE, rxmesh::HOST);
    input.move(rxmesh::DEVICE, rxmesh::HOST);

    // verify
    bool passed = tester.run_test(rx, Faces, input, output);

    td.passed.push_back(passed);
    EXPECT_TRUE(passed) << "Testing: " << td.test_name;

    report.add_test(td);

    RXMESH_INFO(" {} {} time = {} (ms)",
                td.test_name.c_str(),
                (passed ? " passed " : " failed "),
                total_time / float(num_run));
}

TEST(RXMeshStatic, TriangleQueries)
{
    using namespace rxmesh;

    bool oriented = false;

    std::vector<std::vector<float>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(import_obj(rxmesh_args.obj_file_name, Verts, Faces));


    RXMeshStatic rx(Faces);


    // Report
    Report report;
    report = Report("QueryTest_RXMesh");
    report.command_line(rxmesh_args.argc, rxmesh_args.argv);
    report.device();
    report.system();
    report.model_data(rxmesh_args.obj_file_name, rx);
    report.add_member("method", std::string("RXMesh"));


    // Tester to verify all queries
    ::RXMeshTest tester(rx, Faces);
    EXPECT_TRUE(tester.run_ltog_mapping_test(rx, Faces))
        << "Local-to-global mapping test failed";

    {
        // VV
        auto input  = rx.add_vertex_attribute<VertexHandle>("input", 1);
        auto output = rx.add_vertex_attribute<VertexHandle>(
            "output", rx.get_input_max_valence());
        launcher<Op::VV, VertexHandle, VertexHandle>(
            Faces, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }


    {
        // VE
        auto input  = rx.add_vertex_attribute<VertexHandle>("input", 1);
        auto output = rx.add_vertex_attribute<EdgeHandle>(
            "output", rx.get_input_max_valence());
        launcher<Op::VE, VertexHandle, EdgeHandle>(
            Faces, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // VF
        auto input  = rx.add_vertex_attribute<VertexHandle>("input", 1);
        auto output = rx.add_vertex_attribute<FaceHandle>(
            "output", rx.get_input_max_valence());
        launcher<Op::VF, VertexHandle, FaceHandle>(
            Faces, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }


    {
        // EV
        auto input  = rx.add_edge_attribute<EdgeHandle>("input", 1);
        auto output = rx.add_edge_attribute<VertexHandle>("output", 2);
        launcher<Op::EV, EdgeHandle, VertexHandle>(
            Faces, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // EF
        auto input  = rx.add_edge_attribute<EdgeHandle>("input", 1);
        auto output = rx.add_edge_attribute<FaceHandle>(
            "output", rx.get_input_max_edge_incident_faces());
        launcher<Op::EF, EdgeHandle, FaceHandle>(
            Faces, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // FV
        auto input  = rx.add_face_attribute<FaceHandle>("input", 1);
        auto output = rx.add_face_attribute<VertexHandle>("output", 3);
        launcher<Op::FV, FaceHandle, VertexHandle>(
            Faces, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // FE
        auto input  = rx.add_face_attribute<FaceHandle>("input", 1);
        auto output = rx.add_face_attribute<EdgeHandle>("output", 3);
        launcher<Op::FE, FaceHandle, EdgeHandle>(
            Faces, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // FF
        auto input  = rx.add_face_attribute<FaceHandle>("input", 1);
        auto output = rx.add_face_attribute<FaceHandle>(
            "output", rx.get_input_max_face_adjacent_faces() + 2);
        launcher<Op::FF, FaceHandle, FaceHandle>(
            Faces, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    // Write the report
    report.write(
        rxmesh_args.output_folder + "/rxmesh",
        "QueryTest_RXMesh_" + extract_file_name(rxmesh_args.obj_file_name));
}

TEST(RXMeshStatic, TetQueries)
{
    using namespace rxmesh;

    const std::string tet_file = STRINGIFY(INPUT_DIR) "car.msh";
    bool              oriented = false;

    std::vector<std::vector<rx_coord_t>> Verts;
    std::vector<std::vector<uint32_t>>   Tets;
    ASSERT_EQ(load_msh(tet_file, Verts, Tets), MeshKind::Tet);

    std::vector<uint32_t> vertex_tet_degree(Verts.size(), 0);
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edge_tet_degree;
    std::map<std::array<uint32_t, 3>, uint32_t>       face_tet_degree;
    uint32_t                                          max_vt = 0;
    uint32_t                                          max_et = 0;
    uint32_t                                          max_ft = 0;

    constexpr auto edges = tet_edges();
    constexpr auto faces = tet_faces();

    for (const auto& tet : Tets) {
        for (uint32_t v : tet) {
            max_vt = std::max(max_vt, ++vertex_tet_degree[v]);
        }
        for (const auto& edge : edges) {
            const auto key = detail::edge_key(tet[edge[0]], tet[edge[1]]);
            max_et         = std::max(max_et, ++edge_tet_degree[key]);
        }
        for (const auto& face_vertices : faces) {
            std::array<uint32_t, 3> face = {tet[face_vertices[0]],
                                            tet[face_vertices[1]],
                                            tet[face_vertices[2]]};
            std::sort(face.begin(), face.end());
            max_ft = std::max(max_ft, ++face_tet_degree[face]);
        }
    }

    EXPECT_TRUE(std::any_of(face_tet_degree.begin(),
                            face_tet_degree.end(),
                            [](const auto& face) { return face.second == 1; }));
    EXPECT_TRUE(std::any_of(face_tet_degree.begin(),
                            face_tet_degree.end(),
                            [](const auto& face) { return face.second == 2; }));

    RXMeshStatic rx(tet_file);
    EXPECT_GT(rx.get_num_patches(), 1u);

    std::vector<uint32_t> face_list(3 * rx.get_num_faces());
    rx.create_face_list(face_list.data(), true);

    std::vector<std::vector<uint32_t>> Faces(rx.get_num_faces(),
                                             std::vector<uint32_t>(3));

    for (uint32_t f = 0; f < rx.get_num_faces(); ++f) {
        for (uint32_t v = 0; v < 3; ++v) {
            Faces[f][v] = face_list[3 * f + v];
        }
    }

    Report report("QueryTetTest_RXMesh");
    report.command_line(rxmesh_args.argc, rxmesh_args.argv);
    report.device();
    report.system();
    report.model_data(tet_file, rx);
    report.add_member("method", std::string("RXMesh"));

    ::RXMeshTest tester(rx, Faces);
    EXPECT_TRUE(tester.run_ltog_mapping_test(rx, Faces))
        << "Local-to-global mapping test failed";

    {
        // EV
        auto input  = rx.add_edge_attribute<EdgeHandle>("input", 1);
        auto output = rx.add_edge_attribute<VertexHandle>("output", 2);
        launcher<Op::EV, EdgeHandle, VertexHandle>(
            Faces, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // FV
        auto input  = rx.add_face_attribute<FaceHandle>("input", 1);
        auto output = rx.add_face_attribute<VertexHandle>("output", 3);
        launcher<Op::FV, FaceHandle, VertexHandle>(
            Faces, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // FE
        auto input  = rx.add_face_attribute<FaceHandle>("input", 1);
        auto output = rx.add_face_attribute<EdgeHandle>("output", 3);
        launcher<Op::FE, FaceHandle, EdgeHandle>(
            Faces, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // TV
        auto input  = rx.add_tet_attribute<TetHandle>("input", 1);
        auto output = rx.add_tet_attribute<VertexHandle>("output", 4);
        launcher<Op::TV, TetHandle, VertexHandle>(
            Tets, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // TE
        auto input  = rx.add_tet_attribute<TetHandle>("input", 1);
        auto output = rx.add_tet_attribute<EdgeHandle>("output", 6);
        launcher<Op::TE, TetHandle, EdgeHandle>(
            Tets, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // TF
        auto input  = rx.add_tet_attribute<TetHandle>("input", 1);
        auto output = rx.add_tet_attribute<FaceHandle>("output", 4);
        launcher<Op::TF, TetHandle, FaceHandle>(
            Tets, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // VT
        auto input  = rx.add_vertex_attribute<VertexHandle>("input", 1);
        auto output = rx.add_vertex_attribute<TetHandle>("output", max_vt);
        launcher<Op::VT, VertexHandle, TetHandle>(
            Tets, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // ET
        auto input  = rx.add_edge_attribute<EdgeHandle>("input", 1);
        auto output = rx.add_edge_attribute<TetHandle>("output", max_et);
        launcher<Op::ET, EdgeHandle, TetHandle>(
            Tets, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    {
        // FT
        auto input  = rx.add_face_attribute<FaceHandle>("input", 1);
        auto output = rx.add_face_attribute<TetHandle>("output", max_ft);
        launcher<Op::FT, FaceHandle, TetHandle>(
            Tets, rx, *input, *output, tester, report, oriented);
        rx.remove_attribute("input");
        rx.remove_attribute("output");
    }

    report.write(rxmesh_args.output_folder + "/rxmesh",
                 "QueryTetTest_RXMesh_" + extract_file_name(tet_file));
}
