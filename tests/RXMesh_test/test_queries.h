#include <functional>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"
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
    constexpr uint32_t      blockThreads = 320;
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


    for (uint32_t itr = 0; itr < rxmesh_args.num_run; itr++) {
        // Reset input/output
        input.reset(InputHandleT(), rxmesh::DEVICE);
        output.reset(OutputHandleT(), rxmesh::DEVICE);
        CUDA_ERROR(cudaDeviceSynchronize());

        CUDA_ERROR(cudaProfilerStart());
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
        td.time_ms.push_back(timer.elapsed_millis());
    }

    // move containers to the CPU for testing
    output.move(rxmesh::DEVICE, rxmesh::HOST);
    input.move(rxmesh::DEVICE, rxmesh::HOST);

    // verify
    bool passed = tester.run_test(rx, Faces, input, output);

    td.passed.push_back(passed);
    EXPECT_TRUE(passed) << "Testing: " << td.test_name;

    report.add_test(td);

    RXMESH_INFO(" {} {} time = {} (ms) \n",
                td.test_name.c_str(),
                (passed ? " passed " : " failed "),
                total_time / float(rxmesh_args.num_run));
}

TEST(RXMeshStatic, Queries)
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
