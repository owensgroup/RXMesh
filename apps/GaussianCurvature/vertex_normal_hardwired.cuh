#pragma once
#include <vector>
#include "rxmesh/util/log.h"
#include "rxmesh/util/report.h"

template <typename T>
static void __global__
vertex_normal_hardwired_kernel(const uint32_t  num_faces,
                               const uint32_t  num_vertices,
                               const uint32_t* d_faces,
                               const T*        d_vertex_coord,
                               T*              d_vertex_normal)
{
    uint32_t f_id = threadIdx.x + blockIdx.x * blockDim.x;

    auto l2_norm_sq = [](const T ax0,
                         const T ax1,
                         const T ax2,
                         const T bx0,
                         const T bx1,
                         const T bx2) {
        // compute (xa0-xb0)*(xa0-xb0) + (xa1-xb1)*(xa1-xb1) +
        // (xa2-xb2)*(xa2-xb2)
        T x0 = ax0 - bx0;
        T x1 = ax1 - bx1;
        T x2 = ax2 - bx2;
        return x0 * x0 + x1 * x1 + x2 * x2;
    };

    auto cross_product =
        [](T xv1, T yv1, T zv1, T xv2, T yv2, T zv2, T& xx, T& yy, T& zz) {
            xx = yv1 * zv2 - zv1 * yv2;
            yy = zv1 * xv2 - xv1 * zv2;
            zz = xv1 * yv2 - yv1 * xv2;
        };

    if (f_id < num_faces) {
        uint32_t v0 = d_faces[f_id * 3];
        uint32_t v1 = d_faces[f_id * 3 + 1];
        uint32_t v2 = d_faces[f_id * 3 + 2];

        const T v0x(d_vertex_coord[v0 * 3 + 0]),
            v0y(d_vertex_coord[v0 * 3 + 1]), v0z(d_vertex_coord[v0 * 3 + 2]);

        const T v1x(d_vertex_coord[v1 * 3 + 0]),
            v1y(d_vertex_coord[v1 * 3 + 1]), v1z(d_vertex_coord[v1 * 3 + 2]);

        const T v2x(d_vertex_coord[v2 * 3 + 0]),
            v2y(d_vertex_coord[v2 * 3 + 1]), v2z(d_vertex_coord[v2 * 3 + 2]);

        T nx, ny, nz;

        cross_product(v1x - v0x,
                      v1y - v0y,
                      v1z - v0z,
                      v2x - v0x,
                      v2y - v0y,
                      v2z - v0z,
                      nx,
                      ny,
                      nz);
        T l0 = l2_norm_sq(v0x, v0y, v0z, v1x, v1y, v1z);  // v0-v1
        T l1 = l2_norm_sq(v1x, v1y, v1z, v2x, v2y, v2z);  // v1-v2
        T l2 = l2_norm_sq(v2x, v2y, v2z, v0x, v0y, v0z);  // v2-v0

        atomicAdd(&d_vertex_normal[v0 * 3 + 0], nx / (l0 + l2));
        atomicAdd(&d_vertex_normal[v0 * 3 + 1], ny / (l0 + l2));
        atomicAdd(&d_vertex_normal[v0 * 3 + 2], nz / (l0 + l2));

        atomicAdd(&d_vertex_normal[v1 * 3 + 0], nx / (l1 + l0));
        atomicAdd(&d_vertex_normal[v1 * 3 + 1], ny / (l1 + l0));
        atomicAdd(&d_vertex_normal[v1 * 3 + 2], nz / (l1 + l0));

        atomicAdd(&d_vertex_normal[v2 * 3 + 0], nx / (l2 + l1));
        atomicAdd(&d_vertex_normal[v2 * 3 + 1], ny / (l2 + l1));
        atomicAdd(&d_vertex_normal[v2 * 3 + 2], nz / (l2 + l1));
    }
}

template <typename T>
inline void vertex_normal_hardwired(
    const std::vector<std::vector<uint32_t>>& Faces,
    const std::vector<std::vector<T>>&        Verts,
    const std::vector<T>&                     vertex_normal_gold)
{
    using namespace rxmesh;
    uint32_t num_vertices = Verts.size();
    uint32_t num_faces    = Faces.size();

    CustomReport report("VertexNormal_Hardwired");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, num_vertices, num_faces);
    report.add_member("method", std::string("Hardwired"));

    std::vector<uint32_t> h_face(num_faces * 3);
    std::vector<T>        h_verts(num_vertices * 3);


    for (uint32_t i = 0; i < num_faces; i++) {
        h_face[i * 3 + 0] = Faces[i][0];
        h_face[i * 3 + 1] = Faces[i][1];
        h_face[i * 3 + 2] = Faces[i][2];
    }
    for (uint32_t i = 0; i < num_vertices; ++i) {
        h_verts[i * 3 + 0] = Verts[i][0];
        h_verts[i * 3 + 1] = Verts[i][1];
        h_verts[i * 3 + 2] = Verts[i][2];
    }

    uint32_t* d_face(NULL);
    T*        d_verts(NULL);
    T*        d_normals(NULL);
    CUDA_ERROR(cudaMalloc((void**)&d_face, 3 * num_faces * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_verts, 3 * num_vertices * sizeof(T)));
    CUDA_ERROR(cudaMalloc((void**)&d_normals, 3 * num_vertices * sizeof(T)));
    CUDA_ERROR(cudaMemcpy(d_face,
                          h_face.data(),
                          h_face.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_verts,
                          h_verts.data(),
                          h_verts.size() * sizeof(T),
                          cudaMemcpyHostToDevice));

    const uint32_t threads = 256;
    const uint32_t blocks  = DIVIDE_UP(num_faces, threads);

    TestData td;
    td.test_name  = "VertexNormal";
    float vn_time = 0;
    for (uint32_t itr = 0; itr < Arg.num_run; ++itr) {
        CUDA_ERROR(cudaMemset(d_normals, 0, 3 * num_vertices * sizeof(T)));

        GPUTimer timer;
        timer.start();

        vertex_normal_hardwired_kernel<<<blocks, threads>>>(
            num_faces, num_vertices, d_face, d_verts, d_normals);

        timer.stop();
        CUDA_ERROR(cudaDeviceSynchronize());
        CUDA_ERROR(cudaGetLastError());
        td.time_ms.push_back(timer.elapsed_millis());
        vn_time += timer.elapsed_millis();
    }

    vn_time /= Arg.num_run;

    T* verts_normal_hardwired;
    verts_normal_hardwired = (T*)malloc(num_vertices * 3 * sizeof(T));
    CUDA_ERROR(cudaMemcpy(verts_normal_hardwired,
                          d_normals,
                          3 * num_vertices * sizeof(T),
                          cudaMemcpyDeviceToHost));

    GPU_FREE(d_normals);
    GPU_FREE(d_verts);
    GPU_FREE(d_face);


    RXMESH_TRACE("vertex_normal_hardwired() vertex normal kernel took {} (ms)",
                 vn_time);

    bool passed = compare(vertex_normal_gold.data(),
                          verts_normal_hardwired,
                          Verts.size() * 3,
                          false);
    td.passed.push_back(passed);
    EXPECT_TRUE(passed) << " Hardwired Validation failed \n";

    free(verts_normal_hardwired);

    report.add_test(td);

    report.write(
        Arg.output_folder + "/hardwired",
        "VertexNormal_Hardwired_" + extract_file_name(Arg.obj_file_name));
}