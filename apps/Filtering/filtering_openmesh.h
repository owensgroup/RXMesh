#pragma once

#include <omp.h>
#include <queue>
#include "../common/openmesh_report.h"
#include "../common/openmesh_trimesh.h"

/**
 *computeSigma_s()
 */
double computeSigma_s(
    const std::vector<TriMesh::VertexHandle>& vertex_neighbour,
    const TriMesh&                            mesh,
    const TriMesh::Point                      pi,
    const TriMesh::Normal                     ni)
{


    float  offset  = 0;
    float  sum     = 0;
    float  sum_sqs = 0;
    size_t count   = vertex_neighbour.size();
    for (size_t i = 0; i < count; ++i) {
        TriMesh::Point pj = mesh.point(vertex_neighbour[i]);
        float          t  = (pj - pi) | ni;
        t                 = sqrt(t * t);
        sum += t;
        sum_sqs += t * t;
    }
    float c = static_cast<float>(count);
    offset  = (sum_sqs / c) - ((sum * sum) / (c * c));

    float sigma_s =
        (sqrt(offset) < 1.0e-12) ? (sqrt(offset) + 1.0e-12) : sqrt(offset);
    return sigma_s;
}
/**
 * getAdaptiveVertexNeighbor()
 */
void getAdaptiveVertexNeighbor(
    TriMesh&                            mesh,
    TriMesh::VertexHandle               vh,
    float                               sigma_c,
    std::vector<TriMesh::VertexHandle>& vertex_neighbor)
{
    std::vector<bool> mark(mesh.n_vertices(), false);
    vertex_neighbor.clear();
    std::queue<TriMesh::VertexHandle> queue_vertex_handle;
    mark[vh.idx()] = true;
    queue_vertex_handle.push(vh);
    float          radius = 2.0 * sigma_c;
    TriMesh::Point ci     = mesh.point(vh);

    while (!queue_vertex_handle.empty()) {
        TriMesh::VertexHandle vh = queue_vertex_handle.front();
        vertex_neighbor.push_back(vh);
        queue_vertex_handle.pop();
        for (TriMesh::VertexVertexIter vv_it = mesh.vv_iter(vh);
             vv_it.is_valid();
             ++vv_it) {
            TriMesh::VertexHandle vh_neighbor = *vv_it;
            if (mark[vh_neighbor.idx()] == false) {
                TriMesh::Point cj     = mesh.point(vh_neighbor);
                float          length = (cj - ci).length();
                if (length <= radius)
                    queue_vertex_handle.push(vh_neighbor);
                mark[vh_neighbor.idx()] = true;
            }
        }
    }
}

template <typename T>
void filtering_openmesh(const int                    num_omp_threads,
                        TriMesh&                     input_mesh,
                        std::vector<std::vector<T>>& filtered_coord,
                        size_t&                      max_neighbour_size)
{
    // Report
    OpenMeshReport report("Filtering_OpenMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.system();
    report.model_data(Arg.obj_file_name, input_mesh);
    std::string method =
        "OpenMesh " + std::to_string(num_omp_threads) + " Core";
    report.add_member("method", method);
    report.add_member("num_filter_iter", Arg.num_filter_iter);


    // this where each thread will store its neighbour vertices
    // we allocate enough space such that each thread can store as much
    // neighbour vertices as the number of vertices in the mesh.
    std::vector<std::vector<TriMesh::VertexHandle>> vertex_neighbour;
    for (int i = 0; i < num_omp_threads; ++i) {
        std::vector<TriMesh::VertexHandle> vn;
        vn.reserve(input_mesh.n_vertices());
        vertex_neighbour.push_back(vn);
    }

    max_neighbour_size = 0;

    rxmesh::CPUTimer timer;
    timer.start();

    for (uint32_t itr = 0; itr < Arg.num_filter_iter; ++itr) {
        input_mesh.request_face_normals();
        input_mesh.request_vertex_normals();
        input_mesh.update_normals();

        const int num_vertrices = static_cast<int>(input_mesh.n_vertices());
#ifndef _MSC_VER
#pragma omp parallel for schedule(static) num_threads(num_omp_threads) \
    reduction(max                                                      \
              : max_neighbour_size)
#endif
        for (int vert = 0; vert < num_vertrices; vert++) {
            TriMesh::VertexIter v_it = input_mesh.vertices_begin() + vert;

            int tid = omp_get_thread_num();

            // calculate sigma_c
            TriMesh::Point  pi      = input_mesh.point(*v_it);
            TriMesh::Normal ni      = input_mesh.normal(*v_it);
            float           sigma_c = 1e10;
            for (TriMesh::VertexVertexIter vv_it = input_mesh.vv_iter(*v_it);
                 vv_it.is_valid();
                 vv_it++) {
                TriMesh::Point pj     = input_mesh.point(*vv_it);
                float          length = (pi - pj).length();
                if (length < sigma_c) {
                    sigma_c = length;
                }
            }

            // get the neighbor vertices
            vertex_neighbour[tid].clear();
            getAdaptiveVertexNeighbor(
                input_mesh, *v_it, sigma_c, vertex_neighbour[tid]);

            max_neighbour_size =
                max(max_neighbour_size, vertex_neighbour[tid].size());
            // calculate sigma_s
            float sigma_s =
                computeSigma_s(vertex_neighbour[tid], input_mesh, pi, ni);

            float sum        = 0;
            float normalizer = 0;

            // calculate new vertex position
            for (int iv = 0; iv < (int)vertex_neighbour[tid].size(); iv++) {
                TriMesh::Point pj = input_mesh.point(vertex_neighbour[tid][iv]);

                float t  = (pi - pj).length();
                float h  = (pj - pi) | ni;
                float wc = std::exp(-0.5 * t * t / (sigma_c * sigma_c));
                float ws = std::exp(-0.5 * h * h / (sigma_s * sigma_s));
                sum += wc * ws * h;
                normalizer += wc * ws;
            }
            auto updated_point      = pi + ni * (sum / normalizer);
            filtered_coord[vert][0] = updated_point[0];
            filtered_coord[vert][1] = updated_point[1];
            filtered_coord[vert][2] = updated_point[2];
        }

        // update the mesh for the next iterations (needed to update the
        // normals correctly)
#pragma omp parallel for schedule(static) num_threads(num_omp_threads)
        for (int vert = 0; vert < num_vertrices; vert++) {
            TriMesh::VertexIter v_it = input_mesh.vertices_begin() + vert;
            TriMesh::Point      p;
            p[0] = filtered_coord[vert][0];
            p[1] = filtered_coord[vert][1];
            p[2] = filtered_coord[vert][2];
            input_mesh.set_point(*v_it, p);
        }
    }

    timer.stop();

    report.add_member("max_neighbour_size", uint32_t(max_neighbour_size));
    RXMESH_TRACE("filtering_openmesh() max_neighbour_size= {}",
                 max_neighbour_size);
    RXMESH_TRACE("filtering_openmesh() took {} (ms) (i.e., {} ms/iter) ",
                 timer.elapsed_millis(),
                 timer.elapsed_millis() / float(Arg.num_filter_iter));


    // write output
    // std::string fn = STRINGIFY(OUTPUT_DIR) "filtering_openmesh.obj";
    // if (!OpenMesh::IO::write_mesh(input_mesh, fn)) {
    //    RXMESH_WARN("OpenMesh cannot write mesh to file {}", fn);
    //}


    // Finalize report
    report.add_member("total_time (ms)", timer.elapsed_millis());
    rxmesh::TestData td;
    td.test_name   = "MCF";
    td.num_threads = num_omp_threads;
    td.time_ms.push_back(timer.elapsed_millis());
    td.passed.push_back(true);
    report.add_test(td);
    report.write(
        Arg.output_folder + "/openmesh",
        "MCF_OpenMesh_" + rxmesh::extract_file_name(Arg.obj_file_name));
}