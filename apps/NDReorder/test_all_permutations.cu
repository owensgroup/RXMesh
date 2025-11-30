
#include <filesystem>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_factory.h"

#include "rxmesh/matrix/mgnd_permute.cuh"
#include "rxmesh/matrix/nd_permute.cuh"
#include "rxmesh/matrix/permute_util.h"
#include "rxmesh/matrix/sparse_matrix.h"

#include "rxmesh/matrix/cholesky_solver.h"

#include "count_nnz_fillin.h"

#include "render_permutation.h"

#include "metis.h"

using namespace rxmesh;

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "cloth_uni_loop.obj";

    uint32_t device_id = 0;

    int n = -1;
} Arg;

template <typename EigeMatT>
void save_matrix_and_permutation(RXMeshStatic&           rx,
                                 EigeMatT&               eigen_mat,
                                 const std::vector<int>& h_permute,
                                 const std::string       name)
{
    auto* values = eigen_mat.valuePtr();       // pointer to non-zero values
    auto* outer  = eigen_mat.outerIndexPtr();  // row offsets
    // auto* inner  = eigen_mat.innerIndexPtr();  // column indices

    rx.for_each_vertex(HOST, [&](const VertexHandle vh) mutable {
        int row = rx.linear_id(vh);

        for (int i = outer[row]; i < outer[row + 1]; ++i) {
            values[i] = vh.patch_id();
        }
    });

    save_sparse_mat(eigen_mat, "C:\\Github\\VizEigenSparseMat\\" + name);
    save_permutation(h_permute,
                     "C:\\Github\\VizEigenSparseMat\\permute_" + name);
}

template <typename EigeMatT>
void no_permute(RXMeshStatic& rx, const EigeMatT& eigen_mat)
{
    std::vector<int> h_permute(eigen_mat.rows());

    fill_with_sequential_numbers(h_permute.data(), h_permute.size());

    render_permutation(rx, h_permute, "No_PERM");

    int nnz = count_nnz_fillin(eigen_mat, h_permute, "natural");

    RXMESH_INFO(" No-permutation NNZ = {}, sparsity = {} %",
                nnz,
                100 * float(nnz) / float(eigen_mat.rows() * eigen_mat.cols()));
}


template <typename T, typename EigeMatT>
void with_metis(RXMeshStatic&          rx,
                const SparseMatrix<T>& rx_mat,
                EigeMatT               eigen_mat)
{
    assert(rx_mat.rows() == eigen_mat.rows());
    assert(rx_mat.cols() == eigen_mat.cols());
    assert(rx_mat.non_zeros() == eigen_mat.nonZeros());

    idx_t n = eigen_mat.rows();

    // xadj is of length n+1 marking the start of the adjancy list of each
    // vertex in adjncy.
    std::vector<idx_t> xadj(n + 1);

    // adjncy stores the adjacency lists of the vertices. The adjnacy list of a
    // vertex should not contain the vertex itself.
    std::vector<idx_t> adjncy;
    adjncy.reserve(eigen_mat.nonZeros());

    // populate xadj and adjncy
    xadj[0] = 0;
    for (int r = 0; r < rx_mat.rows(); ++r) {
        int start = rx_mat.row_ptr()[r];
        int stop  = rx_mat.row_ptr()[r + 1];
        for (int i = start; i < stop; ++i) {
            int c = rx_mat.col_idx()[i];
            if (r != c) {
                adjncy.push_back(c);
            }
        }
        xadj[r + 1] = adjncy.size();
    }

    // is an array of size n such that if A and A' are the original and
    // permuted matrices, then A'[i] = A[perm[i]].
    std::vector<idx_t> h_permute(n);

    // iperm is an array of size n such that if A and A' are the original
    // and permuted matrices, then A[i] = A'[iperm[i]].
    std::vector<idx_t> h_iperm(n);

    // Metis options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);


    /*// Specifies the partitioning method
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;

    // Specifies the type of objective
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_NODE;

    // Specifies the matching scheme to be used during coarsening
    options[METIS_OPTION_CTYPE] = METIS_CTYPE_RM;

    // Determines the algorithm used during initial partitioning.
    options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_EDGE;

    // Determines the algorithm used for refinement
    options[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP1SIDED;*/

    // Used to indicate which numbering scheme is used for the adjacency
    // structure of a graph or the elementnode structure of a mesh.
    options[METIS_OPTION_NUMBERING] = 0;  // 0-based indexing

    /*// Specifies that the graph should be compressed by combining together
    // vertices that have identical adjacency lists.
    options[METIS_OPTION_COMPRESS] = 0;  // Does not try to compress the graph.

    // Specifies the amount of progress/debugging information will be printed
    options[METIS_OPTION_DBGLVL] = 0;*/

    CPUTimer timer;
    timer.start();

    int metis_ret = METIS_NodeND(&n,
                                 xadj.data(),
                                 adjncy.data(),
                                 NULL,
                                 options,
                                 h_permute.data(),
                                 h_iperm.data());
    timer.stop();

    RXMESH_INFO(" METIS took {} (ms)", timer.elapsed_millis());

    if (metis_ret != 1) {
        RXMESH_ERROR("METIS Failed!");
    }

    if (!is_unique_permutation(h_iperm.size(), h_iperm.data())) {
        RXMESH_ERROR("METIS Permutation is not unique.");
    }

    render_permutation(rx, h_iperm, "METIS");

    int nnz = count_nnz_fillin(eigen_mat, h_iperm, "metis");

    // save_matrix_and_permutation(rx, eigen_mat, h_iperm, "metis.txt");
    RXMESH_INFO(" With METIS Nested Dissection NNZ = {}, sparsity = {} %",
                nnz,
                100 * float(nnz) / float(eigen_mat.rows() * eigen_mat.cols()));
}

template <typename EigeMatT>
void with_gpumgnd(RXMeshStatic& rx, const EigeMatT& eigen_mat)
{
    std::vector<int> h_permute(eigen_mat.rows());

    mgnd_permute(rx, h_permute.data());


    if (!is_unique_permutation(h_permute.size(), h_permute.data())) {
        RXMESH_ERROR("GPUMGND Permutation is not unique.");
    }

    std::vector<int> helper(rx.get_num_vertices());
    inverse_permutation(rx.get_num_vertices(), h_permute.data(), helper.data());

    render_permutation(rx, h_permute, "GPUMGND");

    int nnz = count_nnz_fillin(eigen_mat, h_permute, "gpumgnd");

    RXMESH_INFO(" With GPUMGND NNZ = {}, sparsity = {} %",
                nnz,
                100 * float(nnz) / float(eigen_mat.rows() * eigen_mat.cols()));
}

template <typename EigeMatT>
void with_gpu_nd(RXMeshStatic& rx, EigeMatT eigen_mat)
{
    std::vector<int> h_permute(eigen_mat.rows());

    nd_permute(rx, h_permute.data());

    if (!is_unique_permutation(h_permute.size(), h_permute.data())) {
        RXMESH_ERROR("GPUND Permutation is not unique.");
    }

    std::vector<int> helper(rx.get_num_vertices());
    inverse_permutation(rx.get_num_vertices(), h_permute.data(), helper.data());

    render_permutation(rx, h_permute, "GPUND");

    int nnz = count_nnz_fillin(eigen_mat, h_permute, "gpund");

    // save_matrix_and_permutation(rx, eigen_mat, h_permute, "gpu_nd.txt");
    RXMESH_INFO(" With GPUND NNZ = {}, sparsity = {} %",
                nnz,
                100 * float(nnz) / float(eigen_mat.rows() * eigen_mat.cols()));
}

template <typename T, typename EigeMatT>
void with_amd(RXMeshStatic&    rx,
              SparseMatrix<T>& rx_mat,
              const EigeMatT&  eigen_mat)
{
    std::vector<int> h_permute(eigen_mat.rows());

    CholeskySolver solver(&rx_mat, PermuteMethod::SYMAMD);
    solver.permute_alloc();
    solver.permute(rx);

    const int* h_perm = solver.get_h_permute();

    std::memcpy(h_permute.data(),
                solver.get_h_permute(),
                h_permute.size() * sizeof(int));


    std::vector<int> helper(rx.get_num_vertices());
    inverse_permutation(rx.get_num_vertices(), h_permute.data(), helper.data());


    if (!is_unique_permutation(h_permute.size(), h_permute.data())) {
        RXMESH_ERROR("AMD Permutation is not unique.");
    }

    render_permutation(rx, h_permute, "AMD");

    int nnz = count_nnz_fillin(eigen_mat, h_permute, "amd");

    RXMESH_INFO(" With AMD NNZ = {}, sparsity = {} %",
                nnz,
                100 * float(nnz) / float(eigen_mat.rows() * eigen_mat.cols()));
}

template <typename T, typename EigeMatT>
void with_symrcm(RXMeshStatic&    rx,
                 SparseMatrix<T>& rx_mat,
                 const EigeMatT&  eigen_mat)
{
    std::vector<int> h_permute(eigen_mat.rows());

    CholeskySolver solver(&rx_mat, PermuteMethod::SYMRCM);
    solver.permute_alloc();
    solver.permute(rx);

    const int* h_perm = solver.get_h_permute();

    std::memcpy(h_permute.data(),
                solver.get_h_permute(),
                h_permute.size() * sizeof(int));

    if (!is_unique_permutation(h_permute.size(), h_permute.data())) {
        RXMESH_ERROR("SYMRCM Permutation is not unique.");
    }

    std::vector<int> helper(rx.get_num_vertices());
    inverse_permutation(rx.get_num_vertices(), h_permute.data(), helper.data());

    render_permutation(rx, h_permute, "symrcm");

    int nnz = count_nnz_fillin(eigen_mat, h_permute, "symrcm");

    RXMESH_INFO(" With SYMRCM NNZ = {}, sparsity = {} %",
                nnz,
                100 * float(nnz) / float(eigen_mat.rows() * eigen_mat.cols()));
}

void all_perm(RXMeshStatic& rx)
{
    // VV matrix
    SparseMatrix<float> rx_mat(rx, Op::VV);

    // populate an SPD matrix
    rx_mat.for_each([](int r, int c, float& val) {
        if (r == c) {
            val = 10.0f;
        } else {
            val = -1.0f;
        }
    });

    RXMESH_INFO(
        " Input Matrix NNZ = {}, sparsity = {} %",
        rx_mat.non_zeros(),
        100 * float(rx_mat.non_zeros()) / float(rx_mat.rows() * rx_mat.cols()));

#if USE_POLYSCOPE
    rx.render_face_patch();
    rx.render_vertex_patch();
#endif

    // convert matrix to Eigen
    auto eigen_mat = rx_mat.to_eigen_copy();

    assert(eigen_mat.nonZeros() == rx_mat.non_zeros());

    //no_permute(rx, eigen_mat);

    with_amd(rx, rx_mat, eigen_mat);

    with_symrcm(rx, rx_mat, eigen_mat);

    with_metis(rx, rx_mat, eigen_mat);

    //with_gpumgnd(rx, eigen_mat);

    with_gpu_nd(rx, eigen_mat);

#if USE_POLYSCOPE
    polyscope::show();
#endif
}

int main(int argc, char** argv)
{
    Log::init(spdlog::level::info);

    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: NDReorder.exe < -option X>\n"
                        " -h:          Display this massage and exits\n"
                        " -input:      Input file. Only accepts OBJ files. Default is {}\n"
                        " -n:          Number of grid points for a grid mesh. Default is {}\n"
                        " -device_id:  GPU device ID. Default is {}",
            Arg.obj_file_name,  Arg.n, Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }

        if (cmd_option_exists(argv, argc + argv, "-input")) {
            Arg.obj_file_name =
                std::string(get_cmd_option(argv, argv + argc, "-input"));
        }

        if (cmd_option_exists(argv, argc + argv, "-device_id")) {
            Arg.device_id =
                atoi(get_cmd_option(argv, argv + argc, "-device_id"));
        }

        if (cmd_option_exists(argv, argc + argv, "-n")) {
            Arg.n = atoi(get_cmd_option(argv, argv + argc, "-n"));
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("device_id= {}", Arg.device_id);
    RXMESH_TRACE("n= {}", Arg.n);

    cuda_query(Arg.device_id);

    if (Arg.n > 0) {
        std::vector<std::vector<float>>    verts;
        std::vector<std::vector<uint32_t>> fv;

        create_plane(verts, fv, Arg.n, Arg.n, 2);
        RXMeshStatic rx(fv);
        rx.add_vertex_coordinates(verts, "plane");

        all_perm(rx);

    } else {
        // const std::string p_file = STRINGIFY(OUTPUT_DIR) +
        //                            extract_file_name(Arg.obj_file_name) +
        //                            "_patches";
        RXMeshStatic rx(Arg.obj_file_name, "", 128);
        // if (!std::filesystem::exists(p_file)) {
        //     rx.save(p_file);
        // }

        all_perm(rx);
    }
}
