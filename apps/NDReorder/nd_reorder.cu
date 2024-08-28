#include "gtest/gtest.h"

#include <filesystem>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/mgnd_permute.cuh"
#include "rxmesh/matrix/nd_reorder.cuh"
#include "rxmesh/matrix/permute_util.h"
#include "rxmesh/matrix/sparse_matrix.cuh"

#include <Eigen/Dense>
#include <Eigen/Sparse>

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere1.obj";
    uint16_t    nd_level      = 4;
    uint32_t    device_id     = 0;
} Arg;

/**
 * @brief calculate the number of nnz after Cholesky factorization
 */
int post_chol_factorization_nnz(rxmesh::RXMeshStatic& rx,
                                std::vector<int>&     h_reorder_array)
{
    using namespace rxmesh;

    assert(h_reorder_array.size() == rx.get_num_vertices());

    // VV matrix
    rxmesh::SparseMatrix<float> mat(rx);   

    // populate an SPD matrix
    mat.for_each([](int r, int c, float& val) {
        if (r == c) {
            val = 10.0f;
        } else {
            val = -1.0f;
        }
    });

    // convert matrix to Eigen
    auto eigen_mat = mat.to_eigen_copy();

    //std::cout << "eigen_mat\n" << eigen_mat << "\n";

    // permutation array in Eigen format
    Eigen::Map<Eigen::VectorXi> p(h_reorder_array.data(),
                                  rx.get_num_vertices());

    // permutation matrix
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(
        rx.get_num_vertices());
    for (int i = 0; i < rx.get_num_vertices(); ++i) {
        perm.indices()[i] = h_reorder_array[i];
    }

    Eigen::SparseMatrix<float> permuted_mat =
        perm.transpose() * eigen_mat * perm;

    // compute Cholesky factorization on the permuted matirx

    Eigen::SimplicialLLT<Eigen::SparseMatrix<float>,
                         Eigen::Lower,
                         Eigen::NaturalOrdering<int>>
        solver;
    solver.compute(permuted_mat);

    if (solver.info() != Eigen::Success) {
        RXMESH_ERROR(
            "post_chol_factorization_nnz(): Cholesky decomposition with "
            "reorder failed with code {}",
            solver.info());
        return -1;
    }

    // extract nnz from lower matrix
    Eigen::SparseMatrix<float> ff = solver.matrixL();

    //std::cout << "ff\n" << ff << "\n";

    //these are the nnz on (strictly) the lower part 
    int lower_nnz = ff.nonZeros() - ff.rows();

    //multiply by two to account for lower and upper parts of the matirx
    //add rows() to account for entries along the diagonal  
    return 2*lower_nnz + ff.rows();
}

TEST(Apps, NDReorder)
{
    using namespace rxmesh;

    cuda_query(Arg.device_id);

    const std::string p_file = STRINGIFY(OUTPUT_DIR) +
                               extract_file_name(Arg.obj_file_name) +
                               "_patches";
    RXMeshStatic rx(Arg.obj_file_name, p_file);
    if (!std::filesystem::exists(p_file)) {
        rx.save(p_file);
    }

    // allocate result array
    std::vector<int> h_permute(rx.get_num_vertices());
    fill_with_sequential_numbers(h_permute.data(), h_permute.size());

    // cuda_nd_reorder(rx, h_reorder_array, Arg.nd_level);

    EXPECT_TRUE(is_unique_permutation(rx.get_num_vertices(), h_permute.data()));

    RXMESH_INFO(" Post reorder NNZ = {}",
                post_chol_factorization_nnz(rx, h_permute));

    // processmesh_ordering(Arg.obj_file_name, h_permute);
    //  processmesh_metis(Arg.obj_file_name);
    //  processmesh_original(Arg.obj_file_name);
}

int main(int argc, char** argv)
{
    using namespace rxmesh;
    Log::init();

    ::testing::InitGoogleTest(&argc, argv);

    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: NDReorder.exe < -option X>\n"
                        " -h:          Display this massage and exits\n"
                        " -input:      Input file. Input file should under the input/ subdirectory\n"
                        "              Default is {} \n"
                        "              Hint: Only accepts OBJ files\n"                                              
                        " -device_id:  GPU device ID. Default is {}",
            Arg.obj_file_name,  Arg.device_id);
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

        if (cmd_option_exists(argv, argc + argv, "-nd_level")) {
            Arg.nd_level = atoi(get_cmd_option(argv, argv + argc, "-nd_level"));
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}
