#include <metis.h>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Eigen;

// Structure to hold vertex information
struct Vertex
{
    float x, y, z;
};

// Structure to hold face information
struct Face
{
    int v1, v2, v3;
};

// Function to parse the OBJ file
void parseOBJ(const std::string&   filename,
              std::vector<Vertex>& vertices,
              std::vector<Face>&   faces)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.substr(0, 2) == "v ") {
            Vertex vertex;
            sscanf(line.c_str(), "v %f %f %f", &vertex.x, &vertex.y, &vertex.z);
            vertices.push_back(vertex);
        } else if (line.substr(0, 2) == "f ") {
            Face face;
            sscanf(line.c_str(), "f %d %d %d", &face.v1, &face.v2, &face.v3);
            faces.push_back(face);
        }
    }

    file.close();
}

// Function to construct the adjacency matrix
void constructAdjacencyMatrix(const std::vector<Vertex>&  vertices,
                              const std::vector<Face>&    faces,
                              Eigen::SparseMatrix<float>& adjacencyMatrix)
{
    typedef Eigen::Triplet<float> T;
    std::vector<T>                tripletList;

    for (int i = 0; i < vertices.size(); i++) {
        tripletList.push_back(T(i, i, 10.0f));
    }

    for (const auto& face : faces) {
        int v1 = face.v1 - 1;  // OBJ indexing starts at 1
        int v2 = face.v2 - 1;
        int v3 = face.v3 - 1;

        // printf("v1: %d, v2: %d, v3: %d\n", v1, v2, v3);

        tripletList.push_back(T(v1, v2, 1.0f));
        tripletList.push_back(T(v2, v1, 1.0f));

        tripletList.push_back(T(v1, v3, 1.0f));
        tripletList.push_back(T(v3, v1, 1.0f));

        tripletList.push_back(T(v2, v3, 1.0f));
        tripletList.push_back(T(v3, v2, 1.0f));
    }

    std::sort(
        tripletList.begin(), tripletList.end(), [](const T& a, const T& b) {
            return (a.row() < b.row()) ||
                   (a.row() == b.row() && a.col() < b.col());
        });

    auto last = std::unique(
        tripletList.begin(), tripletList.end(), [](const T& a, const T& b) {
            return a.row() == b.row() && a.col() == b.col();
        });
    tripletList.erase(last, tripletList.end());

    // for (const auto& triplet : tripletList) {
    //     printf("row: %d, col: %d, value: %f\n", triplet.row(), triplet.col(),
    //     triplet.value());
    // }

    printf("Number of vertices: %lu\n", vertices.size());

    // adjacencyMatrix.resize(vertices.size(), vertices.size());
    adjacencyMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
}

// Main function to load an OBJ file and output a sparse matrix
Eigen::SparseMatrix<float> loadOBJToSparseMatrix(const std::string& filename)
{
    printf("Start parsing %s\n", filename.c_str());

    std::vector<Vertex> vertices;
    std::vector<Face>   faces;
    parseOBJ(filename, vertices, faces);

    printf("%s finish parsing\n", filename.c_str());

    Eigen::SparseMatrix<float> adjacencyMatrix(vertices.size(),
                                               vertices.size());
    constructAdjacencyMatrix(vertices, faces, adjacencyMatrix);

    printf("%s finish constructing adjacency matrix\n", filename.c_str());

    return adjacencyMatrix;
}

void printNonZerosRatio(const SparseMatrix<float>& original_matrix,
                        const SparseMatrix<float>& factorized_matrix,
                        const std::string&         name)
{
    int nnz_original   = original_matrix.nonZeros();
    int nnz_factorized = factorized_matrix.nonZeros();
    printf(
        "NNZ ratio for %s: %f\n", name.c_str(), float(nnz_factorized) / float(nnz_original));
}


void processmesh_original(const std::string&      inputfile)
{

    Eigen::SparseMatrix<float> adjMatrix = loadOBJToSparseMatrix(inputfile);

    int num_vertices = adjMatrix.rows();

    // Perform Cholesky factorization without reordering
    Eigen::SimplicialLLT<Eigen::SparseMatrix<float>,
                         Eigen::Lower,
                         Eigen::NaturalOrdering<int>>
        cholesky(adjMatrix);
    if (cholesky.info() != Eigen::Success) {
        printf("Cholesky decomposition failed with code %d\n", cholesky.info());
        return;
    }
    SparseMatrix<float> L = cholesky.matrixL();
    printNonZerosRatio(adjMatrix, L, "the factorized matrix L (without reordering)");
}


void processmesh_metis(const std::string& inputfile) {

    // Load OBJ file and convert to sparse adjacency matrix
    Eigen::SparseMatrix<float> adjMatrix = loadOBJToSparseMatrix(inputfile);

    int num_vertices = adjMatrix.rows();

    // Convert Eigen sparse matrix to METIS format
    idx_t n = num_vertices;
    std::vector<idx_t> xadj(n + 1);
    std::vector<idx_t> adjncy;
    std::vector<real_t> adjwgt;

    xadj[0] = 0;
    for (int k = 0; k < adjMatrix.outerSize(); ++k) {
        for (Eigen::SparseMatrix<float>::InnerIterator it(adjMatrix, k); it; ++it) {
            if (it.row() != it.col()) { // Exclude diagonal elements
                adjncy.push_back(it.col());
                adjwgt.push_back(it.value());
            }
        }
        xadj[k + 1] = adjncy.size();
    }

    std::vector<idx_t> perm(n);
    std::vector<idx_t> iperm(n);
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0; // 0-based indexing

    idx_t nvtxs = n;
    idx_t ncon = 1; // Number of balancing constraints
    METIS_NodeND(&nvtxs, &xadj[0], &adjncy[0], NULL, options, &perm[0], &iperm[0]);

    // Apply permutation to the sparse matrix (P * A * P^T)
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, idx_t> permMatrix(perm.size());
    permMatrix.indices() = Eigen::Map<Eigen::Matrix<idx_t, Eigen::Dynamic, 1>>(perm.data(), perm.size());

    Eigen::SparseMatrix<float> permutedMatrix = permMatrix.transpose() * adjMatrix * permMatrix;

    // Perform Cholesky factorization with reordering
    Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> cholesky(permutedMatrix);

    if (cholesky.info() != Eigen::Success) {
        printf("Cholesky decomposition failed with code %d\n", cholesky.info());
        return;
    }

    Eigen::SparseMatrix<float> L = cholesky.matrixL();
    printNonZerosRatio(adjMatrix, L, "the factorized matrix L with metis");
}


void processmesh_ordering(const std::string&      inputfile,
                 const std::vector<uint32_t>& reordering)
{

    Eigen::SparseMatrix<float> adjMatrix = loadOBJToSparseMatrix(inputfile);

    // Reorder the adjacency matrix using the pr                                                                                           ovided reordering array
    int num_vertices = adjMatrix.rows();
    SparseMatrix<float>         reorderedAdjMatrix(num_vertices, num_vertices);
    std::vector<Triplet<float>> reorderedTripletList; 

    for (int k = 0; k < adjMatrix.outerSize(); ++k) {
        for (SparseMatrix<float>::InnerIterator it(adjMatrix, k); it; ++it) {
            int i = reordering[it.row()];
            int j = reordering[it.col()];
            reorderedTripletList.push_back(Triplet<float>(i, j, it.value()));
        }
    }

    reorderedAdjMatrix.setFromTriplets(reorderedTripletList.begin(),
                                       reorderedTripletList.end());

    // Perform Cholesky factorization on the reordered matrix
    SimplicialLLT<SparseMatrix<float>,
                  Eigen::Lower,
                  Eigen::NaturalOrdering<int>>
        lltOfReorderedAdj(reorderedAdjMatrix);
    if (lltOfReorderedAdj.info() != Eigen::Success) {
        printf("Cholesky decomposition with reorder failed with code %d\n",
               lltOfReorderedAdj.info());
        return;
    }
    SparseMatrix<float> L_reordered = lltOfReorderedAdj.matrixL();
    printNonZerosRatio(adjMatrix, L_reordered, "the factorized matrix L (with reordering)");
}

void test_loader()
{
    std::string                filename = "path/to/your.obj";
    Eigen::SparseMatrix<float> adjacencyMatrix =
        loadOBJToSparseMatrix(filename);
}

int test_solver()
{

    // create sparse Matrix
    int                                 n = 5;
    std::vector<Eigen::Triplet<double>> ijv;
    for (int i = 0; i < n; i++) {
        ijv.push_back(Eigen::Triplet<double>(i, i, 1));
        if (i < n - 1) {
            ijv.push_back(Eigen::Triplet<double>(i + 1, i, -0.9));
        }
    }
    Eigen::SparseMatrix<double> X(n, n);
    X.setFromTriplets(ijv.begin(), ijv.end());
    Eigen::SparseMatrix<double> XX = X * X.transpose();

    // Cholesky decomposition
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> cholesky;
    cholesky.analyzePattern(XX);
    cholesky.factorize(XX);

    std::cout << Eigen::MatrixXd(XX) << std::endl;
    std::cout << Eigen::MatrixXd(cholesky.matrixL()) << std::endl;
}
