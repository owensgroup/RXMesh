#pragma once
#include <vector>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.cuh"

#include "GMGProlongatorCreation.h"
#include "GMGRXMeshKernels.h"
#include "RXMeshMCFSetup.h"
#include "VCycle.h"

class GPUGMG
{
   public:
    float            ratio;
    int              N;
    int              numberOfLevels;
    std::vector<CSR> prolongationOperatorCSR;
    std::vector<CSR> prolongationOperatorCSRTranspose;
    std::vector<CSR> equationsPerLevel;
    VectorCSR3D      B_v;

    GPUGMG()
    {
        ratio          = 3;
        numberOfLevels = 0;
    }
    GPUGMG(RXMeshStatic& rx) : B_v(rx.get_num_vertices())
    {
        N = rx.get_num_vertices();

        ratio          = 7;
        numberOfLevels = 0;
        for (int i = 0; i < 16; i++) {
            if ((int)N / (int)powf(ratio, i) > 6) {
                numberOfLevels++;
            }
        }
        std::cout << "\n Mesh can have " << numberOfLevels << " levels";
    }
    void ConstructOperators(RXMeshStatic&);
};

void GPUGMG::ConstructOperators(RXMeshStatic& rx)
{

    // set up initial variables

    Timers<GPUTimer> timers;
    timers.add("Total");
    timers.add("Sampling");
    timers.add("Clustering level 1");
    timers.add("Neighbors level 1");
    timers.add("Constructing level 1");
    timers.add("Constructing operators without level 1");
    timers.add("Constructing LHS");


    N                                = rx.get_num_vertices();
    int currentLevel                 = 1;                   // first coarse mesh
    int numberOfSamplesForFirstLevel = N / powf(ratio, 1);  // start
    int numberOfSamples              = N / powf(ratio, currentLevel);  // start

    Vec3*  vertices;
    Vec3*  sample_pos;
    float* distanceArray;
    int*   clusterVertices;

    // Allocate unified memory
    cudaMallocManaged(&sample_pos, numberOfSamples * sizeof(Vec3));
    cudaMallocManaged(&vertices, N * sizeof(Vec3));
    cudaMallocManaged(&distanceArray, N * sizeof(int));
    cudaMallocManaged(&clusterVertices, N * sizeof(int));


    VertexAttributes vertexAttributes(rx);
    auto             context = rx.get_context();

    constexpr uint32_t CUDABlockSize = 512;
    cudaDeviceSynchronize();


    // FPS sampling
    timers.start("Total");
    timers.start("Sampling");

    FPSSampler(rx,
               vertexAttributes,
               ratio,
               N,
               numberOfLevels,
               numberOfSamplesForFirstLevel,
               sample_pos);

    timers.stop("Total");
    timers.stop("Sampling");

    rxmesh::LaunchBox<CUDABlockSize> cb;
    rx.prepare_launch_box(
        {rxmesh::Op::VV}, cb, (void*)cluster_points<float, CUDABlockSize>);


    // clustering
    timers.start("Total");
    // timers.start("Clustering");

    clustering(rx, vertexAttributes, currentLevel, vertices);
    timers.stop("Total");
    // timers.stop("Clustering");


    // move rxmesh cluster data to a normal pointer on device
    int* vertexCluster;
    cudaMallocManaged(&vertexCluster, sizeof(int) * N);

    timers.start("Total");

    rx.for_each_vertex(rxmesh::DEVICE,
                       [vertexAttributes, context, vertexCluster] __device__(
                           const rxmesh::VertexHandle vh) {
                           vertexCluster[context.linear_id(vh)] =
                               vertexAttributes.clustered_vertex(vh, 0);
                       });
    timers.stop("Total");

    cudaDeviceSynchronize();


    // neighbor handling
    int* number_of_neighbors;
    cudaMallocManaged(&number_of_neighbors, numberOfSamples * sizeof(int));
    for (int i = 0; i < numberOfSamples; i++) {
        number_of_neighbors[i] = 0;
    }
    cudaDeviceSynchronize();

    // find number of neighbors on the coarse mesh using RXMesh
    rxmesh::LaunchBox<CUDABlockSize> nn;

    rx.prepare_launch_box(
        {rxmesh::Op::VV},
        nn,
        (void*)findNumberOfCoarseNeighbors<float, CUDABlockSize>);

    // Allocate memory for vertex neighbors
    VertexNeighbors* vertexNeighbors;
    cudaError_t      err = cudaMallocManaged(
        &vertexNeighbors, numberOfSamples * sizeof(VertexNeighbors));

    // launch kernel
    timers.start("Total");
    // timers.start("Neighbors level 1");

    findNumberOfCoarseNeighbors<float, CUDABlockSize>
        <<<nn.blocks, nn.num_threads, nn.smem_bytes_dyn>>>(
            rx.get_context(),
            vertexAttributes.clustered_vertex,
            number_of_neighbors,
            vertexNeighbors);
    cudaDeviceSynchronize();

    rx.for_each_vertex(
        rxmesh::DEVICE,
        [numberOfSamples,
         context,
         vertexNeighbors,
         vertexAttributes,
         number_of_neighbors] __device__(const rxmesh::VertexHandle vh) {
            if (vertexAttributes.clustered_vertex(vh, 0) ==
                vertexAttributes.sample_number(vh, 0)) {
                number_of_neighbors[vertexAttributes.clustered_vertex(vh, 0)] =
                    vertexNeighbors[vertexAttributes.sample_number(vh, 0)]
                        .getNumberOfNeighbors();
            }
        });
    timers.stop("Total");
    // timers.stop("Neighbors level 1");


    cudaDeviceSynchronize();

    // constructing the first operator
    int num_rows = numberOfSamples;  // Set this appropriately

    CSR csr(num_rows, number_of_neighbors, vertexNeighbors, N);

    cudaDeviceSynchronize();  // Ensure data is synchronized before accessing

    std::vector<CSR> operatorsCSR;
    operatorsCSR.push_back(CSR(N));
    CSR firstOperator(N);

    timers.start("Total");
    // timers.start("Constructing level 1");

    createProlongationOperator(csr.num_rows,
                               csr.row_ptr,
                               csr.value_ptr,
                               csr.number_of_neighbors,
                               N,
                               vertexCluster,
                               vertices,
                               sample_pos,
                               firstOperator.value_ptr,
                               firstOperator.data_ptr);
    timers.stop("Total");
    // timers.stop("Constructing level 1");

    cudaDeviceSynchronize();
    prolongationOperatorCSR.push_back(firstOperator);
    prolongationOperatorCSRTranspose.push_back(transposeCSR(firstOperator));

    // render the first level
    // csr.render(sample_pos);

    // set 1st level node data
    VertexData* oldVdata;
    cudaMallocManaged(&oldVdata, sizeof(VertexData) * numberOfSamples);
    setVertexData(rx, context, oldVdata, vertexAttributes);

    // create all the operators for moving between each level
    timers.start("Total");
    // timers.start("Constructing operators without level 1");

    createProlongationOperators(N,
                                numberOfSamples,
                                numberOfLevels,
                                ratio,
                                sample_pos,
                                csr,
                                prolongationOperatorCSR,
                                prolongationOperatorCSRTranspose,
                                oldVdata,
                                distanceArray,
                                vertexCluster);

    timers.stop("Total");
    // timers.stop("Constructing operators without level 1");


    // contruct equations as CSR matrices

    // for now this is hard coded to use the MCF, we can abstract this later to
    // handle different Lx=b
    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, rx.get_num_vertices(), 3);
    setupMCF(rx, A_mat, B_mat);
    A_mat.move(DEVICE, HOST);
    B_mat.move(DEVICE, HOST);
    CSR A_csr(A_mat, A_mat.row_ptr(), A_mat.col_idx(), A_mat.non_zeros());

    // std::cout << "\nRHS:";
    // std::cout << "\n Number of rows of B:" << B_mat.rows();

    // copy the RHS to our CSR structure
    for (int i = 0; i < B_mat.rows(); i++) {
        B_v.vector[i * 3]     = B_mat(i, 0);
        B_v.vector[i * 3 + 1] = B_mat(i, 1);
        B_v.vector[i * 3 + 2] = B_mat(i, 2);
    }

    equationsPerLevel.push_back(A_csr);

    // construct the equations
    timers.start("Total");
    // timers.start("Constructing LHS");

    constructLHS(A_csr,
                 prolongationOperatorCSR,
                 prolongationOperatorCSRTranspose,
                 equationsPerLevel,
                 numberOfLevels,
                 numberOfSamples,
                 ratio);
    timers.stop("Total");
    // timers.stop("Constructing LHS");


    std::cout << "\nTotal prolongation construction time: "
              << timers.elapsed_millis("Total");
    std::cout << "\nSampling time : " << timers.elapsed_millis("Sampling");
    // std::cout << "\nClustering level 1 time: " <<
    // timers.elapsed_millis("Clustering level 1"); std::cout << "\nNeighbors
    // level 1 time: " << timers.elapsed_millis("Neighbors level 1"); std::cout
    // << "\nOperator level 1 construction time: " <<
    // timers.elapsed_millis("Constructing level 1al"); std::cout << "\nAll
    // operators but level 1 construction time: " <<
    // timers.elapsed_millis("Constructing operators without level 1");
    // std::cout << "\nLHS construction time: " <<
    // timers.elapsed_millis("Constructing LHS");

    cudaDeviceSynchronize();

    // timers.stop("Total");


    vertexAttributes.sample_number.move(DEVICE, HOST);
    vertexAttributes.distance.move(DEVICE, HOST);
    vertexAttributes.sample_level_bitmask.move(DEVICE, HOST);
    vertexAttributes.addToPolyscope(rx);  // add vertex attributes to finest
                                          // mesh
}
