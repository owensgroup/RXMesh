
#include "include/NeighborHandling.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "include/GMGProlongatorCreation.h"

#include "include/VCycle.h"
#include "include/interactive.h"

#include "include/RXMeshMCFSetup.h"
#include "rxmesh/geometry_factory.h"

class GPUGMG
{
public:
    float     ratio;
    int       N;
    int       numberOfLevels;
    std::vector<CSR> prolongationOperatorCSR;
    std::vector<CSR> equationsPerLevel;
    VectorCSR3D      B_v;

    GPUGMG()
    {
        ratio          = 3;
        numberOfLevels = 0;
    }
    GPUGMG(RXMeshStatic& rx)
        : B_v(rx.get_num_vertices())
    {
        N              = rx.get_num_vertices();

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

void GPUGMG::ConstructOperators(RXMeshStatic &rx)
{

    //set up initial variables

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


     VertexAttributesRXMesh vertexAttributes(rx);
    auto                   context = rx.get_context();

    constexpr uint32_t CUDABlockSize = 512;
    cudaDeviceSynchronize();


    //FPS sampling
    timers.start("Total");
    timers.start("Sampling");

    sampler(rx,
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


    //clustering
    timers.start("Total");
    //timers.start("Clustering");

    clusteringRXMesh(rx, vertexAttributes, currentLevel, vertices);
    timers.stop("Total");
    //timers.stop("Clustering");


    //move rxmesh cluster data to a normal pointer on device
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


    //neighbor handling
    int* number_of_neighbors;
    cudaMallocManaged(&number_of_neighbors, numberOfSamples * sizeof(int));
    for (int i = 0; i < numberOfSamples; i++) {
        number_of_neighbors[i] = 0;
    }
    cudaDeviceSynchronize();

    //find number of neighbors on the coarse mesh using RXMesh
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
    //timers.start("Neighbors level 1");

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
    //timers.stop("Neighbors level 1");

        
    cudaDeviceSynchronize();

    //constructing the first operator 
    int num_rows = numberOfSamples;  // Set this appropriately

    CSR csr(num_rows, number_of_neighbors, vertexNeighbors, N);

    cudaDeviceSynchronize();  // Ensure data is synchronized before accessing

    std::vector<CSR> operatorsCSR;
    operatorsCSR.push_back(CSR(N));
    CSR firstOperator(N);

    timers.start("Total");
    //timers.start("Constructing level 1");

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
    //timers.stop("Constructing level 1");

    cudaDeviceSynchronize();
    prolongationOperatorCSR.push_back(firstOperator);

    //render the first level
    //csr.render(sample_pos);

    // set 1st level node data
    VertexData* oldVdata;
    cudaMallocManaged(&oldVdata, sizeof(VertexData) * numberOfSamples);
    setVertexData(rx, context, oldVdata, vertexAttributes);

    //create all the operators for moving between each level
    timers.start("Total");
    //timers.start("Constructing operators without level 1");

    createProlongationOperators(N,
                                numberOfSamples,
                                numberOfLevels,
                                ratio,
                                sample_pos,
                                csr,
                                prolongationOperatorCSR,
                                oldVdata,
                                distanceArray,
                                vertexCluster);

    timers.stop("Total");
    //timers.stop("Constructing operators without level 1");


    // contruct equations as CSR matrices

    //for now this is hard coded to use the MCF, we can abstract this later to handle different Lx=b
    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, rx.get_num_vertices(), 3);
    setupMCF(rx, A_mat, B_mat);
    A_mat.move(DEVICE, HOST);
    B_mat.move(DEVICE, HOST);
    CSR A_csr(A_mat, A_mat.row_ptr(), A_mat.col_idx(), A_mat.non_zeros());

    //std::cout << "\nRHS:";
    //std::cout << "\n Number of rows of B:" << B_mat.rows();

    //copy the RHS to our CSR structure
    for (int i = 0; i < B_mat.rows(); i++) {
        B_v.vector[i * 3]     = B_mat(i, 0);
        B_v.vector[i * 3 + 1] = B_mat(i, 1);
        B_v.vector[i * 3 + 2] = B_mat(i, 2);
    }
   
    equationsPerLevel.push_back(A_csr);

    //construct the equations
    timers.start("Total");
    //timers.start("Constructing LHS");

    constructLHS(A_csr,
                 prolongationOperatorCSR,
                 equationsPerLevel,
                 numberOfLevels,
                 numberOfSamples,
                 ratio);
    timers.stop("Total");
    //timers.stop("Constructing LHS");



    std::cout << "\nTotal prolongation construction time: " << timers.elapsed_millis("Total");
    std::cout << "\nSampling time : " << timers.elapsed_millis("Sampling");
    //std::cout << "\nClustering level 1 time: " << timers.elapsed_millis("Clustering level 1");
    //std::cout << "\nNeighbors level 1 time: " << timers.elapsed_millis("Neighbors level 1");
    //std::cout << "\nOperator level 1 construction time: " << timers.elapsed_millis("Constructing level 1al");
    //std::cout << "\nAll operators but level 1 construction time: " << timers.elapsed_millis("Constructing operators without level 1");
    //std::cout << "\nLHS construction time: " << timers.elapsed_millis("Constructing LHS");
                 
    cudaDeviceSynchronize();

    //timers.stop("Total");

    
    vertexAttributes.sample_number.move(DEVICE, HOST);
    vertexAttributes.distance.move(DEVICE, HOST);
    vertexAttributes.sample_level_bitmask.move(DEVICE, HOST);
    vertexAttributes.addToPolyscope(rx);//add vertex attributes to finest mesh
}


int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    //RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");
    //RXMeshStatic rx(STRINGIFY(INPUT_DIR) "torus.obj");
    //RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bunnyhead.obj");
    //RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bumpy-cube.obj");
    //RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");
    //RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere1.obj");
    //RXMeshStatic rx(STRINGIFY(INPUT_DIR) "rocker-arm.obj");
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "nefertiti.obj");
    //RXMeshStatic rx(STRINGIFY(INPUT_DIR) "beetle.obj");

    /*
    std::vector<std::vector<float>>    planeVerts;
    std::vector<std::vector<uint32_t>> planeFaces;
    uint32_t                           nx = 10;
    uint32_t                           ny = 10;
    create_plane(planeVerts, planeFaces, nx, ny);
    RXMeshStatic rx(planeFaces);
    rx.add_vertex_coordinates(planeVerts, "plane");
    */

    GPUGMG g(rx);
    g.ConstructOperators(rx);


    //for (int i = 0;i < g.equationsPerLevel.size(); i++)
    //    g.equationsPerLevel[i].printCSR();

    GMGVCycle gmg(g.N);
    
    gmg.prolongationOperators = g.prolongationOperatorCSR;
    gmg.LHS                   = g.equationsPerLevel;
    gmg.RHS                   = g.B_v;
    gmg.max_number_of_levels  = 0;
    gmg.post_relax_iterations = 5;
    gmg.pre_relax_iterations  = 5;
    gmg.ratio                 = g.ratio;
    //std::cout << "\nNumber of equations LHS:" << gmg.LHS.size();
    //std::cout << "\nNumber of operators:" << gmg.prolongationOperators.size();
    //std::cout << "\nMax level:" << gmg.max_number_of_levels;
    
    //gmg.solve();

    //rendering and interactive
    Timers<GPUTimer> timers;
    timers.add("solve");


    std::vector<std::array<double, 3>> vertexMeshPositions;
    vertexMeshPositions.resize(gmg.X.n);

    auto polyscope_callback = [&]() mutable {
        ImGui::Begin("GMG Parameters");

        ImGui::InputInt("Number of Levels", &gmg.max_number_of_levels);
        ImGui::InputInt("Number of V cycles", &gmg.numberOfCycles);
        ImGui::InputInt("Number of pre solve smoothing iterations",
                        &gmg.pre_relax_iterations);
        ImGui::InputInt("Number of post solve smoothing iterations",
                        &gmg.post_relax_iterations);
        ImGui::InputInt("Number of direct solve iterations",
                        &gmg.directSolveIterations);
        ImGui::SliderFloat("Omega", &gmg.omega, 0.0, 1.0);


        if (ImGui::Button("Run V Cycles again")) {
            std::cout<< "\n---------------NEW SOLVE INITIATED--------------------\n";

            gmg.X.reset();
            //start measuring time from here

            timers.start("solve");
            gmg.solve();
            timers.stop("solve");

            std::cout << "\nSolving time: " << timers.elapsed_millis("solve");

            // renderOutputMesh(gmg.X, rx);
            
            for (int i = 0; i < gmg.X.n; i++) {
                vertexMeshPositions[i] = {gmg.X.vector[3 * i],
                                          gmg.X.vector[3 * i + 1],
                                          gmg.X.vector[3 * i + 2]};
            }
            polyscope::registerSurfaceMesh("output mesh",
                                           vertexMeshPositions,
                                           rx.get_polyscope_mesh()->faces);
        }

        ImGui::End();
    };

    polyscope::state::userCallback = polyscope_callback;



#if USE_POLYSCOPE
    polyscope::show();
#endif
}



    ////////////////////////////////////
/*
std::cout << "\n\n\n\n\n\n";
std::cout << "first\n";

CSR t1(3, 3);

VectorCSR3D vectorTest(3);
VectorCSR3D vectorResult(3);

for (int i=0;i<3;i++) {
    vectorTest.vector[i*3] = i+1;
    vectorTest.vector[i*3 + 1] = i + 1;
    vectorTest.vector[i*3+2] = i+1;
}

SpMV_CSR_3D(t1.row_ptr,
         t1.value_ptr,
         t1.data_ptr,
         vectorTest.vector,
         vectorResult.vector,
         3);

std::cout << "\nVector result: ";
for (int i=0;i<vectorResult.n;i++) {
    std::cout << "\n";
    std::cout << vectorResult.vector[i*3] << " ";
    std::cout << vectorResult.vector[i*3+1] << " ";
    std::cout << vectorResult.vector[i*3+2] << " ";
}
*/

    /*
std::cout << "\n\n\n\n\n\n";
std::cout << "first\n";

CSR t1(3, 3,3);

VectorCSR3D b(3);
VectorCSR3D vectorResult(3);
for (int i = 0; i < vectorResult.n * 3; i++) {
    vectorResult.vector[i] = 0.0f;
}
// Set right hand side - different for each component of each point
b.vector[0] = 4.0;  // Point 1 (x,y,z)
b.vector[1] = 2.0;
b.vector[2] = 1.0;

b.vector[3] = 3.0;  // Point 2 (x,y,z)
b.vector[4] = 5.0;
b.vector[5] = 2.0;

b.vector[6] = 1.0;  // Point 3 (x,y,z)
b.vector[7] = 2.0;
b.vector[8] = 6.0;


t1.printCSR();

gauss_jacobi_CSR_3D(t1, vectorResult.vector, b.vector, 50);

std::cout << "\nVector result: ";
for (int i = 0; i < vectorResult.n; i++) {
    std::cout << "\n";
    std::cout << vectorResult.vector[i * 3] << " ";
    std::cout << vectorResult.vector[i * 3 + 1] << " ";
    std::cout << vectorResult.vector[i * 3 + 2] << " ";
}

*/
//////////////////////////////////////////////