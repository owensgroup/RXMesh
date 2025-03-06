#pragma once
#include "rxmesh/query.cuh"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/util/bitmask_util.h"


using namespace rxmesh;

struct Vec3
{
    float x, y, z;
};

// used from 2nd level onwards
struct VertexData
{
    int     cluster;
    float   distance;
    uint8_t bitmask;
    Vec3    position;
    int     sample_number;
    int     linear_id;
};


struct VertexAttributesRXMesh
{
    VertexAttribute<float>    vertex_pos;
    VertexAttribute<int>      sample_number;
    VertexAttribute<float>    distance;
    VertexAttribute<uint16_t> sample_level_bitmask;
    VertexAttribute<int>      clustered_vertex;
    VertexAttribute<int>      number_of_neighbors_coarse;

    VertexAttributesRXMesh(RXMeshStatic& rx)
    {
        vertex_pos    = *rx.get_input_vertex_coordinates();
        sample_number = *rx.add_vertex_attribute<int>("sample_number", 1);
        distance      = *rx.add_vertex_attribute<float>("distance", 1);
        sample_level_bitmask = *rx.add_vertex_attribute<uint16_t>("bitmask", 1);
        clustered_vertex     = *rx.add_vertex_attribute<int>("clustering", 1);
        number_of_neighbors_coarse =
            *rx.add_vertex_attribute<int>("number of neighbors", 1);
    }

    void addToPolyscope(RXMeshStatic& rx)
    {
        rx.get_polyscope_mesh()->addVertexScalarQuantity(
            "sample_number", sample_number);
        rx.get_polyscope_mesh()->addVertexScalarQuantity(
            "distance", distance);
        rx.get_polyscope_mesh()->addVertexScalarQuantity(
            "sample_level_bitmask", sample_level_bitmask);
        rx.get_polyscope_mesh()->addVertexScalarQuantity(
            "clusterPoint", clustered_vertex);
        rx.get_polyscope_mesh()->addVertexScalarQuantity(
            "number of neighbors", number_of_neighbors_coarse);
    }
};

class VertexNeighbors;

template <typename T, uint32_t blockThreads>
__global__ static void findNumberOfCoarseNeighbors(
    const rxmesh::Context        context,
    rxmesh::VertexAttribute<int> clustered_vertices,
    int*                         number_of_neighbors,
    VertexNeighbors*             vns)
{

    auto cluster = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {

            if (clustered_vertices(v_id, 0) != clustered_vertices(vv[i], 0)) {
                int a = clustered_vertices(vv[i], 0);
                int b = clustered_vertices(v_id, 0);

                vns[b].addNeighbor(a);


                // neighbor adding logic here where we say that b is a neighbor
                // of a
            }
        }
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, cluster);
}

/**
 * \brief Clustering after sampling via thrust
 * \tparam T 
 * \tparam blockThreads 
 * \param context 
 * \param vertex_pos 
 * \param distance 
 * \param clustered_vertices 
 * \param flag 
 */
template <typename T, uint32_t blockThreads>
__global__ static void cluster_points(
    const rxmesh::Context        context,
    rxmesh::VertexAttribute<T>   vertex_pos,
    rxmesh::VertexAttribute<T>   distance,
    rxmesh::VertexAttribute<int> clustered_vertices,
    int*                         flag)
{

    auto cluster = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {

            float dist =
                sqrtf(powf(vertex_pos(v_id, 0) - vertex_pos(vv[i], 0), 2) +
                      powf(vertex_pos(v_id, 1) - vertex_pos(vv[i], 1), 2) +
                      powf(vertex_pos(v_id, 2) - vertex_pos(vv[i], 2), 2)) +
                distance(vv[i], 0);


            if (dist < distance(v_id, 0) &&
                clustered_vertices(vv[i], 0) != -1) {
                distance(v_id, 0)           = dist;
                *flag                       = 15;
                clustered_vertices(v_id, 0) = clustered_vertices(vv[i], 0);
            }
        }
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, cluster);
}


/**
 * \brief FPS Sampling via thrust
 * \tparam T 
 * \tparam blockThreads 
 * \param context 
 * \param vertex_pos 
 * \param distance 
 * \param flag 
 */
template <typename T, uint32_t blockThreads>
__global__ static void sample_points(const rxmesh::Context      context,
                                     rxmesh::VertexAttribute<T> vertex_pos,
                                     rxmesh::VertexAttribute<T> distance,
                                     int*                       flag)
{

    auto sampler = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {

            float dist =
                sqrtf(powf(vertex_pos(v_id, 0) - vertex_pos(vv[i], 0), 2) +
                      powf(vertex_pos(v_id, 1) - vertex_pos(vv[i], 1), 2) +
                      powf(vertex_pos(v_id, 2) - vertex_pos(vv[i], 2), 2)) +
                distance(vv[i], 0);

            // printf("\nVertex: %u Distance : %f", context.linear_id(v_id),
            // dist);


            if (dist < distance(v_id, 0)) {
                distance(v_id, 0) = dist;
                *flag             = 15;
            }
        }
        // printf("\nFLAG : %d", *flag);
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, sampler);
}


/**
 * \brief Clustering the data after sampling via RXMesh
 * \param rx 
 * \param vertexAttributes 
 * \param currentLevel 
 * \param vertices 
 */
void clusteringRXMesh(RXMeshStatic&           rx,
                      VertexAttributesRXMesh& vertexAttributes,
                      int                     currentLevel,
                      Vec3*                   vertices)
{

    int* flagger;
    cudaMallocManaged(&flagger, sizeof(int));
    *flagger                                       = 0;
    const Context                    context       = rx.get_context();
    constexpr uint32_t               CUDABlockSize = 512;
    rxmesh::LaunchBox<CUDABlockSize> lb;
    rx.prepare_launch_box(
        {rxmesh::Op::VV}, lb, (void*)sample_points<float, CUDABlockSize>);

    // clustering step
    int j = 0;
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [vertexAttributes, currentLevel, context, vertices] __device__(
            const rxmesh::VertexHandle vh) {
            vertices[context.linear_id(vh)].x =
                vertexAttributes.vertex_pos(vh, 0);
            vertices[context.linear_id(vh)].y =
                vertexAttributes.vertex_pos(vh, 1);
            vertices[context.linear_id(vh)].z =
                vertexAttributes.vertex_pos(vh, 2);

            // if (sample_number(vh, 0) > -1)
            if ((vertexAttributes.sample_level_bitmask(vh, 0) &
                 (1 << (currentLevel - 1))) != 0) {
                vertexAttributes.clustered_vertex(vh, 0) =
                    vertexAttributes.sample_number(vh, 0);
                vertexAttributes.distance(vh, 0) = 0;
            } else {
                vertexAttributes.distance(vh, 0)         = INFINITY;
                vertexAttributes.clustered_vertex(vh, 0) = -1;
            }
        });

    do {
        cudaDeviceSynchronize();
        *flagger = 0;
        cluster_points<float, CUDABlockSize>
            <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                rx.get_context(),
                vertexAttributes.vertex_pos,
                vertexAttributes.distance,
                vertexAttributes.clustered_vertex,
                flagger);
        cudaDeviceSynchronize();
        j++;
    } while (*flagger != 0);

    //vertexAttributes.clustered_vertex.move(DEVICE, HOST);
    //std::cout << "\Clustering iterations: " << j;
}


/**
 * \brief Set vertex data from RXMesh to standard pointer data for processing further multigrid levels. This also reindexes the data according to sample number
 * \param rx 
 * \param context 
 * \param oldVdata 
 * \param vertexAttribute 
 */
void setVertexData(RXMeshStatic&          rx,
                   Context&               context,
                   VertexData*            oldVdata,
                   VertexAttributesRXMesh vertexAttribute)
{
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [oldVdata, vertexAttribute, context] __device__(
            const rxmesh::VertexHandle vh) {
            if (vertexAttribute.sample_number(vh, 0) != -1) {
                // printf("\nputting data for sample %d", sample_number(vh, 0));

                oldVdata[vertexAttribute.sample_number(vh, 0)].distance = 0;
                oldVdata[vertexAttribute.sample_number(vh, 0)].linear_id =
                    context.linear_id(vh);
                oldVdata[vertexAttribute.sample_number(vh, 0)].sample_number =
                    vertexAttribute.sample_number(vh, 0);
                oldVdata[vertexAttribute.sample_number(vh, 0)].bitmask =
                    vertexAttribute.sample_level_bitmask(vh, 0);
                oldVdata[vertexAttribute.sample_number(vh, 0)].position.x =
                    vertexAttribute.vertex_pos(vh, 0);
                oldVdata[vertexAttribute.sample_number(vh, 0)].position.y =
                    vertexAttribute.vertex_pos(vh, 1);
                oldVdata[vertexAttribute.sample_number(vh, 0)].position.z =
                    vertexAttribute.vertex_pos(vh, 2);
                oldVdata[vertexAttribute.sample_number(vh, 0)].cluster =
                    vertexAttribute.clustered_vertex(vh, 0);
            }
        });
}

/**
 * \brief For FPS Sampling in parallel
 * \param rx 
 * \param vertexAttributes 
 * \param ratio 
 * \param N 
 * \param numberOfLevels 
 * \param numberOfSamplesForFirstLevel 
 * \param sample_pos 
 */
void FPSSampler(RXMeshStatic&          rx,
                VertexAttributesRXMesh vertexAttributes,
                float                  ratio,
                int                    N,
                int                    numberOfLevels,
                int                    numberOfSamplesForFirstLevel,
                Vec3*                  sample_pos)
{

    std::random_device rd;
    // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(
        rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dist(0, N - 1);
    // From 0 to (number of points - 1)
    int seed = 0;  // dist(gen);

    //std::cout << "\nSeed: " << seed;

    VertexReduceHandle<float>              reducer(vertexAttributes.distance);
    cub::KeyValuePair<VertexHandle, float> farthestPoint;

    int* flagger;
    cudaMallocManaged(&flagger, sizeof(int));
    *flagger = 0;

    const Context                    context       = rx.get_context();
    constexpr uint32_t               CUDABlockSize = 512;
    rxmesh::LaunchBox<CUDABlockSize> lb;
    rx.prepare_launch_box(
        {rxmesh::Op::VV}, lb, (void*)sample_points<float, CUDABlockSize>);

    int j                  = 0;
    int currentSampleLevel = numberOfLevels;
    
   /* std::cout << "levels:" << numberOfLevels;

    for (int q = 0; q < numberOfLevels; q++) {
        std::cout << "\n  level " << q << " : " << N / powf(ratio, q);
    }*/
    for (int i = 0; i < numberOfSamplesForFirstLevel; i++) {
        if (i == N / (int)powf(ratio, currentSampleLevel)) {
            currentSampleLevel--;
            //std::cout << "\nNext sample level: " << currentSampleLevel;
        }

        rx.for_each_vertex(
            rxmesh::DEVICE,
            [seed,
             context,
             i,
             currentSampleLevel,
             sample_pos,
             vertexAttributes] __device__(const rxmesh::VertexHandle vh) {
                if (seed == context.linear_id(vh)) {
                    vertexAttributes.sample_number(vh, 0) = i;
                    // sample_number_point
                    vertexAttributes.distance(vh, 0) = 0;
                    sample_pos[i].x = vertexAttributes.vertex_pos(vh, 0);
                    sample_pos[i].y = vertexAttributes.vertex_pos(vh, 1);
                    sample_pos[i].z = vertexAttributes.vertex_pos(vh, 2);

                    for (int k = 0; k < currentSampleLevel; k++) {
                        vertexAttributes.sample_level_bitmask(vh, 0) |=
                            (1 << k);
                    }
                } else {
                    if (i == 0) {
                        vertexAttributes.distance(vh, 0)      = INFINITY;
                        vertexAttributes.sample_number(vh, 0) = -1;
                    }
                }
            });

        do {
            cudaDeviceSynchronize();
            *flagger = 0;
            sample_points<float, CUDABlockSize>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    rx.get_context(),
                    vertexAttributes.vertex_pos,
                    vertexAttributes.distance,
                    flagger);
            cudaDeviceSynchronize();
            j++;

        } while (*flagger != 0);


        // reduction step
        farthestPoint = reducer.arg_max(vertexAttributes.distance, 0);
        seed          = rx.linear_id(farthestPoint.key);
    }

    //std::cout << "\nSampling iterations: " << j;


}
