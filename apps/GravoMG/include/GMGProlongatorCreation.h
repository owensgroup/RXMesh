#pragma once



#include "GMGRXMeshKernels.h"

#include "GMGCSR.h"

// Function to compute the projected distance from a point to a triangle
__device__ float projectedDistance(const Eigen::Vector3f& v0,
                                   const Eigen::Vector3f& v1,
                                   const Eigen::Vector3f& v2,
                                   const Eigen::Vector3f& p)
{
    // Compute edges of the triangle
    Eigen::Vector3f edge1 = v1 - v0;
    Eigen::Vector3f edge2 = v2 - v0;

    // Compute the triangle normal
    Eigen::Vector3f normal        = edge1.cross(edge2);
    float           normal_length = normal.norm();

    if (normal_length < 1e-6f) {
        return -1.0f;  // Return -1 to indicate an error
    }

    // Normalize the normal
    normal.normalize();

    // Compute vector from point to the triangle vertex
    Eigen::Vector3f point_to_vertex = p - v0;

    // Project the vector onto the normal
    float distance = point_to_vertex.dot(normal);

    // Return the absolute distance
    return std::fabs(distance);
}


__device__ std::tuple<float, float, float> computeBarycentricCoordinates(
    const Eigen::Vector3f& v0,
    const Eigen::Vector3f& v1,
    const Eigen::Vector3f& v2,
    const Eigen::Vector3f& p)
{
    // Compute edges of the triangle
    Eigen::Vector3f edge1    = v1 - v0;
    Eigen::Vector3f edge2    = v2 - v0;
    Eigen::Vector3f pointVec = p - v0;

    // Compute normal of the triangle
    Eigen::Vector3f normal = edge1.cross(edge2);
    float area2 = normal.squaredNorm();  // Area of the triangle multiplied by 2

    // Compute barycentric coordinates
    float lambda0, lambda1, lambda2;

    // Sub-area with respect to v0
    Eigen::Vector3f normal1 = (v1 - p).cross(v2 - p);
    lambda0                 = normal1.squaredNorm() / area2;

    // Sub-area with respect to v1
    Eigen::Vector3f normal2 = (v2 - p).cross(v0 - p);
    lambda1                 = normal2.squaredNorm() / area2;

    // Sub-area with respect to v2
    Eigen::Vector3f normal3 = (v0 - p).cross(v1 - p);
    lambda2                 = normal3.squaredNorm() / area2;

    // printf("\ncalculated coords are %f, %f, %f", lambda0, lambda1, lambda2);

    // Return the barycentric coordinates
    return std::make_tuple(lambda0, lambda1, lambda2);
}

__device__ void computeBarycentricCoordinates(const Eigen::Vector3f& v0,
                                              const Eigen::Vector3f& v1,
                                              const Eigen::Vector3f& v2,
                                              const Eigen::Vector3f& p,
                                              float&                 a,
                                              float&                 b,
                                              float&                 c)
{
    // Compute edges of the triangle
    Eigen::Vector3f edge1    = v1 - v0;
    Eigen::Vector3f edge2    = v2 - v0;
    Eigen::Vector3f pointVec = p - v0;

    // Compute normal of the triangle
    Eigen::Vector3f normal = edge1.cross(edge2);
    float area2 = normal.squaredNorm();  // Area of the triangle multiplied by 2

    // Compute barycentric coordinates
    float lambda0 = 0, lambda1 = 0, lambda2 = 0;

    // Sub-area with respect to v0
    Eigen::Vector3f normal1 = (v1 - p).cross(v2 - p);
    lambda0                 = normal1.squaredNorm() / area2;

    // Sub-area with respect to v1
    Eigen::Vector3f normal2 = (v2 - p).cross(v0 - p);
    lambda1                 = normal2.squaredNorm() / area2;

    // Sub-area with respect to v2
    Eigen::Vector3f normal3 = (v0 - p).cross(v1 - p);
    lambda2                 = normal3.squaredNorm() / area2;

    a = lambda0;
    b = lambda1;
    c = lambda2;
    // printf("\ncalculated coords are %f, %f, %f", lambda0, lambda1, lambda2);

    // Return the barycentric coordinates
    // return std::make_tuple(lambda0, lambda1, lambda2);
}

void createProlongationOperator(int  numberOfSamples,
                                int* row_ptr,
                                int* value_ptr,
                                int* number_of_neighbors,
                                int  N,

                                int*   clustered_vertex,
                                Vec3*  vertex_pos,
                                Vec3*  sample_pos,
                                float* prolongation_operator)
{
    thrust::device_vector<int> samples(N);
    thrust::sequence(samples.begin(), samples.end());

    thrust::for_each(
        thrust::device,
        samples.begin(),
        samples.end(),
        [=] __device__(int number) {
            // go through every triangle of my cluster
            const int cluster_point = clustered_vertex[number];
            const int start_pointer = row_ptr[clustered_vertex[number]];
            const int end_pointer   = row_ptr[clustered_vertex[number] + 1];

            float                 min_distance = 99999;
            Eigen::Vector3<float> selectedv1{0, 0, 0}, selectedv2{0, 0, 0},
                selectedv3{0, 0, 0};
            const Eigen::Vector3<float> q{vertex_pos[number].x,
                                          vertex_pos[number].y,
                                          vertex_pos[number].z};

            int neighbor                      = 0;
            int selected_neighbor             = 0;
            int neighbor_of_neighbor          = 0;
            int selected_neighbor_of_neighbor = 0;


            for (int i = start_pointer; i < end_pointer; i++) {

                float distance;
                // Get the neighbor vertex
                neighbor = value_ptr[i];  // Assuming col_idx stores column
                                          // indices of neighbors in CSR.

                // Get the range of neighbors for this neighbor
                const int neighbor_start = row_ptr[neighbor];
                const int neighbor_end   = row_ptr[neighbor + 1];

                for (int j = neighbor_start; j < neighbor_end; j++) {
                    neighbor_of_neighbor = value_ptr[j];

                    for (int k = i + 1; k < end_pointer; k++) {
                        if (value_ptr[k] == neighbor_of_neighbor) {


                            Eigen::Vector3<float> v1{
                                sample_pos[cluster_point].x,
                                sample_pos[cluster_point].y,
                                sample_pos[cluster_point].z};
                            Eigen::Vector3<float> v2{sample_pos[neighbor].x,
                                                     sample_pos[neighbor].y,
                                                     sample_pos[neighbor].z};
                            Eigen::Vector3<float> v3{
                                sample_pos[neighbor_of_neighbor].x,
                                sample_pos[neighbor_of_neighbor].y,
                                sample_pos[neighbor_of_neighbor].z};

                            // find distance , if less than min dist, find bary
                            // coords, save them
                            float distance = projectedDistance(v1, v2, v3, q);
                            if (distance < min_distance) {

                                min_distance      = distance;
                                selectedv1        = v1;
                                selectedv2        = v2;
                                selectedv3        = v3;
                                selected_neighbor = neighbor;
                                selected_neighbor_of_neighbor =
                                    neighbor_of_neighbor;
                            }
                        }
                    }
                }
            }


            /*
            printf("\n%d %f %f %f Selected: %d %d %d",
                       number,
                       vertex_pos[number].x,
                       vertex_pos[number].y,
                       vertex_pos[number].z,
                       cluster_point,
                       selected_neighbor,
                       selected_neighbor_of_neighbor);
              */

            float b1 = 0, b2 = 0, b3 = 0;
            computeBarycentricCoordinates(
                selectedv1, selectedv2, selectedv3, q, b1, b2, b3);


            // put it inside prolongation row, it will be unique so no race
            // condition
            int l = number;

            // printf("\n %d final coords: %f %f %f", l, b1, b2, b3);


            prolongation_operator[l * numberOfSamples + cluster_point]     = b1;
            prolongation_operator[l * numberOfSamples + selected_neighbor] = b2;
            prolongation_operator[l * numberOfSamples +
                                  selected_neighbor_of_neighbor]           = b3;
        });
}
