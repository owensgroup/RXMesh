#include "gtest/gtest.h"

#include "rxmesh//rxmesh_types.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/import_obj.h"

TEST(RXMeshStatic, ForEach)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(
        import_obj(STRINGIFY(INPUT_DIR) "cube.obj", Verts, Faces, true));


    RXMeshStatic rxmesh_static(Faces, rxmesh_args.quite);

    std::atomic_uint32_t num_v = 0;
    std::atomic_uint32_t num_e = 0;
    std::atomic_uint32_t num_f = 0;

    rxmesh_static.for_each_vertex([&](const VertexHandle vh) { num_v++; });

    rxmesh_static.for_each_edge([&](const EdgeHandle eh) { num_e++; });

    rxmesh_static.for_each_face([&](const FaceHandle fh) { num_f++; });

    EXPECT_EQ(num_v, rxmesh_static.get_num_vertices());

    EXPECT_EQ(num_e, rxmesh_static.get_num_edges());

    EXPECT_EQ(num_f, rxmesh_static.get_num_faces());
}