#pragma once

#include "rxmesh/rxmesh_static.h"

template <typename T>
void inline render_permutation(rxmesh::RXMeshStatic& rx,
                               std::vector<T>&       h_permute,
                               std::string           name)
{
    assert(h_permute.size() == rx.get_num_vertices());

    using namespace rxmesh;

    auto v_perm = *rx.add_vertex_attribute<int>("Perm", 1, HOST);

    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        v_perm(vh) = h_permute[rx.linear_id(vh)];
    });

#if RXMESH_WITH_POLYSCOPE
    rx.get_polyscope_mesh()->addVertexScalarQuantity(name, v_perm);
#endif

    rx.remove_attribute("Perm");
}