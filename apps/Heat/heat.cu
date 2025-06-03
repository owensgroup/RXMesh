#include "rxmesh/query.cuh"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.h"

using namespace rxmesh;


int main(int argc, char** argv)
{
    rx_init(0);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

#if USE_POLYSCOPE
    polyscope::show();
#endif
}