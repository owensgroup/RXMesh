set(SOURCE_LIST
    remesh.cu
    remesh_rxmesh.cuh
    util.cuh
    split.cuh
    collapse.cuh
    flip.cuh
    smoothing.cuh
    link_condition.cuh
    ../common/openmesh_trimesh.h
    ../common/openmesh_report.h
)


rxmesh_add_app(Remesh
    SOURCES ${SOURCE_LIST}
    LIBS RXMesh gtest_main OpenMeshCore OpenMeshTools
)
