set(SOURCE_LIST
  filtering_util.h
  filtering_rxmesh_kernel.cuh
  filtering.cu
  filtering_openmesh.h
  filtering_rxmesh.cuh
  #../common/openmesh_trimesh.h
  #../common/openmesh_report.h
)


rxmesh_add_app(Filtering
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh gtest_main OpenMeshCore OpenMeshTools
)
