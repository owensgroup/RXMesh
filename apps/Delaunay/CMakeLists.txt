set(SOURCE_LIST
  delaunay_edge_flip.cu
  delaunay_rxmesh.cuh
  mcf_rxmesh.h
  mcf_rxmesh_kernel.cuh
  #../common/openmesh_trimesh.h
  #../common/openmesh_report.h
)

rxmesh_add_app(DelaunayEdgeFlip
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh OpenMeshCore OpenMeshTools CLI11::CLI11
)
