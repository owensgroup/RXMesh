set(SOURCE_LIST
  geodesic.cu
  geodesic_kernel.cuh
  geodesic_ptp_openmesh.h
  geodesic_ptp_rxmesh.h
  #../common/openmesh_trimesh.h
  #../common/openmesh_report.h
)


rxmesh_add_app(Geodesic
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh CLI11::CLI11 OpenMeshCore OpenMeshTools
)
