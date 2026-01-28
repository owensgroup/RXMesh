set(SOURCE_LIST
  sec.cu
  sec_rxmesh.cuh
  sec_kernels.cuh
  link_condition.cuh
  #../common/openmesh_trimesh.h
  #../common/openmesh_report.h
)


rxmesh_add_app(SECHistogram
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh CLI11::CLI11 OpenMeshCore OpenMeshTools
)
