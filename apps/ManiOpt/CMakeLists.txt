set(SOURCE_LIST
  mani_opt.cu
  mean_curv.h
)

rxmesh_add_app(ManiOpt
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh
)
