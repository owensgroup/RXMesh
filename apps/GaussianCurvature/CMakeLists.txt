set(SOURCE_LIST
  gaussian_curvature.cu
  gaussian_curvature_ref.h
  gaussian_curvature_kernel.cuh
)

rxmesh_add_app(GaussianCurvature
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh gtest_main
)
