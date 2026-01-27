set(SOURCE_LIST
  test_all_permutations.cu
  count_nnz_fillin.h
  render_permutation.h
)

rxmesh_add_app(NDReorder
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh CLI11::CLI11 metis Eigen3::Eigen
)
