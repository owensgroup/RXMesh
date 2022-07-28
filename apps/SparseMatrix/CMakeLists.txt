add_executable(SparseMatrix)

set(SOURCE_LIST
    sparse_matrix.cuh
    sparse_matrix_test.cu
)

target_sources(SparseMatrix 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(SparseMatrix PROPERTIES FOLDER "apps")

set_property(TARGET SparseMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "SparseMatrix" FILES ${SOURCE_LIST})

target_link_libraries( SparseMatrix 
    PRIVATE RXMesh_header_lib 
    PRIVATE RXMesh_lib
    PRIVATE gtest_main
)
