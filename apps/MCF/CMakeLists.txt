set(SOURCE_LIST
    mcf.cu
    mcf_kernels.cuh
    mcf_cg.h
    mcf_cg_mat_free.h
    mcf_chol.h
    mcf_gmg.h
    mcf_eigen.h
    mcf_cudss.h
)

rxmesh_add_app(MCF
    SOURCES ${SOURCE_LIST}
    LIBS RXMesh CLI11::CLI11
)
if(${RX_USE_SUITESPARSE})
    target_link_libraries(MCF PRIVATE SuiteSparse::CHOLMOD)
endif()
