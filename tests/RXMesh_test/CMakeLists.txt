if(USE_HIP)
    # HIP test subset: the mesh data-structure + query + dynamic core (the
    # warp-cooperative surface this port targets). Excluded on HIP:
    #   - test_solver / test_sparse_matrix / test_jacobian: need the cusolverSp
    #     sparse direct-solver API, which has no hipSOLVER equivalent.
    #   - test_hess / test_scalar / test_grad.h: autodiff Newton path on the
    #     above solvers.
    #   - test_attribute / test_indices: use the for_each(DEVICE, lambda)
    #     device-iteration API, which HIP/clang cannot gate (no extended-lambda
    #     closure trait).
    #   - test_multiple_meshes: needs get_polyscope_mesh() (polyscope is OFF).
    set(SOURCE_LIST
        rxmesh_test_main.cu
        rxmesh_test.h
        test_util.cu
        test_iterator.cu
        test_queries.h
        test_queries_oriented.cu
        test_higher_queries.cu
        query_kernel.cuh
        higher_query.cuh
        test_for_each.cu
        test_validate.cu
        test_lp_pair.cu
        test_dynamic.cu
        test_triangle_refinement.cu
        test_ev_diamond.cu
        test_patch_lock.cuh
        test_patch_scheduler.cuh
        test_patch_slicing.cu
        test_multi_queries.cu
        test_wasted_work.cuh
        test_boundary.cu
        test_export.cu
        test_tet.cu
    )
else()
set(SOURCE_LIST
    rxmesh_test_main.cu
	rxmesh_test.h
    test_attribute.cu
    test_util.cu
	test_iterator.cu
    test_queries.h
	test_queries_oriented.cu
	test_higher_queries.cu
	query_kernel.cuh
	higher_query.cuh
	test_for_each.cu
	test_validate.cu
	test_lp_pair.cu
	test_dynamic.cu
	test_triangle_refinement.cu
	test_ev_diamond.cu
	test_sparse_matrix.cu
	test_patch_lock.cuh
	test_patch_scheduler.cuh
	test_patch_slicing.cu
	test_multi_queries.cu
	test_wasted_work.cuh
	test_eigen.cu
	test_boundary.cu
	test_dense_matrix.cu
	test_export.cu
	test_svd.cu
	test_scalar.cu
	test_grad.h
	test_solver.cu
	test_hess.cu
	test_tet.cu
	test_indices.cu
	test_jacobian.cu
	test_multiple_meshes.cu
)
endif()

rxmesh_add_app(RXMesh_test
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh gtest_main CLI11::CLI11
  FOLDER "tests"
)