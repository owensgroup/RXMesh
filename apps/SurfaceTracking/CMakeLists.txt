set(SOURCE_LIST
    tracking.cu
    tracking_rxmesh.cuh
    tracking_kernels.cuh
    flipper.cuh
    splitter.cuh
    smoother.cuh
    frame_stepper.h
    simulation.h
    noise.h
    collapser.cuh
    link_condition.cuh
    util.cuh
)

rxmesh_add_app(SurfaceTracking
    SOURCES ${SOURCE_LIST}
    LIBS RXMesh gtest_main
)
