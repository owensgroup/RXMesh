set(SOURCE_LIST
    neo_hookean.cu
    inertial_energy.h
    gravity_energy.h
    barrier_energy.h
    draw.h
    init.h
    neo_hookean_energy.h
    bending_energy.h
    scene_config.h
    naive_hessian_update.h
)

rxmesh_add_app(NeoHookean
    SOURCES ${SOURCE_LIST}
    LIBS RXMesh
)
