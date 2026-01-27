set(SOURCE_LIST
    mass_spring.cu
    boundary_condition.h
    inertial_energy.h
    mass_spring_energy.h
    gravity_energy.h
    barrier_energy.h
    draw.h
)

rxmesh_add_app(MassSpring
    SOURCES ${SOURCE_LIST}
    LIBS RXMesh CLI11::CLI11
)
