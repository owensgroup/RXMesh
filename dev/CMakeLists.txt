set(REQUIRED_CMAKE_VERSION "3.9")

################################################################################
# CLI11
include(cli11)
include(eigen)
include(libigl)

#Create the benchmark library
# Create utils library
add_library(rxmesh_dev_solvers
        LinSysSolvers/LinSysSolver.cpp
        LinSysSolvers/CHOLMODSolver.cpp
)

target_include_directories(rxmesh_dev_solvers PUBLIC LinSysSolvers)
target_include_directories(rxmesh_dev_solvers PUBLIC Reordering)

target_link_libraries(rxmesh_dev_solvers PUBLIC Eigen3::Eigen)
target_link_libraries(rxmesh_dev_solvers PUBLIC ${PROJECT_NAME}_lib)


# Testing Example Creation
add_executable(${PROJECT_NAME}_benchmark benchmark.cpp)
target_link_libraries(${PROJECT_NAME}_benchmark PUBLIC rxmesh_dev_solvers)
target_link_libraries(${PROJECT_NAME}_benchmark PUBLIC CLI11::CLI11)


#include eigen
include(eigen)
target_link_libraries(${PROJECT_NAME}_benchmark PUBLIC Eigen3::Eigen)


#include libigl
include(libigl)
target_link_libraries(${PROJECT_NAME}_benchmark PUBLIC igl::core)
# add spdlog logging
target_link_libraries(${PROJECT_NAME}_benchmark PUBLIC spdlog::spdlog)