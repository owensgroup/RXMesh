
set(REQUIRED_CMAKE_VERSION "3.9")

################################################################################
# CLI11
include(cli11)


# Testing Example Creation
add_executable(${PROJECT_NAME}_sepComp_runtime sepComp_runtime.cpp)
target_link_libraries(${PROJECT_NAME}_sepComp_runtime PUBLIC parth)
target_link_libraries(${PROJECT_NAME}_sepComp_runtime PUBLIC CLI11::CLI11)


#include eigen
include(eigen)
target_link_libraries(${PROJECT_NAME}_sepComp_runtime PUBLIC Eigen3::Eigen)


#include libigl
include(libigl)
target_link_libraries(${PROJECT_NAME}_sepComp_runtime PUBLIC igl::core)