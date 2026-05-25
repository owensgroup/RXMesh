include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(RXMESH_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}"
    CACHE STRING "RXMesh public header install directory")
set(RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}"
    CACHE STRING "RXMesh third-party header install directory")
set(RXMESH_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}"
    CACHE STRING "RXMesh library install directory")
set(RXMESH_INSTALL_BINDIR "${CMAKE_INSTALL_BINDIR}"
    CACHE STRING "RXMesh runtime install directory")
set(RXMESH_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/RXMesh"
    CACHE STRING "RXMesh CMake package install directory")
set(RXMESH_INSTALL_COMPONENT ""
    CACHE STRING "Optional install component used for RXMesh install rules")

set(_RXMESH_INSTALL_COMPONENT_ARGS)
if(NOT "${RXMESH_INSTALL_COMPONENT}" STREQUAL "")
    list(APPEND _RXMESH_INSTALL_COMPONENT_ARGS COMPONENT "${RXMESH_INSTALL_COMPONENT}")
endif()

function(rxmesh_install_directory_if_exists source_dir destination_dir)
    if(EXISTS "${source_dir}")
        install(DIRECTORY "${source_dir}"
                DESTINATION "${destination_dir}"
                ${_RXMESH_INSTALL_COMPONENT_ARGS})
    endif()
endfunction()

function(rxmesh_install_target_if_exists target_name)
    if(TARGET "${target_name}")
        install(TARGETS "${target_name}"
                ARCHIVE DESTINATION "${RXMESH_INSTALL_LIBDIR}"
                LIBRARY DESTINATION "${RXMESH_INSTALL_LIBDIR}"
                RUNTIME DESTINATION "${RXMESH_INSTALL_BINDIR}"
                ${_RXMESH_INSTALL_COMPONENT_ARGS})
    endif()
endfunction()

rxmesh_install_target_if_exists(RXMesh)
rxmesh_install_target_if_exists(GKlib)
rxmesh_install_target_if_exists(metis)
rxmesh_install_target_if_exists(polyscope)
rxmesh_install_target_if_exists(imgui)
rxmesh_install_target_if_exists(glad)
rxmesh_install_target_if_exists(glfw)
rxmesh_install_target_if_exists(stb)
rxmesh_install_target_if_exists(glm)

rxmesh_install_directory_if_exists("${RXMESH_SOURCE_DIR}/include/rxmesh"
                                   "${RXMESH_INSTALL_INCLUDEDIR}")

rxmesh_install_directory_if_exists("${rapidjson_SOURCE_DIR}/include/"
                                   "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
rxmesh_install_directory_if_exists("${rapidobj_SOURCE_DIR}/include/"
                                   "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
rxmesh_install_directory_if_exists("${spdlog_SOURCE_DIR}/include/"
                                   "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
rxmesh_install_directory_if_exists("${cereal_SOURCE_DIR}/include/"
                                   "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
rxmesh_install_directory_if_exists("${glm_SOURCE_DIR}/glm"
                                   "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
rxmesh_install_directory_if_exists("${cubql_SOURCE_DIR}/include/"
                                   "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
rxmesh_install_directory_if_exists("${cubql_SOURCE_DIR}/cuBQL"
                                   "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
rxmesh_install_directory_if_exists("${metis_SOURCE_DIR}/include/"
                                   "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
rxmesh_install_directory_if_exists("${metis_SOURCE_DIR}/GKlib"
                                   "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")

if(DEFINED EIGEN_INCLUDE_DIRS)
    rxmesh_install_directory_if_exists("${EIGEN_INCLUDE_DIRS}/Eigen"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    rxmesh_install_directory_if_exists("${EIGEN_INCLUDE_DIRS}/unsupported"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
endif()

if(RX_USE_POLYSCOPE AND DEFINED polyscope_SOURCE_DIR)
    rxmesh_install_directory_if_exists("${polyscope_SOURCE_DIR}/include/"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    rxmesh_install_directory_if_exists("${polyscope_SOURCE_DIR}/deps/imgui/imgui/"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    rxmesh_install_directory_if_exists("${polyscope_SOURCE_DIR}/deps/imgui/implot/"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    rxmesh_install_directory_if_exists("${polyscope_SOURCE_DIR}/deps/imgui/ImGuizmo/"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    rxmesh_install_directory_if_exists("${polyscope_SOURCE_DIR}/deps/imgui/custom_backends/"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    rxmesh_install_directory_if_exists("${polyscope_SOURCE_DIR}/deps/glad/include/"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    rxmesh_install_directory_if_exists("${polyscope_SOURCE_DIR}/deps/glfw/include/"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    rxmesh_install_directory_if_exists("${polyscope_SOURCE_DIR}/deps/json/include/"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    rxmesh_install_directory_if_exists("${polyscope_SOURCE_DIR}/deps/MarchingCubeCpp/include/"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    rxmesh_install_directory_if_exists("${polyscope_SOURCE_DIR}/deps/IconFontCppHeaders/include/"
                                       "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}")

    if(EXISTS "${polyscope_SOURCE_DIR}/deps/stb")
        install(DIRECTORY "${polyscope_SOURCE_DIR}/deps/stb/"
                DESTINATION "${RXMESH_INSTALL_THIRD_PARTY_INCLUDEDIR}"
                FILES_MATCHING PATTERN "*.h"
                ${_RXMESH_INSTALL_COMPONENT_ARGS})
    endif()
endif()

configure_package_config_file(
    "${RXMESH_SOURCE_DIR}/cmake/RXMeshConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/RXMeshConfig.cmake"
    INSTALL_DESTINATION "${RXMESH_INSTALL_CMAKEDIR}")

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/RXMeshConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/RXMeshConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/RXMeshConfigVersion.cmake"
        DESTINATION "${RXMESH_INSTALL_CMAKEDIR}"
        ${_RXMESH_INSTALL_COMPONENT_ARGS})

unset(_RXMESH_INSTALL_COMPONENT_ARGS)
