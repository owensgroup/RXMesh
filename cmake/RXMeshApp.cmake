include(CMakeParseArguments)

# Create an RXMesh "app" executable
#
# Usage:
#   rxmesh_add_app(MyApp
#     SOURCES <list...>
#     [LIBS <list...>]                # defaults to RXMesh
#     [DEPENDS <list...>]
#     [FOLDER <folder>]               # defaults to "apps"
#     [SOURCE_GROUP_PREFIX <prefix>]  # defaults to target name
#   )
function(rxmesh_add_app target)  
  set(options)
  set(oneValueArgs FOLDER SOURCE_GROUP_PREFIX)
  set(multiValueArgs SOURCES LIBS DEPENDS)

  cmake_parse_arguments(RXAPP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT RXAPP_FOLDER)
    set(RXAPP_FOLDER "apps")
  endif()

  if(NOT RXAPP_SOURCE_GROUP_PREFIX)
    set(RXAPP_SOURCE_GROUP_PREFIX "${target}")
  endif()

  add_executable(${target})

  if(RXAPP_SOURCES)
    target_sources(${target} PRIVATE ${RXAPP_SOURCES})
  endif()

  set_target_properties(${target} PROPERTIES FOLDER "${RXAPP_FOLDER}")

  if(USE_HIP)
    # Compile this target's .cu sources with HIP (the project enables HIP, not
    # CUDA, so unmarked .cu files would be silently dropped from the link).
    foreach(src ${RXAPP_SOURCES})
      if(src MATCHES "\\.cu$")
        set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
      endif()
    endforeach()
    set_target_properties(${target} PROPERTIES
      HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
      HIP_SEPARABLE_COMPILATION ON)
    # Match RXMesh's relocatable device code so cross-TU __device__ symbols link.
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
    target_link_options(${target} PRIVATE $<$<LINK_LANGUAGE:HIP>:-fgpu-rdc> --hip-link)
    # On Windows: -fuse-ld=lld-link (from CMake platform module) is rejected by
    # clang++ gcc-driver under -fgpu-rdc device link; override to empty.
    # --allow-multiple-definition resolves the duplicate this_cluster() device
    # symbol from the HIP cooperative_groups header (non-inline __device__ fn in
    # a system header shows as a strong symbol under -fgpu-rdc on Windows PE).
    if(WIN32)
        target_link_options(${target} PRIVATE
            $<$<LINK_LANGUAGE:HIP>:-fuse-ld=>
            $<$<LINK_LANGUAGE:HIP>:-Xoffload-linker --allow-multiple-definition>)
    endif()
  else()
    set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  endif()

  if(RXAPP_SOURCES)
    source_group(      
      TREE ${CMAKE_CURRENT_LIST_DIR}
      PREFIX "${RXAPP_SOURCE_GROUP_PREFIX}"
      FILES ${RXAPP_SOURCES}
    )
  endif()

  if(RXAPP_LIBS)
    target_link_libraries(${target} PRIVATE ${RXAPP_LIBS})
  else()
    target_link_libraries(${target} PRIVATE RXMesh)
  endif()

  if(WIN32)
    target_compile_definitions(${target}
      PRIVATE _USE_MATH_DEFINES
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS
    )
  endif()

  if(RXAPP_DEPENDS)
    add_dependencies(${target} ${RXAPP_DEPENDS})
  endif()
  
  if(WIN32 AND ${RX_USE_CUDSS} AND TARGET CopyCUDSSDLL)
    add_dependencies(${target} CopyCUDSSDLL)
  endif()
  
  message(STATUS "RXMesh: Added ${target} target")
  
endfunction()

