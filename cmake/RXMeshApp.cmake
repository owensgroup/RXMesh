include_guard(GLOBAL)

include(CMakeParseArguments)

function(rxmesh_resolve_cuda_architectures output_variable)
  set(rxmesh_architectures ${ARGN})

  if("${rxmesh_architectures}" STREQUAL "native")
    if(NOT CMAKE_CUDA_ARCHITECTURES_NATIVE)
      message(FATAL_ERROR
        "RXMesh could not resolve the native GPU to a numeric CUDA "
        "architecture required by device LTO")
    endif()
    set(rxmesh_architectures ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
  endif()

  if(NOT rxmesh_architectures)
    message(FATAL_ERROR
      "RXMesh device LTO requires at least one CUDA architecture")
  endif()

  foreach(rxmesh_architecture IN LISTS rxmesh_architectures)
    if(NOT rxmesh_architecture MATCHES "^[0-9]+(-real|-virtual)?$")
      message(FATAL_ERROR
        "RXMesh device LTO requires numeric CUDA architectures, optionally "
        "followed by '-real' or '-virtual'; got '${rxmesh_architecture}'")
    endif()
  endforeach()

  set(${output_variable} "${rxmesh_architectures}" PARENT_SCOPE)
endfunction()

function(rxmesh_enable_cuda_device_lto target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR
      "rxmesh_enable_cuda_device_lto: '${target}' is not a CMake target")
  endif()

  get_target_property(rxmesh_target_architectures
                      ${target}
                      CUDA_ARCHITECTURES)
  if(NOT rxmesh_target_architectures)
    set(rxmesh_target_architectures ${CMAKE_CUDA_ARCHITECTURES})
  endif()

  rxmesh_resolve_cuda_architectures(
    rxmesh_target_architectures
    ${rxmesh_target_architectures})
  set_property(TARGET ${target}
               PROPERTY CUDA_ARCHITECTURES "${rxmesh_target_architectures}")
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
  set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
endfunction()

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
  set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  rxmesh_enable_cuda_device_lto(${target})

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
  
  if(WIN32 AND RX_USE_CUDSS AND TARGET CopyCUDSSDLL)
    add_dependencies(${target} CopyCUDSSDLL)
  endif()
  
  message(STATUS "RXMesh: Added ${target} target")
  
endfunction()

