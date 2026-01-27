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
  set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

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
endfunction()

