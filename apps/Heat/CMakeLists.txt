if(NOT ${RX_USE_CUDSS})
    return()
endif()

set(SOURCE_LIST
  heat.cu
)

rxmesh_add_app(Heat
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh CLI11::CLI11
)
