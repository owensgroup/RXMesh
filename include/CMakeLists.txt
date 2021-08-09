add_library(RXMesh_lib INTERFACE)

file(GLOB_RECURSE SOURCE_LIST rxmesh/*.*)

target_sources(RXMesh_lib
    INTERFACE ${SOURCE_LIST}
)


#set_property(TARGET RXMesh_lib PROPERTY FOLDER "rx_mesh")
#source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "rx_mesh" FILES ${SOURCE_LIST})