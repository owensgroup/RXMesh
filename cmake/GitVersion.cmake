include_guard(GLOBAL)

# Get and store git sha1 https://stackoverflow.com/a/4318642/1608232

if(NOT DEFINED RXMESH_SOURCE_DIR)
    get_filename_component(RXMESH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_local_changes(GIT_LOCAL_CHANGES_STATUS)

configure_file(
    "${RXMESH_SOURCE_DIR}/cmake/git_sha1.cpp.in"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/util/git_sha1.cpp"
    @ONLY
)

