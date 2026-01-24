include_guard(GLOBAL)

function(rxmesh_enable_cudss target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "rxmesh_enable_cudss: target '${target}' does not exist")
    endif()

    find_package(cudss QUIET)
    if (cudss_FOUND)
        message(STATUS "Found cuDSS version ${cudss_VERSION}")

        target_link_directories(${target} INTERFACE ${cudss_LIBRARY_DIR})

        if (WIN32)
            target_include_directories(${target} INTERFACE ${cudss_INCLUDE_DIR})
            target_link_libraries(${target} INTERFACE cudss)

            # Copy cuDSS DLL to the binary directory
            add_custom_target(CopyCUDSSDLL)
            foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
                add_custom_command(
                    OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll"
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${cudss_BINARY_DIR}/cudss64_0.dll"
                        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll"
                    DEPENDS "${cudss_BINARY_DIR}/cudss64_0.dll"
                    COMMENT "Copying cudss64_0.dll for ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG} configuration"
                    VERBATIM
                )
                add_custom_target("CopyCUDSSDLL${CONFIG}"
                    DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll")
                add_dependencies(CopyCUDSSDLL "CopyCUDSSDLL${CONFIG}")
            endforeach()
        else()
            target_link_libraries(${target} INTERFACE cudss_static)
        endif()

        target_compile_definitions(${target} INTERFACE USE_CUDSS)
    else (cudss_FOUND)
        message(WARNING "Can not find cuDSS")
        set(RX_USE_CUDSS "OFF" CACHE BOOL "")
    endif()
endfunction()

