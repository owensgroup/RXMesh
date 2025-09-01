if (${PROJECT_NAME}_WITH_SUITESPARSE)
  include(FetchContent)

  # ===========================================================
  # 1.  BLAS  (try system first, else build OpenBLAS)
  # ===========================================================
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS QUIET COMPONENTS CBLAS)

  if (NOT BLAS_FOUND)
    message(STATUS "No BLAS on the system – building OpenBLAS")

    FetchContent_Declare(
      openblas
      GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
      GIT_TAG        v0.3.27
    )
    # We need the LAPACK symbols for SuiteSparse
    set(BUILD_WITHOUT_LAPACK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(openblas)
	
    # 1) Pick the real OpenBLAS target (may be an alias)
	set(_blas_target "")
	if(TARGET openblas)
		# openblas is usually an ALIAS → follow it
		get_property(_aliased TARGET openblas PROPERTY ALIASED_TARGET)
		if(_aliased)
			set(_blas_target ${_aliased})      # real imported library
		else()
			set(_blas_target openblas)         # openblas *is* the real one
		endif()
	elseif(TARGET OpenBLAS::OpenBLAS)
		set(_blas_target OpenBLAS::OpenBLAS) # MinGW / some Linux builds
	else()
		message(FATAL_ERROR "OpenBLAS was built but exported no usable target")
	endif()

	get_target_property(OPENBLAS_RELEASE ${_blas_target} IMPORTED_LOCATION_RELEASE)
	get_target_property(OPENBLAS_DEBUG   ${_blas_target} IMPORTED_LOCATION_DEBUG)

	# 2)  Provide a canonical BLAS::BLAS, but only once
	if(NOT TARGET BLAS::BLAS)
		add_library(BLAS::BLAS ALIAS ${_blas_target})
	endif()

	# 3) LAPACK shim (SuiteSparse needs it)
	if(NOT TARGET LAPACK::LAPACK)
		add_library(LAPACK::LAPACK ALIAS ${_blas_target})
	endif()
	

	# 4) Legacy cache variables so FindBLAS/LAPACK are satisfied
	set(BLAS_LIBRARIES   BLAS::BLAS    CACHE STRING "" FORCE)
	set(BLAS_FOUND       TRUE              CACHE BOOL   "" FORCE)	
	set(LAPACK_LIBRARIES LAPACK::LAPACK CACHE STRING "" FORCE)
	set(LAPACK_FOUND     TRUE              CACHE BOOL   "" FORCE)
	set(LAPACK_LIBRARIES ${BLAS_LIBRARIES} CACHE STRING "" FORCE)
	set(LAPACK_FOUND     TRUE              CACHE BOOL   "" FORCE)	
  endif()
  
  # ===========================================================
  # 3.  Fetch & build SuiteSparse
  # ===========================================================
  set(WITH_FORTRAN              OFF CACHE BOOL "" FORCE)
  set(WITH_CUDA                 ON  CACHE BOOL "" FORCE)
  set(WITH_PARTITION            ON  CACHE BOOL "" FORCE)
  set(WITH_DEMOS                OFF CACHE BOOL "" FORCE)
  set(SuiteSparse_ENABLE_LAPACK ON  CACHE BOOL "" FORCE)

  FetchContent_Declare(
    suitesparse
    GIT_REPOSITORY https://github.com/sergiud/SuiteSparse.git
    GIT_TAG        cmake
  )
  FetchContent_MakeAvailable(suitesparse)
  
  # Create suitesparse_umbrella alias if it doesn't exist
  # The SuiteSparse build provides individual component targets like SuiteSparse::cholmod
  # We'll create an umbrella target that links to the main components
  if(NOT TARGET suitesparse_umbrella)
    add_library(suitesparse_umbrella INTERFACE)
    
    # Link to the main SuiteSparse components
    if(TARGET SuiteSparse::cholmod)
      target_link_libraries(suitesparse_umbrella INTERFACE SuiteSparse::cholmod)
    endif()
    if(TARGET SuiteSparse::spqr)
      target_link_libraries(suitesparse_umbrella INTERFACE SuiteSparse::spqr)
    endif()
    if(TARGET SuiteSparse::amd)
      target_link_libraries(suitesparse_umbrella INTERFACE SuiteSparse::amd)
    endif()
    if(TARGET SuiteSparse::colamd)
      target_link_libraries(suitesparse_umbrella INTERFACE SuiteSparse::colamd)
    endif()
    if(TARGET SuiteSparse::camd)
      target_link_libraries(suitesparse_umbrella INTERFACE SuiteSparse::camd)
    endif()
    if(TARGET SuiteSparse::ccolamd)
      target_link_libraries(suitesparse_umbrella INTERFACE SuiteSparse::ccolamd)
    endif()
    if(TARGET SuiteSparse::config)
      target_link_libraries(suitesparse_umbrella INTERFACE SuiteSparse::config)
    endif()
  endif()
endif()
