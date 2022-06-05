if(NOT DEFINED CUDA_ARCHS)
	############################### Autodetect CUDA Arch #####################################################
	#Auto-detect cuda arch. Inspired by https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake
	# This will define and populates CUDA_ARCHS and put it in the cache 
	#Windows users (specially on VS2017 and VS2015) might need to run this 
	#>> "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
	# and change Enterprise to the right edition. More about this here https://stackoverflow.com/a/47746461/1608232
	if(CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA)
		set(cuda_arch_autodetect_file ${CMAKE_BINARY_DIR}/autodetect_cuda_archs.cu)		
		
		file(WRITE ${cuda_arch_autodetect_file} [[
#include <stdio.h>
int main() {
	int count = 0; 
	if (cudaSuccess != cudaGetDeviceCount(&count)) { return -1; }
	if (count == 0) { return -1; }
	for (int device = 0; device < count; ++device) {
		cudaDeviceProp prop; 
		bool is_unique = true; 
		if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {
			for (int device_1 = device - 1; device_1 >= 0; --device_1) {
				cudaDeviceProp prop_1; 
				if (cudaSuccess == cudaGetDeviceProperties(&prop_1, device_1)) {
					if (prop.major == prop_1.major && prop.minor == prop_1.minor) {
						is_unique = false; 
						break; 
					}
				}
				else { return -1; }
			}
			if (is_unique) {
				fprintf(stderr, "%d%d", prop.major, prop.minor);
			}
		}
		else { return -1; }
	}
	return 0; 
}
		]])
		
		execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" "-ccbin" "${CMAKE_CXX_COMPILER}" "--run" "${cuda_arch_autodetect_file}"
						WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"	
						RESULT_VARIABLE CUDA_RETURN_CODE	
						OUTPUT_VARIABLE dummy
						ERROR_VARIABLE fprintf_output					
						OUTPUT_STRIP_TRAILING_WHITESPACE)							
		
		if(CUDA_RETURN_CODE EQUAL 0)			
			set(CMAKE_CUDA_ARCHITECTURES ${fprintf_output})
		else()
			message(STATUS "GPU architectures auto-detect failed. Will build for sm_70.")      
			set(CMAKE_CUDA_ARCHITECTURES 70)		
		endif()  
	endif()	
	message(STATUS "CUDA architectures= " ${CMAKE_CUDA_ARCHITECTURES})	
endif()
###################################################################################

