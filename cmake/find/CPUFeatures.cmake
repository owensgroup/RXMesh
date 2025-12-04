# Find CPU features and characteristics
# This module defines
#  CPU_FEATURES_FOUND, if false, do not try to use CPU features
#  CPU_VENDOR, CPU vendor (Intel, AMD, etc.)
#  CPU_MODEL, CPU model name
#  CPU_CORES, number of CPU cores
#  CPU_CACHE_L1, L1 cache size in KB
#  CPU_CACHE_L2, L2 cache size in KB
#  CPU_CACHE_L3, L3 cache size in KB
#  CPU_FEATURES, list of all available CPU features

include(CheckCXXSourceCompiles)

# Set up CPU features variables
set(CPU_FEATURES_FOUND FALSE)
set(CPU_VENDOR "Unknown")
set(CPU_MODEL "Unknown")
set(CPU_CORES 0)
set(CPU_CACHE_L1 0)
set(CPU_CACHE_L2 0)
set(CPU_CACHE_L3 0)
set(CPU_FEATURES "")

# Create a test program to detect CPU features
set(CPU_TEST_SOURCE "
#include <cpuid.h>
#include <iostream>
#include <string>
#include <cstring>

int main() {
    unsigned int eax, ebx, ecx, edx;
    char vendor[13];
    
    // Get CPU vendor
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        memcpy(vendor, &ebx, 4);
        memcpy(vendor + 4, &edx, 4);
        memcpy(vendor + 8, &ecx, 4);
        vendor[12] = '\\0';
        
        std::cout << \"VENDOR:\" << vendor << std::endl;
        
        // Get CPU model
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            unsigned int family = ((eax >> 8) & 0xF) + ((eax >> 20) & 0xFF);
            unsigned int model = ((eax >> 4) & 0xF) + ((eax >> 12) & 0xF0);
            
            std::cout << \"FAMILY:\" << family << std::endl;
            std::cout << \"MODEL:\" << model << std::endl;
            
            // Get cache information
            if (__get_cpuid(2, &eax, &ebx, &ecx, &edx)) {
                // Parse cache descriptors (simplified)
                std::cout << \"CACHE_INFO:\" << eax << \",\" << ebx << \",\" << ecx << \",\" << edx << std::endl;
            }
            
            // Get extended features
            if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
                std::cout << \"EXTENDED_FEATURES:\" << ebx << \",\" << ecx << \",\" << edx << std::endl;
            }
        }
        
        return 0;
    }
    
    return 1;
}
")

# Try to compile and run the CPU detection test
try_run(CPU_RUNTIME_RESULT CPU_COMPILE_RESULT
    ${CMAKE_BINARY_DIR}
    SOURCES ${CMAKE_BINARY_DIR}/cpu_test.cpp
    RUN_OUTPUT_VARIABLE CPU_RUNTIME_OUTPUT
    COMPILE_OUTPUT_VARIABLE CPU_COMPILE_OUTPUT
)

if(CPU_COMPILE_RESULT AND CPU_RUNTIME_RESULT EQUAL 0)
    set(CPU_FEATURES_FOUND TRUE)
    
    # Parse the output to extract CPU information
    string(REGEX MATCH "VENDOR:([^\\n]+)" VENDOR_MATCH "${CPU_RUNTIME_OUTPUT}")
    if(VENDOR_MATCH)
        set(CPU_VENDOR "${CMAKE_MATCH_1}")
    endif()
    
    string(REGEX MATCH "FAMILY:([^\\n]+)" FAMILY_MATCH "${CPU_RUNTIME_OUTPUT}")
    if(FAMILY_MATCH)
        set(CPU_FAMILY "${CMAKE_MATCH_1}")
    endif()
    
    string(REGEX MATCH "MODEL:([^\\n]+)" MODEL_MATCH "${CPU_RUNTIME_OUTPUT}")
    if(MODEL_MATCH)
        set(CPU_MODEL "${CMAKE_MATCH_1}")
    endif()
    
    # Try to get core count from system
    include(ProcessorCount)
    ProcessorCount(CPU_CORES)
    
    message(STATUS "CPU: Vendor: ${CPU_VENDOR}")
    message(STATUS "CPU: Family: ${CPU_FAMILY}")
    message(STATUS "CPU: Model: ${CPU_MODEL}")
    message(STATUS "CPU: Cores: ${CPU_CORES}")
    
    # Set CPU features based on vendor
    if(CPU_VENDOR STREQUAL "GenuineIntel")
        list(APPEND CPU_FEATURES "Intel")
        if(CPU_FAMILY GREATER_EQUAL 6)
            list(APPEND CPU_FEATURES "x86_64")
            if(CPU_MODEL GREATER_EQUAL 60)
                list(APPEND CPU_FEATURES "Haswell_or_newer")
            endif()
        endif()
    elseif(CPU_VENDOR STREQUAL "AuthenticAMD")
        list(APPEND CPU_FEATURES "AMD")
        list(APPEND CPU_FEATURES "x86_64")
    endif()
    
else()
    message(WARNING "CPU: Could not detect CPU features at runtime")
endif()

# Provide convenience functions
function(target_enable_cpu_features TARGET_NAME)
    if(CPU_FEATURES_FOUND)
        target_compile_definitions(${TARGET_NAME} PRIVATE 
            CPU_VENDOR="${CPU_VENDOR}"
            CPU_CORES=${CPU_CORES}
        )
        message(STATUS "CPU: Enabled features for target ${TARGET_NAME}")
    else()
        message(WARNING "CPU: Cannot enable CPU features for target ${TARGET_NAME} - detection failed")
    endif()
endfunction()

# Mark variables as advanced
mark_as_advanced(CPU_VENDOR CPU_MODEL CPU_CORES CPU_CACHE_L1 CPU_CACHE_L2 CPU_CACHE_L3 CPU_FEATURES)

