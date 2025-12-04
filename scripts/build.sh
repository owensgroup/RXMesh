#!/bin/bash

# RXMesh Simple Build Script
# This script builds the RXMesh project with minimal user input

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
JOBS=$(nproc)

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to get user input with default
get_input() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    echo -n "$prompt [$default]: "
    read -r input
    
    if [ -z "$input" ]; then
        eval "$var_name=\"$default\""
    else
        eval "$var_name=\"$input\""
    fi
}



# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v cmake &> /dev/null; then
        print_error "CMake is not installed. Please install CMake 3.20 or higher."
        exit 1
    fi
    
    if ! command -v make &> /dev/null; then
        print_error "Make is not installed."
        exit 1
    fi
    
    if ! command -v nvcc &> /dev/null; then
        print_warning "NVCC (CUDA compiler) not found. CUDA features may not work."
    fi
    
    print_success "Prerequisites check passed"
}

# Function to build
build_project() {
    local build_dir="cmake-build-${BUILD_TYPE,,}"
    
    print_status "Building $BUILD_TYPE configuration..."
    
    # Create build directory
    mkdir -p "$build_dir"
    cd "$build_dir"
    
    # Configure CMake
    print_status "Configuring CMake..."
    cmake .. \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DRX_USE_POLYSCOPE=ON \
        -DRX_BUILD_TESTS=ON \
        -DRX_BUILD_APPS=ON \
        -DRX_USE_SUITESPARSE=OFF \
        -DRX_USE_CUDSS=OFF
    
    # Build
    print_status "Building with $JOBS jobs..."
    make -j"$JOBS"
    
    print_success "Build completed successfully!"
    
    # Go back to root
    cd ..
}

# Main execution
main() {
    echo "RXMesh Build Script"
    echo "=================="
    echo
    
    # Check prerequisites
    check_prerequisites
    
    # Get build type
    echo "Build type options:"
    echo "  Release - Optimized build (default)"
    echo "  Debug   - Debug build with symbols"
    echo
    get_input "Select build type" "$BUILD_TYPE" "BUILD_TYPE"
    
    # Confirm build
    echo
    echo "Build Configuration:"
    echo "  Build Type: $BUILD_TYPE"
    echo "  Jobs: $JOBS"
    echo "  GPU Architecture: Auto-detected by CMake"
    echo
    
    echo -n "Proceed with build? [Y/n]: "
    read -r confirm
    
    if [[ "$confirm" =~ ^[Nn]$ ]]; then
        print_status "Build cancelled by user"
        exit 0
    fi
    
    # Build the project
    echo
    build_project
    
    # Show results
    echo
    print_success "Build completed successfully!"
    echo
    echo "Build Summary:"
    echo "=============="
    echo "✓ $BUILD_TYPE build: cmake-build-${BUILD_TYPE,,}/"
    echo
    echo "To run the project:"
    echo "  ./cmake-build-${BUILD_TYPE,,}/bin/[executable_name]"
    echo
    echo "To clean: rm -rf cmake-build-*"
}

# Run main function
main "$@"
