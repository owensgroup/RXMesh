#!/bin/bash

# RXMesh Quick Build Script
# Non-interactive build with A6000 defaults

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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
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
    echo "RXMesh Quick Build Script"
    echo "========================"
    echo
    echo "Build Configuration:"
    echo "  Build Type: $BUILD_TYPE"
    echo "  Jobs: $JOBS"
    echo "  GPU Architecture: Auto-detected by CMake"
    echo
    
    # Check prerequisites
    check_prerequisites
    
    # Build the project
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
}

# Run main function
main "$@"
