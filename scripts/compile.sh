#!/bin/bash

# RXMesh Compile Script
# This script only compiles and links existing CMake build directories
# It does NOT run CMake configuration - only make/build

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
JOBS=$(nproc)
BUILD_DEBUG=false
BUILD_RELEASE=false
BUILD_ALL=false

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

# Function to show usage
show_usage() {
    echo "RXMesh Compile Script"
    echo "===================="
    echo "Compiles and links existing CMake build directories (does NOT run CMake configuration)"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -d, --debug       Compile debug build (cmake-build-debug)"
    echo "  -r, --release     Compile release build (cmake-build-release)"
    echo "  -a, --all         Compile both debug and release builds"
    echo "  -j, --jobs N      Number of parallel jobs (default: $(nproc))"
    echo "  -h, --help        Show this help message"
    echo
    echo "Examples:"
    echo "  $0 -d             # Compile debug build only"
    echo "  $0 -r             # Compile release build only"
    echo "  $0 -a             # Compile both debug and release"
    echo "  $0 -d -j 8        # Compile debug with 8 parallel jobs"
    echo
    echo "Note: This script requires the build directories to be already configured with CMake."
    echo "      Use build.sh or build-quick.sh to configure and build from scratch."
}

# Function to check if build directory exists and is configured
check_build_dir() {
    local build_dir="$1"
    local build_type="$2"
    
    if [ ! -d "$build_dir" ]; then
        print_error "$build_type build directory '$build_dir' does not exist"
        echo "Run 'scripts/build.sh' or 'scripts/build-quick.sh' first to configure the build."
        return 1
    fi
    
    if [ ! -f "$build_dir/Makefile" ] && [ ! -f "$build_dir/build.ninja" ]; then
        print_error "$build_type build directory '$build_dir' is not configured"
        echo "Run 'scripts/build.sh' or 'scripts/build-quick.sh' first to configure the build."
        return 1
    fi
    
    return 0
}

# Function to compile a build directory
compile_build() {
    local build_dir="$1"
    local build_type="$2"
    
    print_status "Compiling $build_type build..."
    
    # Change to build directory
    cd "$build_dir"
    
    # Check if we have Makefile or Ninja build files
    if [ -f "Makefile" ]; then
        print_status "Using Make with $JOBS jobs..."
        make -j"$JOBS"
    elif [ -f "build.ninja" ]; then
        print_status "Using Ninja with $JOBS jobs..."
        ninja -j"$JOBS"
    else
        print_error "No valid build system found in $build_dir"
        cd ..
        return 1
    fi
    
    print_success "$build_type compilation completed!"
    
    # Go back to root
    cd ..
    return 0
}

# Function to check prerequisites
check_prerequisites() {
    if ! command -v make &> /dev/null && ! command -v ninja &> /dev/null; then
        print_error "Neither Make nor Ninja is available."
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_DEBUG=true
            shift
            ;;
        -r|--release)
            BUILD_RELEASE=true
            shift
            ;;
        -a|--all)
            BUILD_ALL=true
            shift
            ;;
        -j|--jobs)
            if [[ -n $2 ]] && [[ $2 =~ ^[0-9]+$ ]]; then
                JOBS="$2"
                shift 2
            else
                print_error "Invalid number of jobs: $2"
                exit 1
            fi
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo
            show_usage
            exit 1
            ;;
    esac
done

# If no specific build type is selected, show usage
if [ "$BUILD_DEBUG" = false ] && [ "$BUILD_RELEASE" = false ] && [ "$BUILD_ALL" = false ]; then
    print_warning "No build type specified."
    echo
    show_usage
    exit 1
fi

# If --all is specified, enable both debug and release
if [ "$BUILD_ALL" = true ]; then
    BUILD_DEBUG=true
    BUILD_RELEASE=true
fi

# Main execution
main() {
    echo "RXMesh Compile Script"
    echo "===================="
    echo
    
    # Check prerequisites
    check_prerequisites
    
    # Track compilation results
    local success_count=0
    local total_count=0
    local failed_builds=()
    
    # Compile debug build if requested
    if [ "$BUILD_DEBUG" = true ]; then
        total_count=$((total_count + 1))
        echo "Debug Build:"
        echo "-----------"
        if check_build_dir "cmake-build-debug" "Debug"; then
            if compile_build "cmake-build-debug" "Debug"; then
                success_count=$((success_count + 1))
            else
                failed_builds+=("Debug")
            fi
        else
            failed_builds+=("Debug")
        fi
        echo
    fi
    
    # Compile release build if requested
    if [ "$BUILD_RELEASE" = true ]; then
        total_count=$((total_count + 1))
        echo "Release Build:"
        echo "-------------"
        if check_build_dir "cmake-build-release" "Release"; then
            if compile_build "cmake-build-release" "Release"; then
                success_count=$((success_count + 1))
            else
                failed_builds+=("Release")
            fi
        else
            failed_builds+=("Release")
        fi
        echo
    fi
    
    # Show summary
    echo "Compilation Summary:"
    echo "==================="
    echo "✓ Successful builds: $success_count/$total_count"
    
    if [ ${#failed_builds[@]} -gt 0 ]; then
        echo "✗ Failed builds: ${failed_builds[*]}"
        echo
        print_error "Some builds failed. Check the output above for details."
        exit 1
    else
        echo
        print_success "All requested builds completed successfully!"
        
        # Show available executables
        echo
        echo "Build outputs:"
        echo "=============="
        if [ "$BUILD_DEBUG" = true ] && [ -d "cmake-build-debug/bin" ]; then
            echo "Debug executables: cmake-build-debug/bin/"
            ls -la cmake-build-debug/bin/ 2>/dev/null || echo "  (no executables found)"
        fi
        if [ "$BUILD_RELEASE" = true ] && [ -d "cmake-build-release/bin" ]; then
            echo "Release executables: cmake-build-release/bin/"
            ls -la cmake-build-release/bin/ 2>/dev/null || echo "  (no executables found)"
        fi
    fi
}

# Run main function
main "$@"
