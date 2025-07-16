#!/bin/bash
# Build script for HSOF Docker image

set -e

# Default values
IMAGE_NAME="${IMAGE_NAME:-hsof}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BUILD_TARGET="${BUILD_TARGET:-production}"
NO_CACHE="${NO_CACHE:-false}"

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --name NAME       Image name (default: hsof)"
    echo "  -t, --tag TAG         Image tag (default: latest)"
    echo "  -s, --stage STAGE     Build stage (default: production)"
    echo "  --no-cache            Build without cache"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Build stages:"
    echo "  cuda-base      - Base CUDA image"
    echo "  julia-base     - Julia installation"
    echo "  julia-packages - Julia packages precompiled"
    echo "  app-build      - Application build"
    echo "  production     - Production runtime (default)"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -s|--stage)
            BUILD_TARGET="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build command
BUILD_CMD="docker build"
BUILD_CMD+=" -f docker/Dockerfile"
BUILD_CMD+=" -t ${IMAGE_NAME}:${IMAGE_TAG}"
BUILD_CMD+=" --target ${BUILD_TARGET}"

if [ "$NO_CACHE" = "true" ]; then
    BUILD_CMD+=" --no-cache"
fi

# Build arguments
BUILD_CMD+=" --build-arg JULIA_VERSION=1.10.5"

# Add current directory as build context
BUILD_CMD+=" ."

# Show build info
echo "================================================"
echo " HSOF Docker Image Build"
echo "================================================"
echo " Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo " Stage: ${BUILD_TARGET}"
echo " Cache: $([ "$NO_CACHE" = "true" ] && echo "disabled" || echo "enabled")"
echo "================================================"
echo ""

# Execute build
echo "Running: $BUILD_CMD"
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Run the build
eval $BUILD_CMD

# Show completion
echo ""
echo "================================================"
echo " Build completed successfully!"
echo "================================================"
echo ""
echo "To run the container:"
echo "  docker run --rm --gpus all ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To run with docker-compose:"
echo "  docker-compose up hsof"
echo ""