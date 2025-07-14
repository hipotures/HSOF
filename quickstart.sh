#!/bin/bash
# HSOF Quick Start Script
# Sets up environment and runs initial validation

set -e

echo "======================================"
echo "HSOF Quick Start Setup"
echo "======================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        exit 1
    fi
}

# Check Julia installation
echo -e "\n📋 Checking requirements..."

if ! command -v julia &> /dev/null; then
    echo -e "${RED}Julia is not installed!${NC}"
    echo "Please install Julia 1.9 or higher from https://julialang.org"
    exit 1
fi

JULIA_VERSION=$(julia --version | cut -d' ' -f3)
echo "Found Julia $JULIA_VERSION"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n🎮 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
    echo -e "${YELLOW}⚠️  nvidia-smi not found. GPU support may not be available.${NC}"
fi

# Set up environment
echo -e "\n⚙️  Setting up environment..."

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << EOF
JULIA_NUM_THREADS=auto
JULIA_CUDA_MEMORY_POOL=cuda
JULIA_CUDA_SOFT_MEMORY_LIMIT=0.9
HSOF_ENV=dev
EOF
    print_status 0 "Created .env file"
else
    print_status 0 ".env file exists"
fi

# Create config files from examples
echo -e "\n📁 Setting up configuration files..."
for example in configs/*.toml.example; do
    if [ -f "$example" ]; then
        config="${example%.example}"
        if [ ! -f "$config" ]; then
            cp "$example" "$config"
            echo "  Created $(basename $config)"
        fi
    fi
done

# Install Julia dependencies
echo -e "\n📦 Installing Julia packages..."
julia --project=. -e 'using Pkg; Pkg.instantiate()' || print_status 1 "Failed to install packages"
print_status 0 "Julia packages installed"

# Run validation
echo -e "\n🔍 Running environment validation..."
julia validate_environment.jl || true

# Build project
echo -e "\n🔧 Building project..."
julia build.jl --kernels || print_status 1 "Build failed"

# Run quick test
echo -e "\n🧪 Running quick test..."
julia --project=. -e '
    using HSOF
    println("✓ HSOF loaded successfully")
    
    # Test data generation
    X, y = HSOF.generate_sample_data(n_samples=100, n_features=50)
    println("✓ Sample data generated: ", size(X))
    
    # Test GPU if available
    using CUDA
    if CUDA.functional()
        println("✓ GPU is functional: ", CUDA.name(CUDA.device()))
    else
        println("⚠️  GPU not available, CPU mode will be used")
    end
' || print_status 1 "Quick test failed"

# Success message
echo -e "\n${GREEN}======================================"
echo "✅ HSOF setup complete!"
echo "======================================${NC}"

echo -e "\n🚀 Quick start commands:"
echo "  julia --project=."
echo "  julia> using HSOF"
echo "  julia> ?HSOF  # For help"

echo -e "\n📚 Next steps:"
echo "  - Run full tests: make test"
echo "  - Build docs: make docs"
echo "  - Run benchmarks: make benchmark"
echo "  - See all options: make help"

echo -e "\n📖 Documentation:"
echo "  - Installation: docs/src/getting-started/installation.md"
echo "  - Quick Start: docs/src/getting-started/quickstart.md"
echo "  - GPU Guide: docs/src/tutorials/gpu-programming.md"