# Installation Guide

```@meta
CurrentModule = HSOF
```

## System Requirements

### Hardware Requirements

- **GPUs**: 2x NVIDIA RTX 4090 (24GB VRAM) or equivalent
  - Minimum: 1x GPU with 12GB+ VRAM (reduced performance)
  - Compute Capability: 8.0+ (Ampere or newer)
- **CPU**: 8+ cores recommended
- **RAM**: 64GB recommended (32GB minimum)
- **Storage**: 50GB free space for datasets and models

### Software Requirements

- **Operating System**: Ubuntu 20.04+ or similar Linux distribution
- **CUDA**: 11.8 or higher
- **Julia**: 1.9 or higher
- **Python**: 3.8+ (for some dependencies)
- **Git**: For version control

## CUDA Installation

### 1. Install NVIDIA Driver

```bash
# Check current driver
nvidia-smi

# If not installed or outdated:
sudo apt update
sudo apt install nvidia-driver-525  # or latest version
sudo reboot
```

### 2. Install CUDA Toolkit

```bash
# Download CUDA 12.0 (or latest)
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run

# Install
sudo sh cuda_12.0.0_525.60.13_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

## Julia Installation

### 1. Download and Install Julia

```bash
# Download Julia 1.10 (or latest)
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz

# Extract
tar -xvf julia-1.10.0-linux-x86_64.tar.gz

# Move to /opt
sudo mv julia-1.10.0 /opt/

# Create symlink
sudo ln -s /opt/julia-1.10.0/bin/julia /usr/local/bin/julia

# Verify
julia --version
```

### 2. Configure Julia for CUDA

```julia
# Start Julia REPL
julia

# Install and configure CUDA.jl
using Pkg
Pkg.add("CUDA")
using CUDA
CUDA.versioninfo()  # Should show GPU information
```

## HSOF Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/HSOF.git
cd HSOF
```

### 2. Install Dependencies

```julia
# In Julia REPL
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Build CUDA artifacts
using CUDA
CUDA.precompile_runtime()
```

### 3. Verify Installation

```julia
# Run validation tests
include("test/cuda_validation.jl")

# Should output:
# ✓ CUDA functional
# ✓ Found 2 GPU(s)
# ✓ Sufficient GPU memory
# ✓ Peer access available
```

## Environment Setup

### 1. Create Configuration

```bash
# Copy example configurations
cp configs/gpu_config.toml.example configs/gpu_config.toml
cp configs/algorithm_config.toml.example configs/algorithm_config.toml

# Edit configurations as needed
nano configs/gpu_config.toml
```

### 2. Set Environment Variables

```bash
# Create .env file
cat > .env << EOF
JULIA_CUDA_MEMORY_POOL=cuda
JULIA_CUDA_SOFT_MEMORY_LIMIT=0.9
HSOF_ENV=dev
EOF
```

### 3. Optional: Python Dependencies

Some utilities require Python packages:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install numpy pandas scikit-learn matplotlib
```

## Docker Installation (Alternative)

### 1. Build Docker Image

```dockerfile
# Dockerfile provided in repository
docker build -t hsof:latest .
```

### 2. Run with GPU Support

```bash
# Requires nvidia-container-toolkit
docker run --gpus all -it \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/configs:/workspace/configs \
    hsof:latest
```

## Troubleshooting

### CUDA Not Found

```julia
# Error: CUDA.jl could not find CUDA
# Solution:
ENV["CUDA_HOME"] = "/usr/local/cuda"
using Pkg
Pkg.build("CUDA")
```

### Insufficient GPU Memory

```julia
# Error: CUDAOutOfMemoryError
# Solution: Reduce batch size in config
[gpu.cuda]
memory_limit_gb = 20  # Leave some headroom
```

### Single GPU Mode

```julia
# Warning: Only 1 GPU detected
# Solution: HSOF automatically falls back to single GPU mode
# Adjust workload in config:
[pipeline]
single_gpu_mode = true
```

### Permission Errors

```bash
# Error: Permission denied for /dev/nvidia*
# Solution: Add user to video group
sudo usermod -a -G video $USER
# Log out and back in
```

## Performance Optimization

### 1. Enable GPU Persistence

```bash
# Reduces kernel launch overhead
sudo nvidia-smi -pm 1
```

### 2. Set GPU Clocks

```bash
# Maximum performance mode
sudo nvidia-smi -pl 450  # Power limit for RTX 4090
sudo nvidia-smi --auto-boost-default=0
sudo nvidia-smi -ac 10501,2520  # Memory and graphics clocks
```

### 3. Configure Julia Threading

```bash
# Set thread count
export JULIA_NUM_THREADS=8
export JULIA_CUDA_NUM_THREADS=1
```

## Verification

Run the complete test suite:

```julia
using Pkg
Pkg.test("HSOF")

# Or specific tests:
include("test/gpu/test_device_manager.jl")
include("test/gpu/kernel_tests.jl")
```

Expected output:
```
Test Summary: | Pass  Total  Time
HSOF Tests    |  125    125  45.2s
  GPU Tests   |   42     42  12.3s
  Pipeline    |   38     38  28.5s
  Algorithms  |   45     45   4.4s
```

## Next Steps

- [Quick Start](@ref): Run your first feature selection
- [Configuration Guide](@ref): Customize HSOF settings
- [GPU Programming Tutorial](@ref): Write custom kernels