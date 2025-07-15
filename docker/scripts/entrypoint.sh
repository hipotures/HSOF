#!/bin/bash
# Entrypoint script for HSOF Docker container
# Handles environment setup, GPU configuration, and startup validation

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo "=============================================="
echo "   HSOF - Hybrid Stage-Optimized Feature     "
echo "         Selection System v1.0.0              "
echo "=============================================="
echo ""

# Check if running as correct user
if [ "$(whoami)" != "hsof" ]; then
    log_warn "Not running as 'hsof' user, this may cause permission issues"
fi

# GPU Configuration
log_info "Configuring GPU environment..."

# Check for NVIDIA runtime
if [ -f /usr/bin/nvidia-smi ]; then
    log_info "NVIDIA runtime detected"
    nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits || {
        log_error "Failed to query GPU information"
    }
else
    log_warn "NVIDIA runtime not available - GPU features will be disabled"
fi

# Set CUDA environment variables if not already set
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Configure GPU visibility
if [ -n "$GPU_DEVICE_IDS" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_DEVICE_IDS
    log_info "Using GPU devices: $CUDA_VISIBLE_DEVICES"
elif [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Default to all available GPUs
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//' || echo "0")
    log_info "Auto-detected GPU devices: $CUDA_VISIBLE_DEVICES"
fi

# Julia Configuration
log_info "Configuring Julia environment..."

# Set Julia specific environment variables
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-$(nproc)}
export JULIA_CUDA_MEMORY_POOL=${JULIA_CUDA_MEMORY_POOL:-binned}
export JULIA_CUDA_SOFT_MEMORY_LIMIT=${JULIA_CUDA_SOFT_MEMORY_LIMIT:-0.9}

log_info "Julia threads: $JULIA_NUM_THREADS"
log_info "CUDA memory pool: $JULIA_CUDA_MEMORY_POOL"

# Application Configuration
log_info "Configuring application settings..."

# Set default environment variables for HSOF
export HSOF_LOG_LEVEL=${HSOF_LOG_LEVEL:-INFO}
export HSOF_MAX_MEMORY_GB=${HSOF_MAX_MEMORY_GB:-16}
export HSOF_CHECKPOINT_DIR=${HSOF_CHECKPOINT_DIR:-/app/checkpoints}
export HSOF_RESULTS_DIR=${HSOF_RESULTS_DIR:-/app/results}
export HSOF_CACHE_DIR=${HSOF_CACHE_DIR:-/app/cache}

# Performance tuning
export HSOF_STAGE1_THREADS=${HSOF_STAGE1_THREADS:-$JULIA_NUM_THREADS}
export HSOF_STAGE2_GPU_BATCH_SIZE=${HSOF_STAGE2_GPU_BATCH_SIZE:-1000}
export HSOF_STAGE3_ENSEMBLE_SIZE=${HSOF_STAGE3_ENSEMBLE_SIZE:-10}

# Create required directories
log_info "Setting up directories..."
for dir in "$HSOF_CHECKPOINT_DIR" "$HSOF_RESULTS_DIR" "$HSOF_CACHE_DIR" /app/logs; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" || log_error "Failed to create directory: $dir"
    fi
    # Ensure proper permissions
    if [ -w "$dir" ]; then
        log_info "Directory ready: $dir"
    else
        log_error "Directory not writable: $dir"
    fi
done

# Validate environment
log_info "Validating environment..."

# Check for Project.toml
if [ ! -f "/app/Project.toml" ]; then
    log_error "Project.toml not found in /app"
    exit 1
fi

# Run health check
log_info "Running health check..."
if julia --project=/app /app/scripts/health_check.jl > /tmp/health_check.json 2>&1; then
    log_info "Health check passed"
    cat /tmp/health_check.json | jq -r '.summary' 2>/dev/null || cat /tmp/health_check.json
else
    log_error "Health check failed"
    cat /tmp/health_check.json
    exit 1
fi

# Handle different run modes
case "${HSOF_RUN_MODE:-default}" in
    "benchmark")
        log_info "Starting in benchmark mode..."
        exec julia --project=/app /app/test/benchmarks/benchmark_runner.jl "$@"
        ;;
    "test")
        log_info "Starting in test mode..."
        exec julia --project=/app -e 'using Pkg; Pkg.test()' "$@"
        ;;
    "interactive")
        log_info "Starting interactive Julia session..."
        exec julia --project=/app -i "$@"
        ;;
    "server")
        log_info "Starting HSOF server mode..."
        export HSOF_SERVER_PORT=${HSOF_SERVER_PORT:-8080}
        export HSOF_METRICS_PORT=${HSOF_METRICS_PORT:-9090}
        exec julia --project=/app /app/src/server.jl "$@"
        ;;
    *)
        # Default mode - run the main application
        log_info "Starting HSOF pipeline..."
        
        # Check if command line arguments were provided
        if [ $# -eq 0 ]; then
            log_info "No arguments provided, running with defaults"
            exec julia --project=/app /app/src/main.jl
        else
            # Pass through all arguments
            exec "$@"
        fi
        ;;
esac