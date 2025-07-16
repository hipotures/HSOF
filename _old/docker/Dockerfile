# Multi-stage Dockerfile for HSOF with NVIDIA GPU support
# Stage 1: Base CUDA runtime
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS cuda-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV JULIA_VERSION=1.10.5
ENV JULIA_PATH=/opt/julia
ENV PATH="${JULIA_PATH}/bin:${PATH}"
ENV JULIA_DEPOT_PATH=/opt/julia_depot
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    pkg-config \
    python3 \
    python3-pip \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Julia installation
FROM cuda-base AS julia-base

# Download and install Julia
RUN mkdir -p ${JULIA_PATH} && \
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    tar -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C ${JULIA_PATH} --strip-components=1 && \
    rm julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    julia -e 'using InteractiveUtils; versioninfo()'

# Create non-root user for running the application
RUN useradd -m -u 1000 -s /bin/bash hsof && \
    mkdir -p ${JULIA_DEPOT_PATH} && \
    chown -R hsof:hsof ${JULIA_DEPOT_PATH}

# Stage 3: Julia packages installation
FROM julia-base AS julia-packages

# Switch to hsof user for package installation
USER hsof
WORKDIR /home/hsof

# Copy Project.toml and Manifest.toml for dependency resolution
COPY --chown=hsof:hsof Project.toml Manifest.toml ./

# Pre-compile Julia packages
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' && \
    julia --project=. -e 'using CUDA; CUDA.precompile_runtime()' && \
    julia --project=. -e 'using Flux, MLJ, SQLite'

# Stage 4: Application build
FROM julia-packages AS app-build

# Copy application source code
COPY --chown=hsof:hsof src/ ./src/
COPY --chown=hsof:hsof test/ ./test/
COPY --chown=hsof:hsof scripts/ ./scripts/
COPY --chown=hsof:hsof data/ ./data/

# Pre-compile application modules
RUN julia --project=. -e 'using Pkg; Pkg.precompile()'

# Stage 5: Production runtime
FROM julia-base AS production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenblas0 \
    libcurl4 \
    ca-certificates \
    nvidia-modprobe \
    && rm -rf /var/lib/apt/lists/*

# Copy Julia depot with pre-compiled packages
COPY --from=app-build --chown=hsof:hsof ${JULIA_DEPOT_PATH} ${JULIA_DEPOT_PATH}

# Create application directory
USER hsof
WORKDIR /app

# Copy application files
COPY --from=app-build --chown=hsof:hsof /home/hsof/ ./

# Copy health check and entrypoint scripts
COPY --chown=hsof:hsof docker/scripts/health_check.jl ./scripts/
COPY --chown=hsof:hsof docker/scripts/entrypoint.sh ./scripts/
RUN chmod +x ./scripts/entrypoint.sh

# Create directories for data persistence
RUN mkdir -p /app/checkpoints /app/logs /app/results /app/cache

# Expose ports for metrics and health endpoints
EXPOSE 8080 9090

# Health check command
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD julia --project=. scripts/health_check.jl || exit 1

# Set entrypoint
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Default command
CMD ["julia", "--project=.", "src/main.jl"]

# Labels for metadata
LABEL maintainer="HSOF Team"
LABEL version="1.0.0"
LABEL description="Hybrid Stage-Optimized Feature Selection with GPU Support"
LABEL cuda.version="11.8"
LABEL julia.version="${JULIA_VERSION}"