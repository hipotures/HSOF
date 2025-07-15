# HSOF Docker Container

This directory contains the Docker configuration for running HSOF (Hybrid Stage-Optimized Feature Selection) with NVIDIA GPU support.

## Prerequisites

- Docker Engine 20.10+ with Docker Compose v2
- NVIDIA Container Toolkit (nvidia-docker2)
- NVIDIA GPU drivers 470.57+ (for CUDA 11.8)
- At least 32GB system RAM
- 2x NVIDIA GPUs with 8GB+ VRAM each (RTX 4090 recommended)

## Quick Start

### Build the Docker Image

```bash
# Build the production image
docker-compose build hsof

# Or build directly with Docker
docker build -f docker/Dockerfile -t hsof:latest .
```

### Run the Container

```bash
# Run with docker-compose (recommended)
docker-compose up hsof

# Run with specific GPU devices
docker-compose run -e CUDA_VISIBLE_DEVICES=0,1 hsof

# Run in interactive mode
docker-compose --profile dev up hsof-dev
```

### Run Benchmarks

```bash
# Run performance benchmarks
docker-compose --profile benchmark up hsof-benchmark

# Run tests
docker-compose --profile test up hsof-test
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | all | GPU devices to use (e.g., "0,1") |
| `JULIA_NUM_THREADS` | nproc | Number of Julia threads |
| `JULIA_CUDA_MEMORY_POOL` | binned | CUDA memory pool type |
| `HSOF_LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARN, ERROR) |
| `HSOF_MAX_MEMORY_GB` | 16 | Maximum memory usage in GB |
| `HSOF_RUN_MODE` | default | Run mode (default, benchmark, test, interactive, server) |
| `HSOF_STAGE1_THREADS` | JULIA_NUM_THREADS | Threads for Stage 1 |
| `HSOF_STAGE2_GPU_BATCH_SIZE` | 1000 | GPU batch size for Stage 2 |
| `HSOF_STAGE3_ENSEMBLE_SIZE` | 10 | Ensemble size for Stage 3 |

### Volumes

The container uses the following volume mounts:

- `/app/data` - Input datasets
- `/app/checkpoints` - Model checkpoints
- `/app/results` - Output results
- `/app/logs` - Application logs
- `/app/cache` - Temporary cache files

### Ports

- `8080` - HTTP API endpoint (server mode)
- `9090` - Prometheus metrics endpoint

## Run Modes

### Default Mode
```bash
docker run --rm --gpus all hsof:latest
```
Runs the main HSOF pipeline with default settings.

### Interactive Mode
```bash
docker run --rm -it --gpus all -e HSOF_RUN_MODE=interactive hsof:latest
```
Starts an interactive Julia REPL with the project loaded.

### Server Mode
```bash
docker run --rm -d --gpus all -p 8080:8080 -p 9090:9090 \
  -e HSOF_RUN_MODE=server hsof:latest
```
Runs HSOF as a web service with API and metrics endpoints.

### Benchmark Mode
```bash
docker run --rm --gpus all -e HSOF_RUN_MODE=benchmark \
  hsof:latest --stages gpu,memory,latency --gpu
```
Runs performance benchmarks.

## GPU Configuration

### Single GPU
```bash
docker run --rm --gpus '"device=0"' hsof:latest
```

### Multiple GPUs
```bash
docker run --rm --gpus '"device=0,1"' hsof:latest
```

### All GPUs
```bash
docker run --rm --gpus all hsof:latest
```

## Health Checks

The container includes a comprehensive health check that validates:

- GPU availability and CUDA functionality
- Memory availability (host and device)
- Pipeline component availability
- Environment configuration

Check health status:
```bash
docker exec hsof-main julia --project=/app /app/scripts/health_check.jl
```

## Development

### Building for Development

```bash
# Build development image with source mounting
docker-compose --profile dev build hsof-dev
```

### Running with Local Source

```bash
# Mount local source code for development
docker-compose --profile dev up hsof-dev
```

### Shell Access

```bash
# Get a shell in the running container
docker exec -it hsof-main /bin/bash

# Run as root for debugging
docker exec -it -u root hsof-main /bin/bash
```

## Troubleshooting

### GPU Not Detected

1. Verify NVIDIA Container Toolkit installation:
```bash
nvidia-docker version
```

2. Check GPU availability:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
```

3. Ensure Docker daemon has nvidia runtime:
```bash
docker info | grep nvidia
```

### Out of Memory Errors

1. Increase memory limits in docker-compose.yml
2. Reduce batch sizes via environment variables
3. Use fewer GPUs with CUDA_VISIBLE_DEVICES

### Permission Issues

Ensure volumes have correct permissions:
```bash
# Fix permissions on host
sudo chown -R 1000:1000 ./data ./checkpoints ./results ./logs
```

### Build Cache Issues

Clear Docker build cache:
```bash
docker builder prune -a
docker-compose build --no-cache hsof
```

## Performance Optimization

### Multi-Stage Build Benefits

- Base image: ~2.5GB (CUDA runtime + system deps)
- Julia packages: ~1.2GB (precompiled)
- Final image: ~4.2GB total
- Build time: ~15-20 minutes (with cache)

### Runtime Optimization

- Use `JULIA_NUM_THREADS` for CPU parallelism
- Adjust `HSOF_STAGE2_GPU_BATCH_SIZE` for GPU memory
- Enable `JULIA_CUDA_MEMORY_POOL=binned` for better GPU memory management

## Security Considerations

- Container runs as non-root user `hsof` (UID 1000)
- Minimal runtime dependencies
- No SSH or unnecessary services
- Read-only root filesystem compatible
- Health checks don't expose sensitive data

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Build and Test HSOF
  run: |
    docker-compose build hsof
    docker-compose --profile test up --exit-code-from hsof-test hsof-test
```

### Kubernetes Deployment

See `/k8s` directory for Kubernetes manifests with GPU support.