# Docker Compose Setup for HSOF

This document describes the multi-container Docker Compose setup for the HSOF (Hybrid Search for Optimal Features) system.

## Architecture Overview

The Docker Compose configuration includes the following services:

### Core Services

1. **hsof** - Main application container with GPU support
   - Julia runtime with CUDA.jl
   - Full access to NVIDIA GPUs
   - Connects to PostgreSQL and Redis

2. **postgres** - PostgreSQL database
   - Stores feature selection history and results
   - Persistent volume for data retention
   - Automatic initialization with schema

3. **redis** - Redis cache
   - Caches metamodel predictions
   - Stores MCTS state for fast access
   - LRU eviction policy

### Monitoring Services (Optional)

4. **prometheus** - Metrics collection
   - Scrapes metrics from all services
   - Stores time-series data

5. **grafana** - Visualization
   - Pre-configured dashboards
   - GPU and performance monitoring

## Quick Start

### Prerequisites

1. Docker and Docker Compose installed
2. NVIDIA Docker runtime (nvidia-docker2)
3. NVIDIA GPU with CUDA support

### Starting Services

```bash
# Using the helper script (recommended)
./scripts/docker-compose-helper.sh up

# Or directly with docker-compose
docker-compose up -d
```

### Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to customize:
   - GPU selection (NVIDIA_VISIBLE_DEVICES)
   - Port mappings
   - Memory limits
   - Performance parameters

### Accessing Services

- **HSOF Application**: http://localhost:8080
- **PostgreSQL**: localhost:5432 (user: hsof, password: from .env)
- **Redis**: localhost:6379
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Development Setup

The `docker-compose.override.yml` file provides additional development features:

```bash
# Start with development tools
./scripts/docker-compose-helper.sh dev-tools
```

This adds:
- **pgAdmin**: Database administration UI (http://localhost:5050)
- **Redis Commander**: Redis management UI (http://localhost:8081)
- **Pluto**: Julia notebook server (http://localhost:1234)

### Development Features

- Source code hot-reloading
- Debug logging enabled
- All ports exposed for direct access
- Additional development tools

## GPU Configuration

### Verifying GPU Access

```bash
# Check GPU availability in containers
./scripts/docker-compose-helper.sh gpu-check
```

### Multi-GPU Setup

The system automatically uses all available GPUs. To limit GPU usage:

```bash
# In .env file
NVIDIA_VISIBLE_DEVICES=0,1  # Use only GPU 0 and 1
```

### GPU Memory Management

Configure Julia CUDA memory pool in `.env`:
```bash
JULIA_CUDA_MEMORY_POOL=none  # Disable memory pool
JULIA_CUDA_MEMORY_POOL=cuda  # Use CUDA memory pool
```

## Database Management

### Accessing PostgreSQL

```bash
# Direct database shell
./scripts/docker-compose-helper.sh db-shell

# Or via docker-compose
docker-compose exec postgres psql -U hsof -d hsof
```

### Database Schema

The database is automatically initialized with tables for:
- Feature selection runs
- Stage results
- Feature importance history
- MCTS snapshots
- Metamodel cache
- Performance benchmarks

### Backup and Restore

```bash
# Backup database
docker-compose exec postgres pg_dump -U hsof hsof > backup.sql

# Restore database
docker-compose exec -T postgres psql -U hsof hsof < backup.sql
```

## Monitoring

### Prometheus Metrics

The following metrics are collected:
- GPU utilization and memory usage
- Feature processing throughput
- MCTS nodes evaluated per second
- Database and Redis performance
- Container resource usage

### Grafana Dashboards

Pre-configured dashboards include:
- GPU Performance Overview
- Feature Selection Progress
- System Resource Usage
- Database Performance

### Custom Metrics

Add custom metrics to your Julia code:
```julia
# In your Julia application
using Prometheus
gpu_util = Gauge("hsof_gpu_utilization", "GPU utilization percentage")
set!(gpu_util, current_utilization)
```

## Volumes and Persistence

### Named Volumes

- `postgres-data`: PostgreSQL database files
- `redis-data`: Redis persistence files
- `hsof-checkpoints`: Model checkpoints
- `prometheus-data`: Metrics history
- `grafana-data`: Dashboard configurations

### Bind Mounts

- `./data`: Input/output data files
- `./models`: Trained models
- `./logs`: Application logs
- `./configs`: Configuration files

## Troubleshooting

### Common Issues

1. **GPU not available**
   ```bash
   # Check NVIDIA Docker runtime
   docker info | grep nvidia
   
   # Verify GPU drivers
   nvidia-smi
   ```

2. **Container startup failures**
   ```bash
   # Check logs
   docker-compose logs hsof
   
   # Run health check manually
   docker-compose exec hsof julia scripts/health_check.jl
   ```

3. **Database connection errors**
   ```bash
   # Check PostgreSQL status
   docker-compose ps postgres
   
   # View PostgreSQL logs
   docker-compose logs postgres
   ```

### Debugging

Enable debug mode in `.env`:
```bash
LOG_LEVEL=debug
JULIA_DEBUG=HSOF
```

### Clean Restart

```bash
# Stop and remove everything (WARNING: deletes data)
./scripts/docker-compose-helper.sh clean

# Fresh start
./scripts/docker-compose-helper.sh up
```

## Production Deployment

For production deployment:

1. Use specific image tags instead of `latest`
2. Enable SSL/TLS for external connections
3. Configure proper backup strategies
4. Set resource limits in docker-compose.yml
5. Use external volume drivers for cloud storage
6. Enable authentication for monitoring services

## Performance Tuning

### Container Resources

In `docker-compose.yml`, add resource limits:
```yaml
services:
  hsof:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          cpus: '4'
          memory: 16G
```

### Network Optimization

For multi-GPU systems without NVLink:
- Use host networking mode for minimal latency
- Configure CPU affinity for NUMA optimization
- Enable SR-IOV for network acceleration

## Security Considerations

1. Change default passwords in production
2. Use secrets management for sensitive data
3. Enable firewall rules for exposed ports
4. Regular security updates for base images
5. Implement RBAC for Kubernetes deployments