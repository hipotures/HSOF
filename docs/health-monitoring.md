# Health Monitoring System

The HSOF health monitoring system provides comprehensive health checks and monitoring for all system components including GPUs, models, and pipeline stages.

## Overview

The health monitoring system consists of:
- HTTP health check endpoints
- GPU monitoring with CUDA/NVML integration  
- Model health tracking
- Pipeline component monitoring
- Automatic recovery mechanisms
- Configurable thresholds and alerts

## Health Status Levels

The system uses four health status levels:

1. **HEALTHY** - All components operating normally
2. **WARNING** - Degraded performance but still operational
3. **CRITICAL** - Major issues requiring immediate attention
4. **UNKNOWN** - Unable to determine health status

## Health Check Endpoints

### Main Health Endpoint
```
GET /health
```

Returns aggregated health status of all components:
```json
{
  "overall_status": "HEALTHY",
  "components": {
    "gpu": {
      "status": "HEALTHY",
      "message": "All GPUs healthy",
      "details": {...}
    },
    "model": {
      "status": "HEALTHY", 
      "message": "Model healthy",
      "details": {...}
    },
    "pipeline": {
      "status": "HEALTHY",
      "message": "Pipeline healthy", 
      "details": {...}
    }
  },
  "timestamp": "2025-01-15T12:00:00"
}
```

### GPU Health Endpoint
```
GET /health/gpu
```

Returns detailed GPU health information:
```json
{
  "gpu_count": 2,
  "gpus": [
    {
      "device_id": 0,
      "available": true,
      "memory_used_gb": 8.5,
      "memory_total_gb": 24.0,
      "memory_percentage": 35.4,
      "temperature_celsius": 65.0,
      "utilization_percent": 87.5,
      "power_watts": 250.0
    }
  ]
}
```

### Model Health Endpoint
```
GET /health/model
```

Returns metamodel health status:
```json
{
  "metamodel_loaded": true,
  "metamodel_version": "1.2.0",
  "inference_ready": true,
  "last_inference_time": "2025-01-15T11:59:45",
  "average_inference_ms": 45.2,
  "error_rate": 0.001
}
```

### Pipeline Health Endpoint
```
GET /health/pipeline
```

Returns pipeline component status:
```json
{
  "stage1_operational": true,
  "stage2_operational": true,
  "stage3_operational": true,
  "database_connected": true,
  "redis_connected": true,
  "filesystem_accessible": true
}
```

## Configuration

Health check thresholds can be configured via environment variables or the config file:

```julia
# Default thresholds
HEALTH_CONFIG = Dict(
    "gpu_memory_warning_threshold" => 80.0,     # percentage
    "gpu_memory_critical_threshold" => 95.0,
    "gpu_temp_warning_threshold" => 80.0,       # Celsius
    "gpu_temp_critical_threshold" => 90.0,
    "model_error_rate_warning" => 0.05,         # 5%
    "model_error_rate_critical" => 0.10,        # 10%
    "inference_latency_warning_ms" => 100.0,
    "inference_latency_critical_ms" => 500.0
)
```

## Recovery Mechanisms

The system includes automatic recovery for transient failures:

### GPU Recovery
- Automatic CUDA context reset on GPU errors
- Verification of GPU functionality after reset
- Fallback to CPU processing if GPU recovery fails

### Model Recovery
- Automatic model reload on inference failures
- Checkpoint restoration for corrupted models
- Error rate tracking with automatic recovery triggers

### Pipeline Recovery
- Component restart on failure detection
- Database reconnection with exponential backoff
- Redis connection pooling with automatic recovery

## Integration with Main Application

### Using Health Hooks

The health monitoring system provides hooks for pipeline stages to report their status:

```julia
using HSOF.HealthIntegration.Hooks

# Report stage start
on_stage_start(1)

# Report stage completion
on_stage_complete(1, duration_seconds)

# Report stage error
on_stage_error(1, exception)

# Report model inference
on_model_inference(inference_time_ms)

# Report model loaded
on_model_loaded("v1.2.0")
```

### Starting Health Services

```julia
using HSOF.HealthIntegration

# Start health monitoring services
health_task, server_task = start_health_services(port=8080, host="0.0.0.0")
```

## Docker Integration

The health monitoring system integrates with Docker health checks:

```yaml
healthcheck:
  test: ["CMD", "julia", "-e", "include(\"scripts/health_check.jl\")"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

## Prometheus Metrics

The `/metrics` endpoint exposes metrics in Prometheus format:

```
# HELP hsof_gpu_utilization GPU utilization percentage
# TYPE hsof_gpu_utilization gauge
hsof_gpu_utilization{gpu="0"} 87.5
hsof_gpu_utilization{gpu="1"} 92.3

# HELP hsof_gpu_memory_used_bytes GPU memory usage in bytes
# TYPE hsof_gpu_memory_used_bytes gauge
hsof_gpu_memory_used_bytes{gpu="0"} 9126805504
hsof_gpu_memory_used_bytes{gpu="1"} 10737418240

# HELP hsof_model_inference_duration_seconds Model inference latency
# TYPE hsof_model_inference_duration_seconds histogram
hsof_model_inference_duration_seconds_bucket{le="0.01"} 450
hsof_model_inference_duration_seconds_bucket{le="0.05"} 890
hsof_model_inference_duration_seconds_bucket{le="0.1"} 950
```

## Monitoring Best Practices

1. **Set Appropriate Thresholds**: Adjust warning/critical thresholds based on your hardware and workload
2. **Monitor Trends**: Use Grafana dashboards to identify performance trends
3. **Automate Responses**: Configure alerts for critical health states
4. **Regular Health Checks**: Ensure health endpoints are monitored by external systems
5. **Log Aggregation**: Correlate health events with application logs

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   - Verify NVIDIA drivers are installed
   - Check CUDA installation
   - Ensure nvidia-docker runtime is configured

2. **Database Connection Failed**
   - Verify PostgreSQL is running
   - Check connection string in environment variables
   - Ensure network connectivity between containers

3. **High Memory Usage Warnings**
   - Review batch sizes in configuration
   - Check for memory leaks in custom code
   - Consider increasing GPU memory limits

### Debug Mode

Enable debug logging for detailed health check information:

```bash
export JULIA_DEBUG=HSOF.Health
```

## Performance Impact

The health monitoring system is designed for minimal performance impact:
- Health checks run asynchronously
- GPU metrics use cached NVML data
- Database checks use connection pooling
- Configurable check intervals to balance monitoring vs. performance