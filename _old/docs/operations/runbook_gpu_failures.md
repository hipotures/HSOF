# GPU Failure Runbook

## Overview
This runbook provides procedures for handling GPU failures in the HSOF multi-GPU system.

## Detection

### Symptoms
- CUDA errors in logs: `CUDA_ERROR_*`
- GPU not responding to health checks
- Kernel timeouts
- Sudden drop in GPU utilization to 0%
- Temperature readings unavailable
- Memory allocation failures

### Monitoring Alerts
- `gpu_X_heartbeat_failure`: GPU not responding to heartbeat
- `gpu_X_cuda_error`: CUDA operation failed
- `gpu_X_timeout`: Kernel execution timeout

## Diagnosis

### 1. Check GPU Status
```bash
# Check if GPU is visible
nvidia-smi

# Check detailed GPU info
nvidia-smi -q -d MEMORY,UTILIZATION,TEMPERATURE,POWER,CLOCK,COMPUTE,ECC,ERRORS

# Check system logs
dmesg | grep -i nvidia
journalctl -u hsof-gpu -n 100
```

### 2. Verify CUDA Functionality
```julia
using CUDA
CUDA.versioninfo()
CUDA.functional()

# Test each GPU
for i in 0:length(CUDA.devices())-1
    try
        device!(i)
        arr = CUDA.zeros(Float32, 1000)
        CUDA.@sync arr .= 1.0f0
        println("GPU $i: OK")
    catch e
        println("GPU $i: FAILED - $e")
    end
end
```

### 3. Check PCIe Status
```bash
# List PCIe devices
lspci | grep -i nvidia

# Check PCIe link status
sudo lspci -vv -s [device_id] | grep -i width
```

## Recovery Procedures

### Scenario 1: Transient GPU Error

**Symptoms**: Occasional CUDA errors, GPU recovers after reset

**Actions**:
1. Log the error with full context
2. Reset the affected GPU:
   ```julia
   device!(affected_gpu_id)
   CUDA.device_reset!()
   ```
3. Reinitialize GPU memory and state
4. Resume operations with the same GPU configuration

### Scenario 2: Single GPU Failure

**Symptoms**: One GPU completely unresponsive, other GPUs functional

**Actions**:
1. Isolate the failed GPU:
   ```julia
   # In the fault tolerance module
   mark_gpu_failed!(health_monitor, failed_gpu_id)
   ```

2. Redistribute workload:
   ```julia
   # Automatic redistribution
   redistribute_work!(engine, failed_gpu_id)
   ```

3. Update configuration:
   ```bash
   # Exclude failed GPU
   export CUDA_VISIBLE_DEVICES="0,2,3"  # Skip GPU 1
   ```

4. Continue in degraded mode with remaining GPUs

5. Monitor performance impact:
   - Check if scaling efficiency drops below threshold
   - Verify memory pressure on remaining GPUs
   - Monitor PCIe bandwidth saturation

### Scenario 3: Multiple GPU Failures

**Symptoms**: Multiple GPUs failing simultaneously

**Actions**:
1. **IMMEDIATE**: Switch to single-GPU fallback mode
2. Save current state/checkpoints
3. Investigate common cause:
   - Power supply issues
   - Thermal problems
   - Driver corruption
   - PCIe bus errors

4. System-wide remediation:
   ```bash
   # Restart GPU driver
   sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
   sudo modprobe nvidia
   sudo modprobe nvidia_uvm
   sudo modprobe nvidia_drm
   sudo modprobe nvidia_modeset
   ```

### Scenario 4: GPU Memory Corruption

**Symptoms**: ECC errors, inconsistent results, memory allocation failures

**Actions**:
1. Enable ECC if not already enabled:
   ```bash
   sudo nvidia-smi -e 1  # Enable ECC
   sudo reboot  # Required for ECC change
   ```

2. Run memory tests:
   ```bash
   # Use NVIDIA's GPU memory test
   cuda-memcheck julia your_script.jl
   ```

3. If errors persist:
   - Mark GPU as failed
   - Schedule hardware replacement
   - Continue with remaining GPUs

## Preventive Measures

### 1. Temperature Management
```bash
# Set GPU fan profile
sudo nvidia-settings -a "[gpu:0]/GPUFanControlState=1"
sudo nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=80"

# Set power limit (watts)
sudo nvidia-smi -pl 350  # RTX 4090 default is 450W
```

### 2. Regular Health Checks
```julia
# Add to monitoring
function gpu_health_check()
    for gpu_id in 0:num_gpus-1
        try
            device!(gpu_id)
            # Memory test
            test_array = CUDA.rand(Float32, 10_000_000)
            sum_result = sum(test_array)
            
            # Compute test
            CUDA.@sync test_array .= test_array .* 2.0f0
            
            # Temperature check
            # (Would use NVML for actual temperature)
            
            @info "GPU $gpu_id healthy"
        catch e
            @error "GPU $gpu_id health check failed" exception=e
            # Trigger alert
        end
    end
end
```

### 3. Automatic Recovery Script
```julia
# In fault_tolerance.jl
function auto_recover_gpu!(health_monitor, gpu_id)
    max_attempts = 3
    
    for attempt in 1:max_attempts
        @info "Attempting GPU recovery" gpu=gpu_id attempt=attempt
        
        try
            # Reset device
            device!(gpu_id)
            CUDA.device_reset!()
            
            # Reinitialize
            test_allocation = CUDA.zeros(Float32, 1000)
            CUDA.@sync test_allocation .= 1.0f0
            
            # If we get here, recovery succeeded
            @info "GPU recovered" gpu=gpu_id
            update_gpu_status!(health_monitor, gpu_id, GPU_HEALTHY)
            return true
            
        catch e
            @warn "Recovery attempt failed" gpu=gpu_id attempt=attempt exception=e
            sleep(2^attempt)  # Exponential backoff
        end
    end
    
    @error "GPU recovery failed after $max_attempts attempts" gpu=gpu_id
    return false
end
```

## Post-Recovery Validation

### 1. Verify GPU Functionality
```julia
# Run comprehensive GPU tests
include("test/gpu/test_gpu_health.jl")
run_gpu_validation_suite()
```

### 2. Check Performance Metrics
- Verify scaling efficiency returns to normal (>85%)
- Confirm memory bandwidth utilization
- Check PCIe transfer rates
- Validate synchronization latency

### 3. Update Documentation
- Log the incident with:
  - Timestamp
  - Symptoms observed
  - Root cause (if determined)
  - Recovery actions taken
  - Time to recovery
  - Performance impact

## Emergency Contacts

- **On-call Engineer**: Check rotation schedule
- **Hardware Support**: Vendor support line
- **NVIDIA Enterprise Support**: (if applicable)

## Appendix: Common CUDA Error Codes

| Error Code | Description | Typical Cause |
|-----------|-------------|---------------|
| CUDA_ERROR_OUT_OF_MEMORY | Out of memory | Memory leak or overallocation |
| CUDA_ERROR_NOT_INITIALIZED | Driver not initialized | Driver issue |
| CUDA_ERROR_INVALID_DEVICE | Invalid device ID | GPU offline |
| CUDA_ERROR_ECC_UNCORRECTABLE | Uncorrectable ECC error | Hardware failure |
| CUDA_ERROR_HARDWARE_STACK_ERROR | Hardware stack error | GPU fault |
| CUDA_ERROR_ILLEGAL_INSTRUCTION | Illegal instruction | Kernel bug or corruption |
| CUDA_ERROR_MISALIGNED_ADDRESS | Misaligned address | Memory access bug |
| CUDA_ERROR_INVALID_PTX | Invalid PTX | Compilation issue |

## Related Documents
- [Performance Degradation Runbook](runbook_performance_degradation.md)
- [Memory Issues Runbook](runbook_memory_issues.md)
- [Network/PCIe Issues Runbook](runbook_network_issues.md)