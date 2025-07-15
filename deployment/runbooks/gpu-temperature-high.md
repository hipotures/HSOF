# GPU Temperature High Runbook

## Alert Details

**Alert Name**: GPUTemperatureHigh  
**Severity**: Critical  
**Threshold**: GPU temperature > 85°C for 2 minutes  
**Impact**: Risk of hardware damage, thermal throttling, system instability

## Immediate Actions (< 5 minutes)

### 1. Acknowledge Alert
```bash
# Check current GPU temperatures
curl https://hsof.company.com/health/gpu
```

### 2. Assess Immediate Risk
- **> 90°C**: CRITICAL - Immediate action required
- **85-90°C**: HIGH - Urgent action needed  
- **< 85°C**: Alert may be stale, verify current state

### 3. Check GPU Status
```bash
# SSH to affected node
ssh node-name

# Check GPU status with nvidia-smi
nvidia-smi

# Check detailed temperature info
nvidia-smi -q -d TEMPERATURE
```

## Diagnosis (< 10 minutes)

### 1. Identify Root Cause

#### Check GPU Utilization
```bash
# High utilization may cause thermal issues
nvidia-smi -q -d UTILIZATION

# Check if workload is abnormally intensive
kubectl logs -l app=hsof --tail=50 | grep "GPU\|utilization"
```

#### Check System Ventilation
```bash
# Check fan speeds
nvidia-smi -q -d FAN

# System temperature sensors
sensors

# Check for thermal throttling
nvidia-smi -q -d CLOCK | grep -i throttle
```

#### Check Workload Patterns
```bash
# Recent pipeline activity
curl https://hsof.company.com/metrics | grep hsof_stage_duration

# Check for stuck processes
ps aux | grep julia | grep hsof
```

### 2. Environmental Factors
- Data center cooling status
- Ambient temperature
- Hardware health (fans, heat sinks)
- Recent maintenance activities

## Mitigation (< 15 minutes)

### Immediate Actions (Temperature > 90°C)

#### 1. Emergency Workload Reduction
```bash
# Scale down HSOF deployment immediately
kubectl scale deployment hsof-main --replicas=1

# Reduce GPU workload
kubectl patch deployment hsof-main -p '{"spec":{"template":{"spec":{"containers":[{"name":"hsof","env":[{"name":"JULIA_NUM_THREADS","value":"2"},{"name":"CUDA_VISIBLE_DEVICES","value":"0"}]}]}}}}'
```

#### 2. Force Workload Migration
```bash
# Cordon the node to prevent new workloads
kubectl cordon <node-name>

# Drain workloads from affected node
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
```

### Progressive Actions (Temperature 85-90°C)

#### 1. Reduce Processing Intensity
```bash
# Reduce concurrent GPU operations
kubectl set env deployment/hsof-main MAX_CONCURRENT_BATCHES=1

# Lower GPU memory usage
kubectl set env deployment/hsof-main GPU_MEMORY_FRACTION=0.7
```

#### 2. Implement Throttling
```bash
# Add thermal protection to pipeline config
kubectl create configmap hsof-thermal-config --from-literal=thermal_throttle_threshold=80

# Update deployment to use thermal protection
kubectl patch deployment hsof-main -p '{"spec":{"template":{"spec":{"containers":[{"name":"hsof","env":[{"name":"ENABLE_THERMAL_PROTECTION","value":"true"}]}]}}}}'
```

## Monitoring & Validation

### 1. Continuous Temperature Monitoring
```bash
# Monitor temperature every 30 seconds
watch -n 30 'nvidia-smi -q -d TEMPERATURE | grep "GPU Current Temp"'

# Check Grafana dashboard
# Open: https://grafana.company.com/d/hsof-prod (GPU Temperature panel)
```

### 2. Performance Impact Assessment
```bash
# Check for thermal throttling
nvidia-smi -q -d PERFORMANCE

# Monitor pipeline performance
curl https://hsof.company.com/metrics | grep hsof_stage_duration_seconds

# Check error rates
curl https://hsof.company.com/metrics | grep hsof_errors_total
```

## Recovery Actions

### 1. Temperature Normalized (< 80°C)

#### Gradual Workload Restoration
```bash
# Slowly increase replica count
kubectl scale deployment hsof-main --replicas=2
# Wait 10 minutes, monitor temperature

kubectl scale deployment hsof-main --replicas=3
# Wait 10 minutes, monitor temperature

# Return to normal replica count if temperature stable
kubectl scale deployment hsof-main --replicas=4
```

#### Remove Throttling
```bash
# Remove thermal protection if temperature stable for 30 minutes
kubectl set env deployment/hsof-main ENABLE_THERMAL_PROTECTION-

# Restore normal GPU memory usage
kubectl set env deployment/hsof-main GPU_MEMORY_FRACTION-

# Restore normal thread count
kubectl set env deployment/hsof-main JULIA_NUM_THREADS-
```

### 2. Node Recovery
```bash
# Uncordon node once temperature stable
kubectl uncordon <node-name>

# Verify node is ready
kubectl get nodes
```

## Prevention & Follow-up

### 1. Hardware Investigation
- Schedule hardware inspection if temperature exceeded 90°C
- Check data center cooling systems
- Review recent hardware maintenance
- Validate fan operation and heat sink condition

### 2. Monitoring Improvements
```bash
# Lower temperature alert threshold if frequent alerts
# Update prometheus-alerts.yml:
# expr: hsof_gpu_temperature_celsius > 80  # Reduced from 85

# Add predictive alerting
# expr: predict_linear(hsof_gpu_temperature_celsius[10m], 300) > 85
```

### 3. Workload Optimization
- Review GPU workload scheduling
- Implement better thermal management in application
- Consider workload distribution across multiple GPUs
- Optimize CUDA kernel efficiency

### 4. Documentation
- Update incident log with root cause
- Share learnings with infrastructure team
- Update thermal management procedures if needed

## Escalation

### When to Escalate
- Temperature exceeds 95°C
- Unable to reduce temperature within 15 minutes
- Hardware appears to be malfunctioning
- Repeated thermal alerts (>3 per day)

### Escalation Path
1. **Infrastructure Team** (Slack: @infra-oncall)
2. **Data Center Operations** (for cooling issues)
3. **Hardware Vendor** (for suspected hardware failure)

## Post-Incident Actions

### Required Documentation
1. **Incident Summary**
   - Peak temperature reached
   - Duration of thermal event
   - Actions taken
   - Impact on service

2. **Root Cause Analysis**
   - Environmental factors
   - Workload patterns
   - Hardware condition
   - Process improvements

3. **Prevention Measures**
   - Monitoring improvements
   - Application changes
   - Infrastructure changes
   - Process updates

---

**Last Updated**: $(date)  
**Owner**: ML Platform Team  
**Reviewer**: Infrastructure Team