# HSOF Scaling Operations Runbook

## Overview

This runbook covers scaling operations for the HSOF system, including horizontal scaling (adding/removing replicas), vertical scaling (resource adjustments), and GPU resource scaling.

## Pre-Scaling Checklist

### 1. Capacity Planning
- [ ] Review current resource utilization
- [ ] Check GPU availability across nodes
- [ ] Verify storage capacity
- [ ] Assess network bandwidth requirements
- [ ] Review cost implications

### 2. Monitoring Setup
- [ ] Ensure monitoring dashboards are accessible
- [ ] Set up temporary alerts for scaling events
- [ ] Prepare rollback procedures
- [ ] Notify relevant teams

## Horizontal Scaling

### Scaling Up (Adding Replicas)

#### 1. Pre-Scaling Validation
```bash
# Check current deployment status
kubectl get deployment hsof-main -o wide

# Check node capacity
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check GPU availability
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"

# Verify storage capacity
df -h /data /logs /checkpoints
```

#### 2. Gradual Scaling Process
```bash
# Get current replica count
CURRENT_REPLICAS=$(kubectl get deployment hsof-main -o jsonpath='{.spec.replicas}')
echo "Current replicas: $CURRENT_REPLICAS"

# Scale up incrementally (add 1 replica at a time for safety)
NEW_REPLICAS=$((CURRENT_REPLICAS + 1))
kubectl scale deployment hsof-main --replicas=$NEW_REPLICAS

# Wait for pod to be ready
kubectl wait --for=condition=ready pod -l app=hsof --timeout=300s

# Monitor for 5 minutes
echo "Monitoring new replica for 5 minutes..."
sleep 300

# Check health and performance
curl https://hsof.company.com/health
kubectl logs -l app=hsof --tail=20 | grep -i error
```

#### 3. Validation After Scaling Up
```bash
# Check all pods are running
kubectl get pods -l app=hsof

# Verify GPU allocation
kubectl describe pods -l app=hsof | grep -A 5 "nvidia.com/gpu"

# Check resource utilization
kubectl top pods -l app=hsof

# Monitor key metrics
curl https://hsof.company.com/metrics | grep -E "(hsof_gpu_utilization|hsof_memory_usage)"

# Verify load distribution
kubectl logs -l app=hsof --tail=50 | grep "Processing started"
```

### Scaling Down (Removing Replicas)

#### 1. Pre-Scaling Down Validation
```bash
# Check current workload
curl https://hsof.company.com/metrics | grep hsof_active_jobs

# Identify least utilized pod
kubectl top pods -l app=hsof --sort-by=cpu

# Check for any running critical operations
kubectl logs -l app=hsof --tail=100 | grep -i "backup\|checkpoint\|critical"
```

#### 2. Graceful Scale Down Process
```bash
# Get current replica count
CURRENT_REPLICAS=$(kubectl get deployment hsof-main -o jsonpath='{.spec.replicas}')
echo "Current replicas: $CURRENT_REPLICAS"

# Ensure we don't scale below minimum (at least 2 for HA)
if [ $CURRENT_REPLICAS -le 2 ]; then
    echo "Warning: Already at minimum replica count"
    exit 1
fi

# Scale down by 1
NEW_REPLICAS=$((CURRENT_REPLICAS - 1))
kubectl scale deployment hsof-main --replicas=$NEW_REPLICAS

# Monitor termination process
kubectl get pods -l app=hsof -w
```

#### 3. Validation After Scaling Down
```bash
# Verify remaining pods are healthy
kubectl get pods -l app=hsof

# Check performance impact
curl https://hsof.company.com/metrics | grep hsof_stage_duration_seconds

# Monitor for increased latency or errors
kubectl logs -l app=hsof --tail=50 | grep -E "(ERROR|WARN|timeout)"
```

## Vertical Scaling (Resource Adjustments)

### CPU/Memory Scaling

#### 1. Current Resource Assessment
```bash
# Check current resource requests/limits
kubectl describe deployment hsof-main | grep -A 10 "Requests\|Limits"

# Check actual usage
kubectl top pods -l app=hsof

# Check resource pressure
kubectl describe nodes | grep -E "(cpu|memory).*pressure"
```

#### 2. Update Resource Allocation
```bash
# Update CPU requests/limits
kubectl patch deployment hsof-main -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "hsof",
            "resources": {
              "requests": {
                "cpu": "4",
                "memory": "16Gi"
              },
              "limits": {
                "cpu": "8", 
                "memory": "32Gi"
              }
            }
          }
        ]
      }
    }
  }
}'

# Monitor rollout
kubectl rollout status deployment/hsof-main

# Verify new resource allocation
kubectl describe pods -l app=hsof | grep -A 10 "Requests\|Limits"
```

### GPU Scaling

#### 1. GPU Resource Adjustment
```bash
# Check current GPU allocation
kubectl describe deployment hsof-main | grep "nvidia.com/gpu"

# Update GPU resource requests
kubectl patch deployment hsof-main -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "hsof",
            "resources": {
              "requests": {
                "nvidia.com/gpu": "2"
              },
              "limits": {
                "nvidia.com/gpu": "2"
              }
            }
          }
        ]
      }
    }
  }
}'

# Monitor rollout
kubectl rollout status deployment/hsof-main

# Verify GPU allocation
kubectl exec -it <hsof-pod> -- nvidia-smi
```

## Auto-Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

#### 1. Create HPA for CPU-based scaling
```bash
# Create HPA
kubectl create hpa hsof-main \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  --dry-run=client -o yaml > hsof-hpa.yaml

# Edit the HPA for additional metrics
cat << EOF > hsof-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hsof-main
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hsof-main
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: hsof_gpu_utilization_percent
      target:
        type: AverageValue
        averageValue: "85"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
EOF

# Apply HPA
kubectl apply -f hsof-hpa.yaml

# Monitor HPA status
kubectl get hpa hsof-main -w
```

#### 2. Vertical Pod Autoscaler (VPA) - Optional
```bash
# Create VPA for automatic resource recommendations
cat << EOF > hsof-vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: hsof-main
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hsof-main
  updatePolicy:
    updateMode: "Off"  # Start with recommendations only
  resourcePolicy:
    containerPolicies:
    - containerName: hsof
      maxAllowed:
        cpu: 16
        memory: 64Gi
      minAllowed:
        cpu: 1
        memory: 4Gi
EOF

# Apply VPA
kubectl apply -f hsof-vpa.yaml

# Check VPA recommendations
kubectl describe vpa hsof-main
```

## Node Scaling (Cluster Level)

### Adding Nodes for GPU Workloads

#### 1. Check Current Node Capacity
```bash
# Check GPU node availability
kubectl get nodes -l node-type=gpu-node

# Check resource allocation
kubectl describe nodes | grep -A 20 "Allocated resources"

# Check pending pods
kubectl get pods --field-selector=status.phase=Pending
```

#### 2. Request Additional Nodes
```bash
# For cloud environments (example with AWS EKS)
# This would typically be done through infrastructure team

# Check node group scaling
aws eks describe-nodegroup --cluster-name hsof-cluster --nodegroup-name gpu-nodes

# Scale node group (if automated scaling is configured)
aws eks update-nodegroup-config \
  --cluster-name hsof-cluster \
  --nodegroup-name gpu-nodes \
  --scaling-config minSize=2,maxSize=6,desiredSize=4
```

#### 3. Validate New Nodes
```bash
# Wait for nodes to become ready
kubectl get nodes -w

# Check GPU availability on new nodes
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu",STATUS:.status.conditions[-1].type

# Verify node labels and taints
kubectl describe nodes | grep -E "(Labels|Taints)"
```

## Scaling Monitoring & Validation

### 1. Performance Monitoring During Scaling
```bash
# Monitor key metrics during scaling
watch -n 30 'curl -s https://hsof.company.com/metrics | grep -E "(hsof_gpu_utilization|hsof_stage_duration|hsof_errors_total)"'

# Monitor resource utilization
watch -n 30 'kubectl top pods -l app=hsof; kubectl top nodes'

# Check for errors or warnings
kubectl logs -l app=hsof --tail=50 | grep -E "(ERROR|WARN)" | tail -10
```

### 2. Load Testing After Scaling
```bash
# Simple load test
for i in {1..10}; do
  curl -X POST https://hsof.company.com/api/v1/select-features \
    -H "Content-Type: application/json" \
    -d '{"dataset_size": 1000, "n_features": 100}' &
done
wait

# Monitor response times
curl https://hsof.company.com/metrics | grep hsof_request_duration_seconds
```

### 3. Cost Monitoring
```bash
# Check current resource costs (cloud-specific)
# AWS example:
aws ce get-cost-and-usage \
  --time-period Start=2023-01-01,End=2023-01-02 \
  --granularity DAILY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE

# Check resource utilization efficiency
kubectl describe nodes | grep -A 10 "Allocated resources" | grep -E "(cpu|memory|nvidia)"
```

## Troubleshooting Scaling Issues

### Common Issues and Solutions

#### 1. Pods Stuck in Pending State
```bash
# Check reason for pending
kubectl describe pods -l app=hsof | grep -A 10 Events

# Common causes and solutions:
# - Insufficient GPU resources: Add more GPU nodes
# - Resource limits: Adjust resource requests
# - Node affinity issues: Check node labels
# - Image pull issues: Check image availability
```

#### 2. Performance Degradation After Scaling
```bash
# Check for resource contention
kubectl top nodes
kubectl top pods -l app=hsof

# Check network bottlenecks
kubectl exec -it <hsof-pod> -- netstat -i
kubectl exec -it <hsof-pod> -- ss -tuln

# Check storage I/O
kubectl exec -it <hsof-pod> -- iostat -x 1 5
```

#### 3. Uneven Load Distribution
```bash
# Check pod distribution across nodes
kubectl get pods -l app=hsof -o wide

# Check for anti-affinity rules
kubectl describe deployment hsof-main | grep -A 20 affinity

# Update pod anti-affinity for better distribution
kubectl patch deployment hsof-main -p '{
  "spec": {
    "template": {
      "spec": {
        "affinity": {
          "podAntiAffinity": {
            "preferredDuringSchedulingIgnoredDuringExecution": [
              {
                "weight": 100,
                "podAffinityTerm": {
                  "labelSelector": {
                    "matchExpressions": [
                      {
                        "key": "app",
                        "operator": "In",
                        "values": ["hsof"]
                      }
                    ]
                  },
                  "topologyKey": "kubernetes.io/hostname"
                }
              }
            ]
          }
        }
      }
    }
  }
}'
```

## Rollback Procedures

### 1. Emergency Scale Down
```bash
# Immediate scale to minimum safe replicas
kubectl scale deployment hsof-main --replicas=2

# Check system stability
curl https://hsof.company.com/health
```

### 2. Resource Rollback
```bash
# Rollback to previous deployment revision
kubectl rollout undo deployment/hsof-main

# Check rollout status
kubectl rollout status deployment/hsof-main

# Verify rollback success
kubectl describe deployment hsof-main | grep -A 10 "Requests\|Limits"
```

### 3. Node Scaling Rollback
```bash
# Cordon new nodes if they're causing issues
kubectl cordon <problematic-node>

# Drain workloads from problematic nodes
kubectl drain <problematic-node> --ignore-daemonsets --delete-emptydir-data

# Scale down node group (cloud-specific)
aws eks update-nodegroup-config \
  --cluster-name hsof-cluster \
  --nodegroup-name gpu-nodes \
  --scaling-config minSize=2,maxSize=4,desiredSize=2
```

## Post-Scaling Validation

### 1. System Health Check
```bash
# Comprehensive health check
curl https://hsof.company.com/health
curl https://hsof.company.com/health/gpu
curl https://hsof.company.com/health/pipeline

# Check all pods are healthy
kubectl get pods -l app=hsof

# Verify no errors in logs
kubectl logs -l app=hsof --tail=100 | grep -i error
```

### 2. Performance Validation
```bash
# Run performance test
curl -X POST https://hsof.company.com/api/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{"test_type": "quick"}'

# Check key metrics
curl https://hsof.company.com/metrics | grep -E "(hsof_stage_duration|hsof_gpu_utilization|hsof_throughput)"
```

### 3. Cost Analysis
```bash
# Calculate cost impact of scaling changes
# Document before/after resource utilization
# Update capacity planning documentation
```

---

**Last Updated**: $(date)  
**Owner**: ML Platform Team  
**Review Schedule**: Quarterly