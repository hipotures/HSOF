# HSOF Kubernetes Deployment

Production-ready Kubernetes manifests for deploying HSOF (Hybrid Search for Optimal Features) with GPU support, monitoring, and scalability.

## Prerequisites

### Required Tools
- **kubectl** (v1.24+) - Kubernetes command-line tool
- **kustomize** (v4.5+) - Kubernetes configuration management
- **Access to Kubernetes cluster** with:
  - GPU nodes with NVIDIA drivers and nvidia-device-plugin
  - Storage classes: `fast-ssd`, `shared-storage`, `standard`
  - NVIDIA Container Toolkit configured

### GPU Requirements
- Nodes with `nvidia.com/gpu.present=true` label
- At least 2 GPUs per node (preferably RTX 4090 or equivalent)
- NVIDIA drivers v515+ and CUDA 11.8+
- 24GB+ GPU memory per device

## Quick Start

### 1. Install Dependencies
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/
```

### 2. Deploy to Staging
```bash
# Validate manifests
./k8s/deploy.sh --validate --environment staging

# Preview deployment
./k8s/deploy.sh --dry-run --environment staging

# Apply deployment
./k8s/deploy.sh --apply --environment staging
```

### 3. Deploy to Production
```bash
# Deploy with monitoring
./k8s/deploy.sh --apply --environment production

# Check status
kubectl get pods -n hsof-prod
kubectl get services -n hsof-prod
```

## Architecture

### Directory Structure
```
k8s/
├── base/                          # Base Kubernetes resources
│   ├── namespace.yaml            # Namespace and RBAC
│   ├── configmap.yaml            # Application configuration
│   ├── persistentvolume.yaml     # Storage claims
│   ├── deployment.yaml           # Main application deployment
│   ├── service.yaml              # Service definitions
│   ├── hpa.yaml                  # Horizontal Pod Autoscaler
│   ├── pdb.yaml                  # Pod Disruption Budget
│   ├── networkpolicy.yaml        # Network security policies
│   └── kustomization.yaml        # Base kustomization
├── overlays/
│   ├── production/               # Production environment
│   │   ├── kustomization.yaml   # Production overrides
│   │   ├── production-ingress.yaml
│   │   └── production-monitoring.yaml
│   └── staging/                  # Staging environment
│       └── kustomization.yaml   # Staging overrides
├── deploy.sh                     # Deployment script
└── README.md                     # This file
```

### Key Components

#### 1. GPU-Optimized Deployment
- **Resource Limits**: `nvidia.com/gpu: 2` per pod
- **Node Affinity**: Ensures scheduling on GPU-capable nodes
- **Tolerations**: Handles GPU node taints
- **Memory**: 16-64GB RAM depending on environment
- **CPU**: 4-16 cores with proper resource quotas

#### 2. Storage Configuration
- **Model Storage**: 50-200GB for metamodels and checkpoints
- **Checkpoint Storage**: 100-500GB for MCTS state and results
- **Data Storage**: 200GB shared storage for datasets
- **Logs Storage**: 20GB for application logs

#### 3. Networking & Security
- **Services**: API (port 8000), Health (8080), Metrics (9090)
- **Ingress**: TLS termination, rate limiting, authentication
- **NetworkPolicy**: Restricted pod-to-pod communication
- **RBAC**: Minimal permissions for service accounts

#### 4. Monitoring & Observability
- **Prometheus Integration**: ServiceMonitor and PodMonitor
- **Alert Rules**: GPU, performance, and availability alerts
- **Health Checks**: Startup, liveness, and readiness probes
- **Metrics Export**: Comprehensive GPU and pipeline metrics

#### 5. High Availability
- **HPA**: Auto-scaling based on CPU, memory, and GPU utilization
- **PDB**: Ensures minimum availability during updates
- **Anti-Affinity**: Spreads pods across nodes
- **Rolling Updates**: Zero-downtime deployments

## Configuration

### Environment Variables
Key configuration options in ConfigMaps:

| Variable | Description | Default |
|----------|-------------|---------|
| `JULIA_NUM_THREADS` | Julia thread count | 8 |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | "0,1" |
| `HSOF_STAGE1_TIMEOUT` | Stage 1 timeout (seconds) | 30 |
| `HSOF_MAX_MEMORY_GB` | Memory limit | 32 |
| `HSOF_PROMETHEUS_ENABLED` | Enable metrics | true |

### GPU Configuration
```toml
[gpu]
device_count = 2
memory_pool_size_gb = 8
memory_fraction = 0.8

[performance]
target_utilization = 0.85
memory_warning_threshold = 0.9
```

### Storage Classes
Ensure the following storage classes exist in your cluster:
- `fast-ssd`: High-performance SSD storage for models
- `shared-storage`: Network-attached storage for data
- `standard`: Standard storage for logs

## Environments

### Base
- Single replica
- Minimal resources
- Development configuration
- No ingress or monitoring

### Staging
- 1 replica, can scale to 2
- Reduced resources (8-16GB RAM)
- Debug logging enabled
- Basic monitoring

### Production
- 2 replicas, can scale to 5
- Full resources (32-64GB RAM)
- Optimized configuration
- Complete monitoring stack
- TLS ingress with authentication

## Deployment Operations

### Scaling
```bash
# Manual scaling
kubectl scale deployment hsof-main -n hsof-prod --replicas=3

# Check HPA status
kubectl get hpa -n hsof-prod
```

### Updates
```bash
# Update image
kubectl set image deployment/hsof-main hsof-main=hsof:v1.1.0 -n hsof-prod

# Check rollout status
kubectl rollout status deployment/hsof-main -n hsof-prod

# Rollback if needed
kubectl rollout undo deployment/hsof-main -n hsof-prod
```

### Monitoring
```bash
# Check pod status
kubectl get pods -n hsof-prod -o wide

# View logs
kubectl logs -f deployment/hsof-main -n hsof-prod

# Check metrics endpoint
kubectl port-forward svc/hsof-metrics 9090:9090 -n hsof-prod
curl http://localhost:9090/metrics
```

### Debugging
```bash
# Exec into pod
kubectl exec -it deployment/hsof-main -n hsof-prod -- /bin/bash

# Check GPU access
kubectl exec -it deployment/hsof-main -n hsof-prod -- nvidia-smi

# View events
kubectl get events -n hsof-prod --sort-by='.lastTimestamp'
```

## Troubleshooting

### Common Issues

#### 1. Pod Stuck in Pending
```bash
# Check node resources
kubectl describe node <node-name>

# Check events
kubectl describe pod <pod-name> -n hsof-prod
```

**Possible causes:**
- No GPU nodes available
- Insufficient resources
- Storage class not found
- Image pull issues

#### 2. GPU Not Detected
```bash
# Check GPU device plugin
kubectl get pods -n kube-system | grep nvidia

# Verify node labels
kubectl get nodes -l nvidia.com/gpu.present=true
```

**Solutions:**
- Install NVIDIA device plugin
- Verify node labels
- Check NVIDIA drivers

#### 3. High Memory Usage
```bash
# Check memory usage
kubectl top pods -n hsof-prod
kubectl get hpa -n hsof-prod
```

**Solutions:**
- Increase memory limits
- Check for memory leaks
- Adjust batch sizes in config

#### 4. Slow Performance
```bash
# Check GPU utilization
kubectl exec -it deployment/hsof-main -n hsof-prod -- nvidia-smi

# View metrics
kubectl port-forward svc/hsof-metrics 9090:9090 -n hsof-prod
```

**Solutions:**
- Verify GPU utilization >80%
- Check Stage 1 timeout <30s
- Review metamodel batch sizes

## Monitoring Dashboards

### Prometheus Alerts
- **GPU Utilization Low**: <50% for 5 minutes
- **GPU Memory High**: >90% for 2 minutes
- **Inference Latency High**: >100ms P95 for 5 minutes
- **Stage 1 Timeout**: >30s P90 for 1 minute

### Grafana Metrics
Access metrics at `https://metrics.hsof.example.com/metrics`:
- `hsof_gpu_utilization`
- `hsof_stage_duration_seconds`
- `hsof_model_inference_duration_seconds`
- `hsof_features_processed_total`

## Security

### Network Policies
- Pod-to-pod communication restricted
- Ingress from monitoring namespace allowed
- Database access limited to specific ports

### RBAC
- Minimal service account permissions
- Read-only access to pods and configmaps
- No cluster-wide permissions

### Container Security
- Non-root user (UID 1000)
- Read-only root filesystem where possible
- Security context with dropped capabilities

## Performance Tuning

### GPU Optimization
- Memory pooling with 80% allocation
- Dual GPU utilization >85%
- CUDA context persistence

### Memory Management
- 32-64GB RAM allocation
- Memory warnings at 90%
- Automatic garbage collection

### Pipeline Performance
- Stage 1: <30s for 5000 features
- Stage 2: >80% GPU utilization
- Stage 3: >0.95 correlation accuracy

## Production Checklist

- [ ] GPU nodes labeled and ready
- [ ] Storage classes configured
- [ ] TLS certificates installed
- [ ] Monitoring stack deployed
- [ ] Backup system configured
- [ ] Resource quotas set
- [ ] Network policies applied
- [ ] RBAC permissions verified
- [ ] Health checks passing
- [ ] Performance targets met

## Support

For deployment issues:
1. Check pod logs: `kubectl logs -f deployment/hsof-main -n hsof-prod`
2. Verify GPU access: `kubectl exec -it deployment/hsof-main -n hsof-prod -- nvidia-smi`
3. Review metrics: Access `/metrics` endpoint
4. Check cluster events: `kubectl get events -n hsof-prod`

For configuration questions, refer to the main HSOF documentation.