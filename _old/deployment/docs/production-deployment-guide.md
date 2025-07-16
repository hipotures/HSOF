# HSOF Production Deployment Guide

## Overview

This guide covers the complete production deployment process for the HSOF (Hybrid Search for Optimal Features) system, including infrastructure setup, monitoring configuration, and operational procedures.

## Prerequisites

### Infrastructure Requirements

- **Kubernetes Cluster**: v1.24+ with GPU node support
- **GPU Nodes**: 2+ nodes with NVIDIA RTX 4090 (24GB VRAM each)
- **Storage**: 1TB+ SSD storage for data and checkpoints
- **Network**: 10Gbps inter-node connectivity
- **CPU**: 32+ cores total across cluster
- **Memory**: 128GB+ RAM total across cluster

### Software Dependencies

- **Container Runtime**: containerd with NVIDIA container runtime
- **Monitoring**: Prometheus, Grafana, AlertManager
- **Logging**: Elasticsearch, Kibana, Fluentd
- **Service Mesh**: Istio (optional but recommended)

## Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │    Monitoring   │    │     Logging     │
│   (Ingress)     │    │   (Prometheus)  │    │ (Elasticsearch) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   HSOF      │  │   HSOF      │  │   HSOF      │            │
│  │   Pod       │  │   Pod       │  │   Pod       │            │
│  │ (GPU Node)  │  │ (GPU Node)  │  │ (CPU Node)  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Infrastructure Setup

### 1.1 Kubernetes Cluster Setup

```bash
# Ensure GPU nodes are properly labeled
kubectl label nodes gpu-node-1 node-type=gpu-node
kubectl label nodes gpu-node-2 node-type=gpu-node

# Verify GPU availability
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.\"nvidia\.com/gpu\"

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml
```

### 1.2 Namespace and RBAC Setup

```bash
# Create namespace
kubectl create namespace hsof

# Apply RBAC configuration
kubectl apply -f deployment/rbac/
```

### 1.3 Storage Setup

```bash
# Create persistent volumes for data and checkpoints
kubectl apply -f deployment/storage/persistent-volumes.yaml

# Verify storage classes
kubectl get storageclass
```

## Step 2: Configuration Management

### 2.1 Secrets Management

```bash
# Create database credentials
kubectl create secret generic hsof-db-credentials \
  --from-literal=username="hsof_user" \
  --from-literal=password="secure_password" \
  -n hsof

# Create API keys for external services
kubectl create secret generic hsof-api-keys \
  --from-literal=monitoring_key="monitoring_api_key" \
  --from-literal=logging_key="logging_api_key" \
  -n hsof
```

### 2.2 Configuration Maps

```bash
# Apply configuration maps
kubectl apply -f deployment/config/
```

## Step 3: Core Application Deployment

### 3.1 Database Deployment

```bash
# Deploy SQLite database with persistent storage
kubectl apply -f deployment/database/sqlite-deployment.yaml

# Wait for database to be ready
kubectl wait --for=condition=ready pod -l app=hsof-db -n hsof --timeout=300s
```

### 3.2 HSOF Application Deployment

```bash
# Deploy HSOF main application
kubectl apply -f deployment/kubernetes/hsof-deployment.yaml

# Deploy service
kubectl apply -f deployment/kubernetes/hsof-service.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=hsof -n hsof --timeout=600s

# Verify deployment
kubectl get pods -n hsof -o wide
kubectl logs -f deployment/hsof-main -n hsof
```

### 3.3 Ingress Configuration

```bash
# Deploy ingress controller (if not already installed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Apply HSOF ingress
kubectl apply -f deployment/ingress/hsof-ingress.yaml

# Verify ingress
kubectl get ingress -n hsof
```

## Step 4: Monitoring Setup

### 4.1 Prometheus Stack

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values deployment/monitoring/prometheus-values.yaml

# Apply HSOF-specific monitoring rules
kubectl apply -f deployment/monitoring/prometheus-alerts.yml -n monitoring
```

### 4.2 Grafana Dashboard Setup

```bash
# Import HSOF dashboards
kubectl create configmap hsof-dashboards \
  --from-file=deployment/monitoring/grafana-dashboards.json \
  -n monitoring

# Apply dashboard configuration
kubectl apply -f deployment/monitoring/grafana-dashboard-config.yaml -n monitoring
```

### 4.3 SLO Monitoring

```bash
# Deploy SLO monitoring
kubectl apply -f deployment/monitoring/slo-config.yaml -n hsof

# Verify SLO metrics
kubectl port-forward svc/slo-exporter 8080:8080 -n hsof &
curl http://localhost:8080/metrics | grep hsof_slo
```

## Step 5: Logging Infrastructure

### 5.1 Elasticsearch Setup

```bash
# Deploy Elasticsearch cluster
kubectl apply -f deployment/logging/elasticsearch-setup.yaml

# Wait for cluster to be ready
kubectl wait --for=condition=ready pod -l app=elasticsearch -n logging --timeout=600s

# Verify cluster health
kubectl port-forward svc/elasticsearch 9200:9200 -n logging &
curl http://localhost:9200/_cluster/health
```

### 5.2 Fluentd Configuration

```bash
# Deploy Fluentd for log collection
kubectl apply -f deployment/logging/fluentd-config.yaml -n hsof

# Verify log collection
kubectl logs -f daemonset/fluentd -n hsof
```

### 5.3 Kibana Setup

```bash
# Access Kibana dashboard
kubectl port-forward svc/kibana 5601:5601 -n logging &

# Configure HSOF index patterns in Kibana
# Navigate to http://localhost:5601 and create index patterns:
# - hsof-logs-*
# - hsof-errors-*
# - hsof-performance-*
```

## Step 6: Security Configuration

### 6.1 Network Policies

```bash
# Apply network policies for traffic isolation
kubectl apply -f deployment/security/network-policies.yaml -n hsof
```

### 6.2 Pod Security Standards

```bash
# Apply pod security policies
kubectl apply -f deployment/security/pod-security-policies.yaml -n hsof
```

### 6.3 Service Mesh (Optional)

```bash
# Install Istio
curl -L https://istio.io/downloadIstio | sh -
istioctl install --set values.defaultRevision=default

# Enable sidecar injection
kubectl label namespace hsof istio-injection=enabled

# Apply Istio configuration
kubectl apply -f deployment/istio/ -n hsof
```

## Step 7: Backup and Disaster Recovery

### 7.1 Backup Configuration

```bash
# Deploy backup system
kubectl apply -f deployment/backup/backup-cronjob.yaml -n hsof

# Test backup process
kubectl create job --from=cronjob/hsof-backup hsof-backup-test -n hsof
kubectl logs job/hsof-backup-test -n hsof
```

### 7.2 Disaster Recovery Setup

```bash
# Configure cross-region backup storage
kubectl apply -f deployment/backup/disaster-recovery.yaml -n hsof

# Document recovery procedures
# See: deployment/runbooks/disaster-recovery.md
```

## Step 8: Performance Tuning

### 8.1 GPU Optimization

```bash
# Apply GPU-specific configurations
kubectl apply -f deployment/performance/gpu-config.yaml -n hsof

# Verify GPU scheduling
kubectl describe nodes | grep nvidia.com/gpu
```

### 8.2 Resource Limits and Requests

```bash
# Apply resource quotas
kubectl apply -f deployment/performance/resource-quotas.yaml -n hsof

# Configure horizontal pod autoscaler
kubectl apply -f deployment/performance/hpa.yaml -n hsof
```

## Step 9: Health Checks and Validation

### 9.1 Application Health

```bash
# Check pod status
kubectl get pods -n hsof

# Test health endpoints
kubectl port-forward svc/hsof-main 8080:8080 -n hsof &
curl http://localhost:8080/health
curl http://localhost:8080/health/gpu
curl http://localhost:8080/health/pipeline
```

### 9.2 Integration Tests

```bash
# Run integration test suite
kubectl apply -f deployment/testing/integration-tests.yaml -n hsof

# Monitor test results
kubectl logs job/hsof-integration-tests -n hsof -f
```

### 9.3 Performance Benchmarks

```bash
# Run performance benchmarks
kubectl apply -f deployment/testing/benchmark-tests.yaml -n hsof

# Check benchmark results
kubectl logs job/hsof-benchmarks -n hsof
```

## Step 10: Operational Procedures

### 10.1 Monitoring Access

- **Grafana Dashboard**: https://grafana.company.com/d/hsof-prod
- **Prometheus**: https://prometheus.company.com
- **AlertManager**: https://alertmanager.company.com
- **Kibana Logs**: https://kibana.company.com/app/discover#/hsof-logs

### 10.2 Common Operations

```bash
# Scale deployment
kubectl scale deployment hsof-main --replicas=5 -n hsof

# Rolling update
kubectl set image deployment/hsof-main hsof=hsof:v1.2.0 -n hsof
kubectl rollout status deployment/hsof-main -n hsof

# Rollback if needed
kubectl rollout undo deployment/hsof-main -n hsof

# Check resource usage
kubectl top pods -n hsof
kubectl top nodes
```

### 10.3 Troubleshooting

```bash
# Check pod logs
kubectl logs -f deployment/hsof-main -n hsof

# Debug pod issues
kubectl describe pod <pod-name> -n hsof

# Check events
kubectl get events -n hsof --sort-by='.lastTimestamp'

# Access pod for debugging
kubectl exec -it <pod-name> -n hsof -- /bin/bash
```

## Step 11: Maintenance Procedures

### 11.1 Regular Updates

1. **Weekly**: Review monitoring dashboards and alerts
2. **Bi-weekly**: Update container images and security patches
3. **Monthly**: Review resource usage and scaling requirements
4. **Quarterly**: Performance optimization and capacity planning

### 11.2 Backup Verification

```bash
# Test backup restoration
kubectl apply -f deployment/testing/backup-restore-test.yaml -n hsof

# Verify backup integrity
kubectl logs job/backup-restore-test -n hsof
```

### 11.3 Security Updates

```bash
# Scan for vulnerabilities
kubectl apply -f deployment/security/vulnerability-scan.yaml -n hsof

# Review security policies
kubectl get networkpolicies -n hsof
kubectl get podsecuritypolicies -n hsof
```

## Emergency Procedures

### System Down

1. Check overall cluster health: `kubectl get nodes`
2. Check HSOF pod status: `kubectl get pods -n hsof`
3. Review recent events: `kubectl get events -n hsof`
4. Check logs: `kubectl logs -f deployment/hsof-main -n hsof`
5. Follow runbook: `deployment/runbooks/system-down.md`

### Performance Degradation

1. Check GPU utilization: Grafana GPU dashboard
2. Review error rates: Prometheus alerts
3. Check resource constraints: `kubectl top pods -n hsof`
4. Follow runbook: `deployment/runbooks/performance-degradation.md`

### Data Loss

1. Stop all HSOF pods: `kubectl scale deployment hsof-main --replicas=0 -n hsof`
2. Assess damage: Check data integrity
3. Restore from backup: Follow disaster recovery procedures
4. Verify restoration: Run integration tests
5. Resume operations: Scale pods back up

## Support and Escalation

### Primary Contacts

- **On-Call Engineer**: ml-platform-oncall@company.com
- **Infrastructure Team**: infra-oncall@company.com
- **Management Escalation**: director-ml@company.com

### Communication Channels

- **Slack**: #hsof-production, #hsof-incidents
- **Status Page**: https://status.company.com
- **Documentation**: https://internal-docs.company.com/hsof

## Post-Deployment Checklist

- [ ] All pods are running and healthy
- [ ] Health endpoints responding correctly
- [ ] Monitoring dashboards showing data
- [ ] Alerts configured and tested
- [ ] Logging working and searchable
- [ ] Backup system operational
- [ ] Performance benchmarks passed
- [ ] Security scans completed
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Runbooks accessible and current
- [ ] Emergency procedures tested

---

**Last Updated**: $(date)  
**Version**: 1.0  
**Owner**: ML Platform Team  
**Review Schedule**: Monthly