# HSOF Operational Runbooks

This directory contains operational runbooks for managing the HSOF (Hybrid Search for Optimal Features) production system. These runbooks provide step-by-step procedures for common operational scenarios, incident response, and system maintenance.

## Quick Reference

### Emergency Contacts
- **Primary On-Call**: ML Platform Team
- **Secondary**: Infrastructure Team  
- **Escalation**: Engineering Management

### Critical Service URLs
- **Production Dashboard**: https://grafana.company.com/d/hsof-prod
- **Alerting**: https://alertmanager.company.com
- **Logs**: https://kibana.company.com/app/discover#/hsof-logs
- **Status Page**: https://status.company.com

## Runbook Index

### System Health & Monitoring
- [Health Check Failing](./health-check-failing.md)
- [System Down](./system-down.md)
- [High Memory Usage](./memory-usage-high.md)
- [High CPU Usage](./cpu-usage-high.md)
- [Disk Space Low](./disk-space-low.md)

### GPU & Performance
- [GPU Temperature High](./gpu-temperature-high.md)
- [GPU Memory Exhaustion](./gpu-memory-exhaustion.md)
- [GPU Utilization Low](./gpu-utilization-low.md)
- [GPU Utilization High](./gpu-utilization-high.md)
- [GPU Monitoring Down](./gpu-monitoring-down.md)

### Pipeline & Processing
- [Pipeline Stage Timeout](./pipeline-stage-timeout.md)
- [Feature Selection Quality Low](./quality-low.md)
- [Model Inference Latency High](./inference-latency-high.md)
- [Processing Stalled](./processing-stalled.md)
- [High Error Rate](./error-rate-high.md)

### Backup & Data
- [Backup Failed](./backup-failed.md)
- [Backup Overdue](./backup-overdue.md)
- [Backup Storage Low](./backup-storage-low.md)

### Performance & Cost
- [Throughput Low](./throughput-low.md)
- [Feature Reduction Ratio Anomaly](./reduction-ratio-anomaly.md)
- [Cost Budget Exceeded](./cost-budget-exceeded.md)

### Availability & SLO
- [High Latency](./high-latency.md)
- [Error Budget Exhausted](./error-budget-exhausted.md)

### Maintenance & Operations
- [Scaling Operations](./scaling-operations.md)
- [System Updates](./system-updates.md)
- [Incident Response](./incident-response.md)
- [Disaster Recovery](./disaster-recovery.md)

## General Procedures

### 1. Incident Response Process

1. **Acknowledge Alert** (< 5 minutes)
   - Check alert details in monitoring dashboard
   - Acknowledge alert to prevent notification spam
   - Assess severity and impact

2. **Initial Assessment** (< 10 minutes)
   - Check system status dashboard
   - Review recent deployments/changes
   - Identify affected users/services

3. **Mitigation** (< 30 minutes for P0/P1)
   - Follow specific runbook procedures
   - Implement temporary fixes if needed
   - Document actions taken

4. **Resolution & Validation**
   - Verify fix resolves issue
   - Monitor for stability
   - Update incident status

5. **Post-Incident**
   - Write incident summary
   - Schedule post-mortem if needed
   - Update runbooks if necessary

### 2. Escalation Matrix

| Severity | Initial Response | Escalation Time | Escalation Path |
|----------|------------------|-----------------|-----------------|
| P0 (Critical) | 5 minutes | 15 minutes | On-call → Manager → Director |
| P1 (High) | 15 minutes | 30 minutes | On-call → Manager |
| P2 (Medium) | 1 hour | 4 hours | On-call → Manager |
| P3 (Low) | 4 hours | Next business day | On-call |

### 3. Communication Guidelines

- **Internal**: Use #hsof-incidents Slack channel
- **External**: Update status page for user-facing issues
- **Management**: Escalate P0/P1 incidents immediately
- **Users**: Proactive communication for >15min downtime

### 4. Log Analysis

```bash
# Check recent errors
kubectl logs -f deployment/hsof-main --tail=100 | grep ERROR

# Search for specific patterns
kubectl logs deployment/hsof-main --since=1h | grep "GPU\|Memory\|Timeout"

# Check multiple pods
kubectl logs -l app=hsof --all-containers=true --since=30m
```

### 5. Quick Health Checks

```bash
# System health
curl https://hsof.company.com/health

# GPU status
curl https://hsof.company.com/health/gpu

# Pipeline status  
curl https://hsof.company.com/health/pipeline

# Metrics endpoint
curl https://hsof.company.com/metrics
```

### 6. Common Kubectl Commands

```bash
# Check pod status
kubectl get pods -l app=hsof

# Describe problematic pod
kubectl describe pod <pod-name>

# Scale deployment
kubectl scale deployment hsof-main --replicas=3

# Restart deployment
kubectl rollout restart deployment/hsof-main

# Check resource usage
kubectl top pods -l app=hsof
kubectl top nodes
```

### 7. Monitoring Queries

#### Prometheus Queries
```promql
# GPU utilization
avg(hsof_gpu_utilization_percent)

# Error rate
rate(hsof_errors_total[5m]) * 100

# Latency P95
histogram_quantile(0.95, rate(hsof_request_duration_seconds_bucket[5m]))

# Memory usage
avg(hsof_memory_usage_bytes / hsof_memory_limit_bytes) * 100
```

#### Log Queries (Elasticsearch/Kibana)
```
# Recent errors
level:ERROR AND @timestamp:[now-1h TO now]

# GPU related issues
message:GPU AND level:(ERROR OR WARN)

# Performance issues
message:(timeout OR slow OR latency) AND @timestamp:[now-30m TO now]
```

## Best Practices

### 1. Safety First
- Always verify changes in staging first
- Use feature flags for risky deployments  
- Have rollback plan ready
- Never make changes during high-traffic periods

### 2. Documentation
- Update runbooks after each incident
- Document any manual interventions
- Keep change logs for major modifications
- Share learnings with team

### 3. Monitoring
- Monitor key metrics during changes
- Set up temporary alerts for new deployments
- Use canary deployments for major updates
- Validate monitoring after system changes

### 4. Communication
- Keep stakeholders informed
- Use clear, concise language
- Provide regular updates during incidents
- Post-mortem critical issues

## Emergency Contacts

### Primary On-Call
- **Slack**: @ml-platform-oncall
- **Phone**: Via PagerDuty
- **Email**: ml-platform-oncall@company.com

### Infrastructure Team
- **Slack**: @infra-oncall  
- **Phone**: Via PagerDuty
- **Email**: infra-oncall@company.com

### Management Escalation
- **Director, ML Platform**: director-ml@company.com
- **VP Engineering**: vp-eng@company.com

---

**Last Updated**: $(date)
**Owner**: ML Platform Team
**Review Schedule**: Monthly