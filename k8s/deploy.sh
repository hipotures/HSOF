#!/bin/bash

# HSOF Kubernetes Deployment Script
# Supports deployment to different environments using Kustomize

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE=""
ENVIRONMENT="base"
DRY_RUN=false
APPLY=false
VALIDATE=true
WAIT_TIMEOUT="300s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
log() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy HSOF to Kubernetes using Kustomize

OPTIONS:
    -e, --environment ENV    Environment to deploy (base, staging, production) [default: base]
    -n, --namespace NS       Override namespace
    -d, --dry-run           Show what would be deployed without applying
    -a, --apply             Apply the deployment
    -v, --validate          Validate manifests (default: true)
    --no-validate           Skip manifest validation
    -w, --wait TIMEOUT      Wait for deployment rollout [default: 300s]
    -h, --help              Show this help message

EXAMPLES:
    $0 --dry-run --environment staging
    $0 --apply --environment production
    $0 --apply --environment staging --namespace hsof-dev
    $0 --validate --environment base

EOF
}

# Function to validate environment
validate_environment() {
    local env=$1
    local env_dir="${SCRIPT_DIR}/overlays/${env}"
    
    if [[ "$env" == "base" ]]; then
        return 0
    fi
    
    if [[ ! -d "$env_dir" ]]; then
        log $RED "Error: Environment '$env' not found at $env_dir"
        log $YELLOW "Available environments:"
        find "${SCRIPT_DIR}/overlays" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; 2>/dev/null || true
        exit 1
    fi
}

# Function to validate kubectl and kustomize
validate_tools() {
    log $BLUE "Validating required tools..."
    
    if ! command -v kubectl &> /dev/null; then
        log $RED "Error: kubectl is required but not installed"
        exit 1
    fi
    
    if ! command -v kustomize &> /dev/null; then
        log $RED "Error: kustomize is required but not installed"
        log $YELLOW "Install with: curl -s \"https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh\" | bash"
        exit 1
    fi
    
    # Test kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log $RED "Error: Cannot connect to Kubernetes cluster"
        log $YELLOW "Please ensure kubectl is configured and cluster is accessible"
        exit 1
    fi
    
    log $GREEN "✓ All required tools are available"
}

# Function to validate GPU nodes (for production)
validate_gpu_nodes() {
    local env=$1
    
    if [[ "$env" == "production" ]] || [[ "$env" == "staging" ]]; then
        log $BLUE "Validating GPU nodes..."
        
        local gpu_nodes
        gpu_nodes=$(kubectl get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | wc -l || echo "0")
        
        if [[ "$gpu_nodes" -eq 0 ]]; then
            log $YELLOW "Warning: No GPU nodes found in cluster for environment '$env'"
            log $YELLOW "HSOF requires nodes with nvidia.com/gpu.present=true label"
            
            if [[ "$env" == "production" ]]; then
                read -p "Continue anyway? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 1
                fi
            fi
        else
            log $GREEN "✓ Found $gpu_nodes GPU node(s)"
        fi
    fi
}

# Function to build manifests
build_manifests() {
    local env=$1
    local manifest_dir
    
    if [[ "$env" == "base" ]]; then
        manifest_dir="${SCRIPT_DIR}/base"
    else
        manifest_dir="${SCRIPT_DIR}/overlays/${env}"
    fi
    
    log $BLUE "Building manifests for environment: $env"
    
    if ! kustomize build "$manifest_dir"; then
        log $RED "Error: Failed to build manifests"
        exit 1
    fi
}

# Function to validate manifests
validate_manifests() {
    local env=$1
    local temp_file
    temp_file=$(mktemp)
    
    log $BLUE "Validating manifests..."
    
    if ! build_manifests "$env" > "$temp_file"; then
        rm -f "$temp_file"
        exit 1
    fi
    
    # Validate with kubectl
    if ! kubectl apply --dry-run=client -f "$temp_file" &> /dev/null; then
        log $RED "Error: Manifest validation failed"
        kubectl apply --dry-run=client -f "$temp_file"
        rm -f "$temp_file"
        exit 1
    fi
    
    # Additional validations
    local issues=0
    
    # Check for required resources
    if ! grep -q "kind: Namespace" "$temp_file"; then
        log $YELLOW "Warning: No namespace resource found"
        ((issues++))
    fi
    
    if ! grep -q "nvidia.com/gpu" "$temp_file"; then
        log $YELLOW "Warning: No GPU resources requested"
        ((issues++))
    fi
    
    if ! grep -q "kind: PersistentVolumeClaim" "$temp_file"; then
        log $YELLOW "Warning: No persistent storage configured"
        ((issues++))
    fi
    
    rm -f "$temp_file"
    
    if [[ $issues -gt 0 ]]; then
        log $YELLOW "Found $issues potential issues in manifests"
    else
        log $GREEN "✓ Manifest validation passed"
    fi
}

# Function to apply manifests
apply_manifests() {
    local env=$1
    local manifest_dir
    
    if [[ "$env" == "base" ]]; then
        manifest_dir="${SCRIPT_DIR}/base"
    else
        manifest_dir="${SCRIPT_DIR}/overlays/${env}"
    fi
    
    log $BLUE "Applying manifests for environment: $env"
    
    # Apply with server-side validation
    if ! kustomize build "$manifest_dir" | kubectl apply -f -; then
        log $RED "Error: Failed to apply manifests"
        exit 1
    fi
    
    # Get the namespace from the manifests
    local ns
    ns=$(kustomize build "$manifest_dir" | grep -A 5 "kind: Namespace" | grep "name:" | head -1 | awk '{print $2}' || echo "hsof")
    
    if [[ -n "$NAMESPACE" ]]; then
        ns="$NAMESPACE"
    fi
    
    log $GREEN "✓ Manifests applied successfully"
    
    # Wait for deployment if requested
    if [[ "$WAIT_TIMEOUT" != "0" ]]; then
        log $BLUE "Waiting for deployment rollout (timeout: $WAIT_TIMEOUT)..."
        
        if kubectl rollout status deployment/hsof-main -n "$ns" --timeout="$WAIT_TIMEOUT"; then
            log $GREEN "✓ Deployment rolled out successfully"
        else
            log $RED "Error: Deployment rollout failed or timed out"
            log $YELLOW "Check deployment status with: kubectl get pods -n $ns"
            exit 1
        fi
    fi
}

# Function to show deployment status
show_status() {
    local env=$1
    local ns
    
    # Get namespace
    if [[ -n "$NAMESPACE" ]]; then
        ns="$NAMESPACE"
    elif [[ "$env" == "production" ]]; then
        ns="hsof-prod"
    elif [[ "$env" == "staging" ]]; then
        ns="hsof-staging"
    else
        ns="hsof"
    fi
    
    log $BLUE "Deployment Status for environment: $env"
    echo
    
    # Check if namespace exists
    if ! kubectl get namespace "$ns" &> /dev/null; then
        log $YELLOW "Namespace '$ns' does not exist"
        return
    fi
    
    log $BLUE "Pods:"
    kubectl get pods -n "$ns" -o wide
    echo
    
    log $BLUE "Services:"
    kubectl get services -n "$ns"
    echo
    
    log $BLUE "PVCs:"
    kubectl get pvc -n "$ns"
    echo
    
    log $BLUE "Recent Events:"
    kubectl get events -n "$ns" --sort-by='.lastTimestamp' | tail -10
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -a|--apply)
            APPLY=true
            shift
            ;;
        -v|--validate)
            VALIDATE=true
            shift
            ;;
        --no-validate)
            VALIDATE=false
            shift
            ;;
        -w|--wait)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log $RED "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log $GREEN "HSOF Kubernetes Deployment"
    log $BLUE "Environment: $ENVIRONMENT"
    
    # Validate environment
    validate_environment "$ENVIRONMENT"
    
    # Validate tools
    validate_tools
    
    # Validate GPU nodes for production/staging
    validate_gpu_nodes "$ENVIRONMENT"
    
    # Validate manifests if requested
    if [[ "$VALIDATE" == true ]]; then
        validate_manifests "$ENVIRONMENT"
    fi
    
    # Handle different modes
    if [[ "$DRY_RUN" == true ]]; then
        log $BLUE "DRY RUN: Showing manifests that would be applied"
        build_manifests "$ENVIRONMENT"
    elif [[ "$APPLY" == true ]]; then
        apply_manifests "$ENVIRONMENT"
        show_status "$ENVIRONMENT"
    else
        log $YELLOW "No action specified. Use --dry-run to preview or --apply to deploy"
        log $YELLOW "Use --help for more options"
        exit 1
    fi
}

# Run main function
main