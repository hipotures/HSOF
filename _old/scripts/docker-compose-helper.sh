#!/bin/bash

# Docker Compose Helper Script for HSOF
# Provides convenient commands for managing the multi-container setup

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_usage() {
    echo "Usage: $0 {up|down|restart|logs|status|gpu-check|db-shell|redis-cli|dev-tools|clean}"
    echo ""
    echo "Commands:"
    echo "  up          - Start all services"
    echo "  down        - Stop all services"
    echo "  restart     - Restart all services"
    echo "  logs        - Show logs (optional: service name)"
    echo "  status      - Show service status"
    echo "  gpu-check   - Check GPU availability in containers"
    echo "  db-shell    - Connect to PostgreSQL shell"
    echo "  redis-cli   - Connect to Redis CLI"
    echo "  dev-tools   - Start development tools (pgAdmin, Redis Commander)"
    echo "  clean       - Stop services and remove volumes (WARNING: deletes data)"
    echo ""
    echo "Examples:"
    echo "  $0 up"
    echo "  $0 logs hsof"
    echo "  $0 dev-tools"
}

check_nvidia_docker() {
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected.${NC}"
        echo "Please ensure nvidia-docker2 is installed for GPU support."
        echo "Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
}

case "$1" in
    up)
        echo -e "${GREEN}Starting HSOF services...${NC}"
        check_nvidia_docker
        
        # Create .env file if it doesn't exist
        if [ ! -f .env ]; then
            echo -e "${YELLOW}Creating .env file from template...${NC}"
            cp .env.example .env
        fi
        
        # Create required directories
        mkdir -p data models logs configs checkpoints notebooks
        
        docker-compose up -d
        echo -e "${GREEN}Services started. Run '$0 logs' to see output.${NC}"
        ;;
        
    down)
        echo -e "${YELLOW}Stopping HSOF services...${NC}"
        docker-compose down
        echo -e "${GREEN}Services stopped.${NC}"
        ;;
        
    restart)
        echo -e "${YELLOW}Restarting HSOF services...${NC}"
        docker-compose restart
        echo -e "${GREEN}Services restarted.${NC}"
        ;;
        
    logs)
        if [ -z "$2" ]; then
            docker-compose logs -f --tail=100
        else
            docker-compose logs -f --tail=100 "$2"
        fi
        ;;
        
    status)
        echo -e "${GREEN}HSOF Service Status:${NC}"
        docker-compose ps
        echo ""
        echo -e "${GREEN}Resource Usage:${NC}"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
            $(docker-compose ps -q 2>/dev/null)
        ;;
        
    gpu-check)
        echo -e "${GREEN}Checking GPU availability in containers...${NC}"
        
        # Check main HSOF container
        echo -e "\n${YELLOW}HSOF Container:${NC}"
        docker-compose exec hsof nvidia-smi || echo -e "${RED}GPU not available in HSOF container${NC}"
        
        # Check if GPUs are visible to Julia
        echo -e "\n${YELLOW}Julia GPU Check:${NC}"
        docker-compose exec hsof julia -e "using CUDA; println(\"CUDA functional: \", CUDA.functional()); println(\"GPU count: \", length(CUDA.devices()))" || echo -e "${RED}Julia CUDA check failed${NC}"
        ;;
        
    db-shell)
        echo -e "${GREEN}Connecting to PostgreSQL...${NC}"
        docker-compose exec postgres psql -U hsof -d hsof
        ;;
        
    redis-cli)
        echo -e "${GREEN}Connecting to Redis...${NC}"
        docker-compose exec redis redis-cli
        ;;
        
    dev-tools)
        echo -e "${GREEN}Starting development tools...${NC}"
        docker-compose --profile dev-tools up -d
        echo -e "${GREEN}Development tools started:${NC}"
        echo "  - pgAdmin: http://localhost:5050 (admin@hsof.local / admin)"
        echo "  - Redis Commander: http://localhost:8081"
        ;;
        
    clean)
        echo -e "${RED}WARNING: This will delete all data!${NC}"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v
            echo -e "${GREEN}Services stopped and volumes removed.${NC}"
        else
            echo "Cancelled."
        fi
        ;;
        
    *)
        print_usage
        exit 1
        ;;
esac