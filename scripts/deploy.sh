#!/bin/bash
# MemEvolve API Deployment Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Build the Docker image
build_image() {
    log_info "Building MemEvolve API Docker image..."
    docker build -t memevolve-api:latest .
    log_success "Docker image built successfully"
}

# Start services with docker-compose
start_services() {
    log_info "Starting MemEvolve API services..."

    if [ ! -f ".env" ]; then
        log_warning ".env file not found. Creating from .env.example..."
        cp .env.example .env
        log_warning "Please edit .env file with your configuration before running again."
    fi

    docker-compose up -d
    log_success "Services started successfully"

    log_info "API available at: http://localhost:8001"
    log_info "API docs at: http://localhost:8001/docs"
    log_info "Health check: curl http://localhost:8001/health"
}

# Stop services
stop_services() {
    log_info "Stopping MemEvolve API services..."
    docker-compose down
    log_success "Services stopped successfully"
}

# Show logs
show_logs() {
    log_info "Showing MemEvolve API logs..."
    docker-compose logs -f memevolve-api
}

# Show status
show_status() {
    log_info "Service Status:"
    docker-compose ps

    log_info "Health Check:"
    if curl -f http://localhost:8001/health &> /dev/null; then
        log_success "API is healthy"
        curl -s http://localhost:8001/health | python3 -m json.tool
    else
        log_error "API is not responding"
    fi
}

# Clean up
cleanup() {
    log_info "Cleaning up Docker resources..."
    docker-compose down -v
    docker rmi memevolve-api:latest 2>/dev/null || true
    log_success "Cleanup completed"
}

# Show usage
usage() {
    cat << EOF
MemEvolve API Deployment Script

Usage: $0 [COMMAND]

Commands:
    build       Build the Docker image
    start       Start the API services
    stop        Stop the API services
    restart     Restart the API services
    logs        Show API service logs
    status      Show service status and health
    cleanup     Remove containers and images
    help        Show this help message

Examples:
    $0 build && $0 start    # Build and start services
    $0 logs                   # View logs
    $0 status                 # Check status

Environment Variables (set in .env):
    MEMEVOLVE_API_HOST=0.0.0.0
    MEMEVOLVE_API_PORT=8001
    MEMEVOLVE_UPSTREAM_BASE_URL=http://llm-service:8000/v1

EOF
}

# Main script logic
main() {
    local command="${1:-help}"

    case "$command" in
        build)
            check_docker
            build_image
            ;;
        start)
            check_docker
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 2
            start_services
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"