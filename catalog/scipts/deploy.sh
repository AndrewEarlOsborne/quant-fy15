#!/bin/bash
set -e

# Enhanced Deployment Script for Ethereum Trading System
LOGFILE="logs/deployment.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create logs directory
mkdir -p logs

# Logging function
log_step() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$1] $2" | tee -a "$LOGFILE"
}

log_step "INIT" "=== Ethereum Trading System Deployment Starting ==="
log_step "INIT" "Script location: $SCRIPT_DIR"

# Check if .env exists
if [ ! -f .env ]; then
    log_step "ERROR" ".env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi
log_step "CONFIG" ".env file found and loaded"

# Build Docker image
log_step "BUILD" "Building Docker image..."
docker build -t ethereum-trading:latest . 2>&1 | tee -a "$LOGFILE" || {
    log_step "ERROR" "Docker build failed"
    exit 1
}
log_step "BUILD" "Docker image built successfully"

# Stop existing container if running
log_step "CLEANUP" "Stopping existing container..."
docker stop ethereum-trader-system 2>/dev/null || true
docker rm ethereum-trader-system 2>/dev/null || true
log_step "CLEANUP" "Existing container cleanup completed"

# Create necessary directories on host
log_step "DIRS" "Creating required directories..."
mkdir -p ./data/{raw,processed,backups}
mkdir -p ./models
mkdir -p ./logs
log_step "DIRS" "Directories created successfully"

# Start container with enhanced logging
log_step "START" "Starting container..."
docker run -d \
    --name ethereum-trader-system \
    --restart unless-stopped \
    --env-file .env \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    ethereum-trading:latest 2>&1 | tee -a "$LOGFILE" || {
    log_step "ERROR" "Container startup failed"
    exit 1
}

# Verify container is running
sleep 3
CONTAINER_STATUS=$(docker inspect --format='{{.State.Status}}' ethereum-trader-system 2>/dev/null || echo "not_found")
log_step "VERIFY" "Container status: $CONTAINER_STATUS"

if [ "$CONTAINER_STATUS" = "running" ]; then
    log_step "SUCCESS" "=== Deployment completed successfully ==="
    log_step "SUCCESS" "Container: ethereum-trader-system is running"
    log_step "SUCCESS" "Check status: docker logs ethereum-trader-system"
    log_step "SUCCESS" "Monitor: docker exec -it ethereum-trader-system python scripts/check_status.py"
    log_step "SUCCESS" "Deployment logs: $LOGFILE"
else
    log_step "ERROR" "Container failed to start properly"
    log_step "ERROR" "Check container logs: docker logs ethereum-trader-system"
    exit 1
fi
