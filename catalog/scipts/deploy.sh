#!/bin/bash
set -e

echo "🚀 Deploying Ethereum Trading System"

# Check if .env exists
if [ ! -f .env ]; then
    echo ".env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t ethereum-trading:latest .

# Stop existing container if running
echo "Stopping existing container..."
docker stop ethereum-trading-system 2>/dev/null || true
docker rm ethereum-trading-system 2>/dev/null || true

# Create necessary directories on host
echo "📁 Creating directories..."
mkdir -p ./data/{raw,processed,backups}
mkdir -p ./models
mkdir -p ./logs

# Start container
echo "🏃 Starting container..."
docker run -d \
    --name ethereum-trading-system \
    --restart unless-stopped \
    --env-file .env \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    ethereum-trading:latest

echo "Deployment complete!"
echo "Check status: docker logs ethereum-trading-system"
echo "Monitor: docker exec -it ethereum-trading-system python scripts/check_status.py"
