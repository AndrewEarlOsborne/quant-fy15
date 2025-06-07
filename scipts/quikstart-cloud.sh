#!/bin/bash
set -e

# Simple cloud deployment script for a single VPS/EC2 instance
echo "Deploying to Cloud Instance"

# Variables
INSTANCE_IP=${INSTANCE_IP:-"server-ip"}
SSH_KEY=${SSH_KEY:-"~/.ssh/id_rsa"}
REMOTE_USER=${REMOTE_USER:-"ubuntu"}

echo "Deploying to: $REMOTE_USER@$INSTANCE_IP"

echo "Copying files..."
rsync -avz --exclude='data/' --exclude='logs/' --exclude='models/' \
    -e "ssh -i $SSH_KEY" \
    ./ $REMOTE_USER@$INSTANCE_IP:~/ethereum-trading/

# Run deployment on remote server
echo "Running remote deployment..."
ssh -i $SSH_KEY $REMOTE_USER@$INSTANCE_IP << 'EOF'
cd ~/ethereum-trading

if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

sudo docker stop ethereum-trading-system 2>/dev/null || true
sudo docker rm ethereum-trading-system 2>/dev/null || true

sudo docker build -t ethereum-trading:latest .
sudo ./deploy.sh

echo "Cloud deployment complete!"
EOF

echo "Deployment to cloud instance complete!"