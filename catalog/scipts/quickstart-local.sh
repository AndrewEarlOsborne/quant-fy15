#!/bin/bash
set -e

echo "Beginning Install"

# Copy files to /opt
sudo mkdir -p /opt/ethereum-trading
sudo cp -r . /opt/ethereum-trading/
sudo chown -R $USER:$USER /opt/ethereum-trading

# Install systemd service
sudo cp ethereum-trading.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ethereum-trading
sudo systemctl start ethereum-trading

echo "Service installed."
echo "Status: sudo systemctl status ethereum-trading"
# echo "Logs: sudo journalctl -u ethereum-trading -f"