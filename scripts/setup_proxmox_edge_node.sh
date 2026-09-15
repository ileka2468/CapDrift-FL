#!/bin/bash
# Edge Node Setup Script for Proxmox VE (Debian / Ubuntu LXC or VM)
# Run inside Hospital A (pve01), Hospital B (pve02), and Hospital C (pve03) containers

set -e

echo "=== Setting up Edge Client on Proxmox Node ==="

# Update package manager and install Python & system tools
apt update && apt install -y python3 python3-pip python3-venv git curl htop tcpdump

# Create project directory and virtual environment
mkdir -p /opt/fl_edge && cd /opt/fl_edge
python3 -m venv venv
source venv/bin/activate

# Install CPU-optimized PyTorch and Flower
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flwr scikit-learn pillow tqdm psutil

echo "=== Edge Client Environment Ready ==="
echo "To run the client, connect to the coordinator:"
echo "python client.py --server 192.168.30.102:8080 --node_id node_1 --data_dir data/node_1"
