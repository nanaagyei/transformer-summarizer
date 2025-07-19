#!/bin/bash
# Template startup script for cloud GPU instances
# Copy this file and customize it for your deployment

set -e

echo "🚀 Starting cloud GPU setup..."

# Update system packages
apt-get update
apt-get install -y git wget curl htop

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi

# Check disk space
echo "💾 Checking disk space..."
df -h

# Clone repository (update with your repository URL)
echo "📥 Cloning repository..."
git clone https://github.com/YOUR_USERNAME/transformer-summarizer.git
cd transformer-summarizer

# Create necessary directories
mkdir -p experiments/models
mkdir -p experiments/logs

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Verify PyTorch CUDA installation
echo "✅ Verifying PyTorch CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"

echo "✅ Setup complete!"
echo "📝 You can now run training commands"

# Keep instance alive
tail -f /dev/null 