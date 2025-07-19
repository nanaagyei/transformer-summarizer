# Vast.ai Quick Start Guide

Based on the [official Vast.ai API documentation](https://docs.vast.ai/api/search-templates)

## 🚀 Quick Start

### 1. Setup Vast.ai Account

```bash
# 1. Sign up at https://vast.ai/console/account
# 2. Add credits to your account
# 3. Get your API key from the console
```

### 2. Install and Configure CLI

```bash
# Install Vast.ai CLI
pip install vast-ai

# Set your API key
export VAST_API_KEY=your_api_key_here

# Generate SSH key (if you don't have one)
ssh-keygen -t rsa -b 4096 -C 'your_email@example.com'
```

### 3. Search for PyTorch-Compatible Instances

```bash
# Search for RTX 4090 instances with sufficient disk space
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100'

# Search for A100 instances
vastai search offers 'gpu_name:A100 num_gpus:1 disk_space:>100'

# Search for cost-effective options
vastai search offers 'gpu_name:RTX_3090 num_gpus:1 disk_space:>100 price_per_hour<0.40'
```

### 4. Deploy Training Job

```bash
# Use our automated deployment script (RECOMMENDED)
python scripts/deploy.py --action search

# This will:
# 1. Search for available instances
# 2. Let you select one
# 3. Provide the command to create it

# Then create the instance with the selected ID
python scripts/deploy.py --action create --instance-id <selected_instance_id>

# Or manually (ADVANCED)
# First search and note the instance ID
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100'

# Then create using the instance ID (NOT the JSON file)
vastai create instance <instance_id> --ssh-key <your_ssh_key> --onstart <startup_script>
```

### 5. Monitor and Access

```bash
# List your instances
python scripts/deploy.py --action list
# or
vastai show instances

# SSH into your instance
python scripts/deploy.py --action ssh --instance-id <instance_id>
# or
vastai ssh <instance_id>

# Monitor GPU usage (on the instance)
nvidia-smi

# Check system resources
htop
df -h
```

## 🔧 Vast.ai API Best Practices

### Instance Selection

- **GPU Memory**: Minimum 8GB (RTX 4090 has 24GB)
- **Disk Space**: Minimum 50GB (we use 100GB)
- **Internet Speed**: Minimum 100Mbps
- **Price**: Set maximum price to avoid overpaying

### Search Templates

Based on the [official Vast.ai API documentation](https://docs.vast.ai/api/search-templates), here are effective search queries:

```bash
# Basic RTX 4090 search
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100'

# Cost-optimized search
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100 price_per_hour<0.50'

# High-performance search
vastai search offers 'gpu_name:A100 num_gpus:1 disk_space:>100'

# Spot instances for cost savings
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100 spot:true'

# Complex search with multiple criteria
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100 price_per_hour<0.50 verified:true'
```

### PyTorch Setup

```bash
# Use official PyTorch Docker image
pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install uv (modern Python package manager) - only if not already installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
fi

# Create virtual environment with uv - only if not exists
if [ ! -d ".venv" ]; then
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install PyTorch with CUDA support using uv
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies with uv
uv sync  # if pyproject.toml exists
# or manually:
uv add transformers datasets rouge-score tqdm wandb pyyaml numpy scikit-learn

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name()}')"
```

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Monitoring

```bash
# GPU monitoring
nvidia-smi
nvidia-smi --query-gpu=memory.total,memory.free --format=csv

# System monitoring
htop
df -h
free -h

# Training logs
tail -f experiments/logs/training.log
```

## 💰 Cost Optimization

### Spot Instances

```bash
# Use spot instances for cost savings
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100 spot:true'
```

### Price Monitoring

```bash
# Check current prices
vastai search offers 'gpu_name:RTX_4090 num_gpus:1' --raw

# Set price alerts
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 price_per_hour<0.30'
```

### Instance Management

```bash
# Destroy instance when done
python scripts/deploy.py --action destroy --instance-id <instance_id>
# or
vastai destroy instance <instance_id>

# Stop instance (keeps data)
vastai stop instance <instance_id>

# Start stopped instance
vastai start instance <instance_id>
```

## 🐛 Troubleshooting

### Common Issues

**Instance won't start:**

```bash
# Check credit balance
vastai show credit

# Check instance status
python scripts/deploy.py --action list
# or
vastai show instances

# Check logs
vastai show instance <instance_id> --raw
```

**SSH connection issues:**

```bash
# Verify SSH key
cat ~/.ssh/id_rsa.pub

# Test SSH connection
ssh -i ~/.ssh/id_rsa root@instance_ip
```

**PyTorch CUDA issues:**

```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch if needed using uv
uv remove torch torchvision torchaudio
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Training fails:**

```bash
# Check system resources
nvidia-smi
free -h
df -h

# Check training logs
tail -f experiments/logs/training.log
tail -f experiments/logs/system_monitor.log
```

## 📊 Performance Tips

### GPU Optimization

- Use mixed precision training (FP16)
- Optimize batch size for your GPU memory
- Use gradient accumulation for larger effective batch sizes
- Monitor GPU utilization with `nvidia-smi`

### Data Loading

- Use multiple workers for data loading
- Enable pin memory for faster GPU transfer
- Use prefetch factor for better throughput
- Cache datasets when possible

### Cost Management

- Use spot instances for non-critical training
- Monitor instance usage regularly
- Set up billing alerts
- Destroy instances when not in use

## 🔗 Useful Commands

### Instance Management

```bash
# List all instances
python scripts/deploy.py --action list
# or
vastai show instances

# Show instance details
vastai show instance <instance_id>

# SSH into instance
python scripts/deploy.py --action ssh --instance-id <instance_id>
# or
vastai ssh <instance_id>

# Destroy instance
python scripts/deploy.py --action destroy --instance-id <instance_id>
# or
vastai destroy instance <instance_id>

# Copy files to/from instance
vastai copy <instance_id>:/path/to/file /local/path
```

### Search and Filtering

```bash
# Search by GPU type
vastai search offers 'gpu_name:RTX_4090'

# Search by price
vastai search offers 'price_per_hour<0.50'

# Search by disk space
vastai search offers 'disk_space>100'

# Complex search
vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100 price_per_hour<0.50'
```

### Monitoring

```bash
# Show credit balance
vastai show credit

# Show account info
vastai show account

# Show bids (for spot instances)
vastai show bids
```

## 🚀 Automated Deployment

Our deployment script automates the entire process:

```bash
# 1. Search and select instance
python scripts/deploy.py --action search

# 2. Create instance (after selecting from search)
python scripts/deploy.py --action create --instance-id <selected_id>

# 3. SSH into instance
python scripts/deploy.py --action ssh --instance-id <instance_id>

# 4. Monitor instance
python scripts/deploy.py --action monitor --instance-id <instance_id>

# 5. Destroy when done
python scripts/deploy.py --action destroy --instance-id <instance_id>
```

## 📚 Additional Resources

- [Official Vast.ai API Documentation](https://docs.vast.ai/api/search-templates)
- [Vast.ai PyTorch Documentation](https://docs.vast.ai/pytorch)
- [Vast.ai Console](https://vast.ai/console/)
- [Vast.ai CLI Documentation](https://docs.vast.ai/cli/)

---

**Happy Training on Vast.ai! 🚀☁️**
