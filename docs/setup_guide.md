# Setup Guide

This guide will help you set up the transformer summarizer project for your own use.

## 🔐 Sensitive Configuration

The following files contain sensitive information and are not included in the repository:

- SSH keys (`vastai*`, `*.pem`, `*.pub`)
- Cloud provider configurations (`*_job.json`, `*_setup.json`)
- Cost and billing information (`cost_estimator.json`)

These files are stored in `.local_config/` directory and ignored by git.

## 🚀 Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/transformer-summarizer.git
cd transformer-summarizer
```

### 2. Set Up Python Environment

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 3. Configure Cloud Deployment (Optional)

If you want to deploy to cloud GPU providers:

#### Vast.ai Setup

1. Copy the template configuration:

   ```bash
   cp configs/vast_ai_template.json vast_job.json
   ```

2. Edit `vast_job.json` with your settings:

   - Replace `YOUR_SSH_PUBLIC_KEY_HERE` with your SSH public key
   - Update the repository URL
   - Adjust GPU type, disk space, and price limits

3. Generate SSH key pair (if needed):
   ```bash
   ssh-keygen -t rsa -b 4096 -f vastai -C "your-email@example.com" -N ""
   ```

#### Other Cloud Providers

- Copy `configs/cloud_setup_template.sh` and customize for your provider
- Update the startup script with your repository URL and requirements

### 4. Run Training

#### Local Training

```bash
# CPU training
python scripts/train.py --config configs/training_config_cpu.yaml

# MPS training (Apple Silicon)
python scripts/train_mps.py --config configs/training_config_mps.yaml
```

#### Cloud Training

```bash
# Deploy to cloud
python scripts/deploy.py --action search
python scripts/deploy.py --action create --instance-id <INSTANCE_ID>

# SSH into instance and run training
ssh -i vastai.pem -p <PORT> root@<HOST>
cd transformer-summarizer
python scripts/train_remote.py --config configs/training_config_cloud.yaml
```

## 📁 Project Structure

```
transformer-summarizer/
├── src/                    # Source code
│   └── transformer_summarizer/
│       ├── models/         # Model implementations
│       ├── data/           # Data processing
│       ├── training/       # Training utilities
│       └── utils/          # Utility functions
├── configs/                # Configuration files
├── scripts/                # Training and deployment scripts
├── experiments/            # Training artifacts (gitignored)
├── data/                   # Datasets (gitignored)
├── docs/                   # Documentation
└── tests/                  # Unit tests
```

## 🔧 Configuration

### Training Configuration

- `configs/training_config_cpu.yaml` - CPU-optimized training
- `configs/training_config_mps.yaml` - Apple Silicon training
- `configs/training_config_cloud.yaml` - Cloud GPU training

### Model Configuration

- `configs/model_config_cpu.yaml` - CPU-optimized model
- `configs/model_config_mps.yaml` - Apple Silicon model
- `configs/model_config.yaml` - Default model configuration

## 📊 Monitoring

The project supports Weights & Biases for experiment tracking. Set up your W&B account and configure the project name in the training configs.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA not available**: Make sure you have the correct PyTorch version installed
2. **Memory issues**: Reduce batch size in the training config
3. **SSH connection failed**: Check your SSH key configuration and instance status

### Getting Help

- Check the logs in `experiments/logs/`
- Review the configuration files
- Check GPU availability with `nvidia-smi`
