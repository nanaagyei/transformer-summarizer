#!/usr/bin/env python3
"""
Cloud Training Script for Transformer Summarizer

This script handles training on cloud GPU providers:
- Vast.ai (cost-effective, pay-per-use)
- Modal (serverless, easy deployment)
- JarvisLabs (specialized ML platform)

Usage:
    python scripts/cloud_train.py --provider vast --config configs/training_config_mps.yaml
    python scripts/cloud_train.py --provider modal --config configs/training_config_mps.yaml
    python scripts/cloud_train.py --provider jarvis --config configs/training_config_mps.yaml
"""

from src.transformer_summarizer.utils.device_optimization import get_optimal_device
from src.transformer_summarizer.data.dataset import SummarizationDataset
from src.transformer_summarizer.training.trainer import TransformerTrainer
import argparse
import os
import sys
import yaml
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudTrainer:
    """Manages training on cloud GPU providers"""

    def __init__(self, provider: str, config_path: str):
        self.provider = provider.lower()
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.project_root = Path(__file__).parent.parent

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_vast_ai(self) -> Dict[str, Any]:
        """Setup Vast.ai training environment based on official documentation"""
        logger.info("🚀 Setting up Vast.ai training environment...")

        # Vast.ai configuration based on official PyTorch guide
        vast_config = {
            "image": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",  # Official PyTorch template
            "gpu_count": 1,
            "gpu_type": "RTX_4090",  # Cost-effective option with 24GB VRAM
            "disk_space": 100,  # GB - increased for datasets and checkpoints
            "price_max": 0.50,  # $/hr max
            "ssh_key": self._get_ssh_key(),
            "startup_script": self._generate_vast_startup_script(),
            "requirements": self._get_requirements(),
            "environment": {
                "cuda_version": "12.1",
                "pytorch_version": "2.1.0",
                "min_gpu_memory": "8GB",  # Vast.ai recommendation
                "min_disk_space": "50GB",  # Vast.ai recommendation
                "min_internet_speed": "100Mbps"  # Vast.ai recommendation
            }
        }

        return vast_config

    def setup_modal(self) -> Dict[str, Any]:
        """Setup Modal training environment"""
        logger.info("🚀 Setting up Modal training environment...")

        modal_config = {
            "gpu": "T4",  # or "A100", "H100" based on needs
            "memory": 16000,  # MB
            "cpu": 4,
            "timeout": 3600,  # 1 hour
            "requirements": self._get_requirements(),
            "entrypoint": "python scripts/train_remote.py",
        }

        return modal_config

    def setup_jarvis(self) -> Dict[str, Any]:
        """Setup JarvisLabs training environment"""
        logger.info("🚀 Setting up JarvisLabs training environment...")

        jarvis_config = {
            "instance_type": "gpu.1x",  # Adjust based on needs
            "gpu_type": "RTX_4090",
            "framework": "PyTorch",
            "python_version": "3.11",
            "requirements": self._get_requirements(),
            "startup_script": self._generate_jarvis_startup_script(),
        }

        return jarvis_config

    def _get_ssh_key(self) -> str:
        """Get SSH public key for cloud access"""
        ssh_key_path = Path.home() / ".ssh" / "id_rsa.pub"
        if ssh_key_path.exists():
            return ssh_key_path.read_text().strip()
        else:
            logger.warning(
                "SSH key not found. Please generate one with: ssh-keygen -t rsa")
            return ""

    def _get_requirements(self) -> str:
        """Get project requirements for uv"""
        requirements_path = self.project_root / "pyproject.toml"
        if requirements_path.exists():
            return requirements_path.read_text()
        else:
            # Fallback requirements for uv
            return """[project]
name = "transformer-summarizer"
version = "0.1.0"
description = "Transformer-based text summarization model"
requires-python = ">=3.8"
dependencies = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.35.0",
    "datasets>=2.14.0",
    "rouge-score>=0.1.2",
    "tqdm>=4.65.0",
    "wandb>=0.15.0",
    "pyyaml>=6.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
]"""

    def _generate_vast_startup_script(self) -> str:
        """Generate startup script for Vast.ai based on official documentation"""
        return f"""#!/bin/bash
# Vast.ai startup script for transformer summarizer training
# Based on official Vast.ai PyTorch documentation

set -e  # Exit on any error

echo "🚀 Starting Vast.ai PyTorch training setup..."

# Update system packages
apt-get update
apt-get install -y git wget curl htop nvtop

# Check GPU availability (Vast.ai recommendation)
echo "🔍 Checking GPU availability..."
nvidia-smi
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.total,memory.free --format=csv

# Check disk space (Vast.ai recommendation)
echo "💾 Checking disk space..."
df -h

# Check internet speed (Vast.ai recommendation)
echo "🌐 Checking internet speed..."
curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python3 -

# Clone repository
echo "📥 Cloning transformer summarizer repository..."
git clone https://github.com/nanaagyei/transformer-summarizer.git
cd transformer-summarizer

# Install uv (modern Python package manager) - only if not already installed
if ! command -v uv &> /dev/null; then
    echo "🐍 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
fi

# Create virtual environment with uv - only if not exists
if [ ! -d ".venv" ]; then
    echo "🔧 Setting up Python environment with uv..."
    uv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install PyTorch with CUDA support using uv (Vast.ai recommendation)
echo "🔥 Installing PyTorch with CUDA support using uv..."
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies using uv
echo "📦 Installing project dependencies with uv..."
if [ -f "pyproject.toml" ]; then
    uv sync
else
    # Fallback: install dependencies manually
    uv add transformers datasets rouge-score tqdm wandb pyyaml numpy scikit-learn
fi

# Verify PyTorch CUDA installation (Vast.ai recommendation)
echo "✅ Verifying PyTorch CUDA installation..."
python3 -c "import torch; print(f'PyTorch version: {{torch.__version__}}'); print(f'CUDA available: {{torch.cuda.is_available()}}'); print(f'CUDA version: {{torch.version.cuda}}'); print(f'GPU count: {{torch.cuda.device_count()}}'); print(f'Current GPU: {{torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}}')"

# Setup environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

# Create output directories
mkdir -p experiments/remote_training
mkdir -p experiments/models
mkdir -p experiments/logs

# Monitor system resources (Vast.ai recommendation)
echo "📊 Starting system monitoring..."
(
    while true; do
        echo "=== $(date) ===" >> experiments/logs/system_monitor.log
        nvidia-smi >> experiments/logs/system_monitor.log 2>&1
        free -h >> experiments/logs/system_monitor.log 2>&1
        df -h >> experiments/logs/system_monitor.log 2>&1
        sleep 300  # Log every 5 minutes
    done
) &

# Start training with proper error handling
echo "🚀 Starting transformer training..."
python scripts/train_remote.py --config {self.config_path} --output-dir experiments/remote_training

# Save final results
echo "💾 Saving final results..."
cp -r experiments/remote_training/* /root/shared/ 2>/dev/null || echo "No shared directory found"

# Keep instance alive for debugging (Vast.ai recommendation)
echo "✅ Training completed. Keeping instance alive for debugging..."
echo "📝 You can SSH into this instance to check results"
echo "📁 Results are saved in: experiments/remote_training/"
echo "📊 System logs are in: experiments/logs/"

# Keep the instance running
tail -f /dev/null
"""

    def _generate_jarvis_startup_script(self) -> str:
        """Generate startup script for JarvisLabs"""
        return f"""#!/bin/bash
# JarvisLabs startup script for transformer summarizer training

# Clone repository
git clone https://github.com/nanaagyei/transformer-summarizer.git
cd transformer-summarizer

# Install uv (modern Python package manager) - only if not already installed
if ! command -v uv &> /dev/null; then
    echo "🐍 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
fi

# Create virtual environment with uv - only if not exists
if [ ! -d ".venv" ]; then
    echo "🔧 Setting up Python environment with uv..."
    uv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install PyTorch with CUDA support using uv
echo "🔥 Installing PyTorch with CUDA support using uv..."
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies using uv
echo "📦 Installing project dependencies with uv..."
if [ -f "pyproject.toml" ]; then
    uv sync
else
    # Fallback: install dependencies manually
    uv add transformers datasets rouge-score tqdm wandb pyyaml numpy scikit-learn
fi

# Setup environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Start training
python scripts/train_remote.py --config {self.config_path}
"""

    def deploy_to_vast(self, config: Dict[str, Any]):
        """Deploy training job to Vast.ai based on official documentation"""
        logger.info("📤 Deploying to Vast.ai...")

        # Create Vast.ai job configuration
        job_config = {
            "image": config["image"],
            "gpu_count": config["gpu_count"],
            "gpu_type": config["gpu_type"],
            "disk_space": config["disk_space"],
            "price_max": config["price_max"],
            "ssh_key": config["ssh_key"],
            "startup_script": config["startup_script"],
            "environment": config["environment"],
            "onstart": config["startup_script"],  # Vast.ai specific field
        }

        # Save job config
        job_file = self.project_root / "vast_job.json"
        with open(job_file, 'w') as f:
            json.dump(job_config, f, indent=2)

        logger.info(f"✅ Vast.ai job config saved to {job_file}")
        logger.info("📋 Next steps based on Vast.ai PyTorch documentation:")
        logger.info("")
        logger.info("1️⃣ Install Vast.ai CLI:")
        logger.info("   pip install vast-ai")
        logger.info("")
        logger.info("2️⃣ Set your API key:")
        logger.info("   export VAST_API_KEY=your_api_key_here")
        logger.info(
            "   # Get your API key from: https://vast.ai/console/account")
        logger.info("")
        logger.info("3️⃣ Generate SSH key (if you don't have one):")
        logger.info("   ssh-keygen -t rsa -b 4096 -C 'your_email@example.com'")
        logger.info("")
        logger.info("4️⃣ Search for available instances:")
        logger.info(
            "   vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100'")
        logger.info("")
        logger.info("5️⃣ Create instance using the job config:")
        logger.info("   vastai create instance --from-json vast_job.json")
        logger.info("")
        logger.info("6️⃣ SSH into your instance:")
        logger.info("   vastai ssh instance_id")
        logger.info("")
        logger.info("7️⃣ Monitor your instance:")
        logger.info("   vastai show instances")
        logger.info("")
        logger.info("8️⃣ Destroy instance when done:")
        logger.info("   vastai destroy instance instance_id")
        logger.info("")
        logger.info(
            "🔗 Vast.ai PyTorch Documentation: https://docs.vast.ai/pytorch")
        logger.info("🔗 Vast.ai Console: https://vast.ai/console/")
        logger.info("")
        logger.info("💡 Pro Tips:")
        logger.info("   • Use 'vastai search offers' to find the best deals")
        logger.info("   • Monitor GPU utilization with 'nvidia-smi'")
        logger.info("   • Check system logs in experiments/logs/")
        logger.info("   • Use spot instances for cost savings")

    def deploy_to_modal(self, config: Dict[str, Any]):
        """Deploy training job to Modal"""
        logger.info("📤 Deploying to Modal...")

        # Create Modal app file
        modal_app = f"""import modal
from pathlib import Path

stub = modal.Stub("transformer-summarizer-training")

# Create image with uv and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        # Install uv only if not already installed
        "if ! command -v uv &> /dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi",
        "source ~/.cargo/env",
        # Create venv only if not exists
        "if [ ! -d '.venv' ]; then uv venv; fi",
        "source .venv/bin/activate",
        "uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "uv add transformers datasets rouge-score tqdm wandb pyyaml numpy scikit-learn"
    )
)

@stub.function(
    image=image,
    gpu=modal.gpu.{config['gpu']}(),
    memory={config['memory']},
    cpu={config['cpu']},
    timeout={config['timeout']},
)
def train_model():
    import subprocess
    import sys
    import os
    
    # Activate uv environment
    os.system("source ~/.cargo/env")
    os.system("source .venv/bin/activate")
    
    # Clone repository
    subprocess.run(["git", "clone", "https://github.com/nanaagyei/transformer-summarizer.git"])
    
    # Change to project directory
    os.chdir("transformer-summarizer")
    
    # Set Python path
    sys.path.append(str(Path.cwd()))
    
    # Install project dependencies with uv
    if Path("pyproject.toml").exists():
        subprocess.run(["uv", "sync"])
    else:
        subprocess.run(["uv", "add", "transformers", "datasets", "rouge-score", "tqdm", "wandb", "pyyaml", "numpy", "scikit-learn"])
    
    # Run training
    subprocess.run([
        "python", "scripts/train_remote.py",
        "--config", "{self.config_path}"
    ])

@stub.local_entrypoint()
def main():
    train_model.remote()
"""

        # Save Modal app
        modal_file = self.project_root / "modal_app.py"
        with open(modal_file, 'w') as f:
            f.write(modal_app)

        logger.info(f"✅ Modal app saved to {modal_file}")
        logger.info("📋 Next steps:")
        logger.info("1. Install Modal: pip install modal")
        logger.info("2. Authenticate: modal token new")
        logger.info("3. Deploy: modal deploy modal_app.py")

    def deploy_to_jarvis(self, config: Dict[str, Any]):
        """Deploy training job to JarvisLabs"""
        logger.info("📤 Deploying to JarvisLabs...")

        # Create JarvisLabs configuration
        jarvis_config = {
            "instance_type": config["instance_type"],
            "gpu_type": config["gpu_type"],
            "framework": config["framework"],
            "python_version": config["python_version"],
            "startup_script": config["startup_script"],
        }

        # Save JarvisLabs config
        jarvis_file = self.project_root / "jarvis_config.json"
        with open(jarvis_file, 'w') as f:
            json.dump(jarvis_config, f, indent=2)

        logger.info(f"✅ JarvisLabs config saved to {jarvis_file}")
        logger.info("📋 Next steps:")
        logger.info("1. Go to https://jarvislabs.ai")
        logger.info("2. Create new instance with the saved configuration")
        logger.info("3. Upload your code or connect via Git")
        logger.info("4. Start training")

    def run(self):
        """Main execution method"""
        logger.info(f"🚀 Starting cloud training with {self.provider}")

        if self.provider == "vast":
            config = self.setup_vast_ai()
            self.deploy_to_vast(config)
        elif self.provider == "modal":
            config = self.setup_modal()
            self.deploy_to_modal(config)
        elif self.provider == "jarvis":
            config = self.setup_jarvis()
            self.deploy_to_jarvis(config)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        logger.info("✅ Cloud training setup complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Cloud Training for Transformer Summarizer")
    parser.add_argument("--provider", type=str, required=True,
                        choices=["vast", "modal", "jarvis"],
                        help="Cloud GPU provider")
    parser.add_argument("--config", type=str, default="configs/training_config_mps.yaml",
                        help="Training configuration file")

    args = parser.parse_args()

    # Create cloud trainer
    trainer = CloudTrainer(args.provider, args.config)
    trainer.run()


if __name__ == "__main__":
    main()
