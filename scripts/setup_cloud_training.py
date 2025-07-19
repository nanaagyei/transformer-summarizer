#!/usr/bin/env python3
"""
Cloud Training Setup Script

This script helps you set up cloud training for your transformer summarizer project.
It guides you through the setup process for different cloud GPU providers.
"""

import argparse
import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any


def setup_vast_ai():
    """Setup Vast.ai for cloud training based on official documentation"""
    print("🚀 Setting up Vast.ai cloud training...")
    print("\n📋 Steps to get started with Vast.ai (based on official PyTorch docs):")
    print("")
    print("1️⃣ Sign up and get API key:")
    print("   • Go to https://vast.ai/console/account")
    print("   • Create account and add credits")
    print("   • Copy your API key")
    print("")
    print("2️⃣ Install Vast.ai CLI:")
    print("   pip install vast-ai")
    print("")
    print("3️⃣ Set your API key:")
    print("   export VAST_API_KEY=your_api_key_here")
    print("")
    print("4️⃣ Generate SSH key (if you don't have one):")
    print("   ssh-keygen -t rsa -b 4096 -C 'your_email@example.com'")
    print("")
    print("5️⃣ Search for PyTorch-compatible instances:")
    print("   vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100'")
    print("")
    print("6️⃣ Run the cloud training script:")
    print("   python scripts/cloud_train.py --provider vast --config configs/training_config_cloud.yaml")
    print("")
    print("🔗 Official Vast.ai PyTorch Documentation: https://docs.vast.ai/pytorch")
    print("🔗 Vast.ai Console: https://vast.ai/console/")

    # Create Vast.ai specific configuration based on official docs
    vast_config = {
        "provider": "vast",
        "gpu_types": ["RTX_4090", "RTX_3090", "A100", "H100"],
        "price_range": [0.20, 3.00],  # $/hr
        "estimated_cost": {
            "RTX_4090": "$0.30-0.50/hr",
            "RTX_3090": "$0.25-0.40/hr",
            "A100": "$1.00-1.50/hr",
            "H100": "$2.00-3.00/hr"
        },
        "recommended": "RTX_4090 for cost-effectiveness",
        "official_docs": "https://docs.vast.ai/pytorch",
        "best_practices": [
            "Use PyTorch 2.1.0 with CUDA 12.1",
            "Use uv for fast Python package management",
            "Minimum 8GB GPU memory",
            "Minimum 50GB disk space",
            "Minimum 100Mbps internet speed",
            "Use spot instances for cost savings",
            "Monitor GPU utilization with nvidia-smi",
            "Set up proper error handling and logging"
        ],
        "pytorch_specific": {
            "image": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            "install_method": "uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "cuda_version": "12.1",
            "min_gpu_memory": "8GB",
            "package_manager": "uv"
        }
    }

    config_file = Path("vast_setup.json")
    with open(config_file, 'w') as f:
        json.dump(vast_config, f, indent=2)

    print(f"\n✅ Vast.ai configuration saved to {config_file}")
    print("\n💡 Key advantages of Vast.ai for PyTorch training:")
    print("   • Official PyTorch support and documentation")
    print("   • Pre-configured PyTorch Docker images")
    print("   • Fast package management with uv")
    print("   • Cost-effective GPU pricing")
    print("   • Spot instances for additional savings")
    print("   • Easy SSH access for debugging")
    print("   • Comprehensive monitoring tools")

    return vast_config


def setup_modal():
    """Setup Modal for cloud training"""
    print("🚀 Setting up Modal cloud training...")
    print("\n📋 Steps to get started with Modal:")
    print("1. Sign up at https://modal.com")
    print("2. Install Modal CLI:")
    print("   pip install modal")
    print("3. Authenticate:")
    print("   modal token new")
    print("4. Run the cloud training script:")
    print("   python scripts/cloud_train.py --provider modal --config configs/training_config_cloud.yaml")

    # Create Modal specific configuration
    modal_config = {
        "provider": "modal",
        "gpu_types": ["T4", "A100", "H100"],
        "pricing": {
            "T4": "$0.40/hr",
            "A100": "$2.00/hr",
            "H100": "$4.00/hr"
        },
        "advantages": [
            "Serverless - no instance management",
            "Pay only for actual compute time",
            "Easy deployment",
            "Automatic scaling"
        ],
        "recommended": "T4 for development, A100 for production"
    }

    config_file = Path("modal_setup.json")
    with open(config_file, 'w') as f:
        json.dump(modal_config, f, indent=2)

    print(f"\n✅ Modal configuration saved to {config_file}")
    return modal_config


def setup_jarvis():
    """Setup JarvisLabs for cloud training"""
    print("🚀 Setting up JarvisLabs cloud training...")
    print("\n📋 Steps to get started with JarvisLabs:")
    print("1. Sign up at https://jarvislabs.ai")
    print("2. Create a new project")
    print("3. Choose your GPU instance type")
    print("4. Upload your code or connect via Git")
    print("5. Run the cloud training script:")
    print("   python scripts/cloud_train.py --provider jarvis --config configs/training_config_cloud.yaml")

    # Create JarvisLabs specific configuration
    jarvis_config = {
        "provider": "jarvis",
        "instance_types": ["gpu.1x", "gpu.2x", "gpu.4x", "gpu.8x"],
        "gpu_options": ["RTX_4090", "A100", "H100"],
        "pricing": {
            "gpu.1x": "$0.50-1.00/hr",
            "gpu.2x": "$1.00-2.00/hr",
            "gpu.4x": "$2.00-4.00/hr",
            "gpu.8x": "$4.00-8.00/hr"
        },
        "advantages": [
            "Specialized for ML/AI workloads",
            "Easy-to-use interface",
            "Integrated Jupyter notebooks",
            "Team collaboration features"
        ],
        "recommended": "gpu.1x with RTX_4090 for cost-effectiveness"
    }

    config_file = Path("jarvis_setup.json")
    with open(config_file, 'w') as f:
        json.dump(jarvis_config, f, indent=2)

    print(f"\n✅ JarvisLabs configuration saved to {config_file}")
    return jarvis_config


def compare_providers():
    """Compare different cloud providers"""
    print("📊 Cloud Provider Comparison")
    print("=" * 50)

    providers = {
        "Vast.ai": {
            "cost": "Lowest ($0.20-3.00/hr)",
            "ease_of_use": "Medium (CLI required)",
            "features": "Pay-per-use, spot instances",
            "best_for": "Cost-conscious users, batch training"
        },
        "Modal": {
            "cost": "Medium ($0.40-4.00/hr)",
            "ease_of_use": "High (serverless)",
            "features": "Serverless, auto-scaling, easy deployment",
            "best_for": "Quick experiments, serverless workflows"
        },
        "JarvisLabs": {
            "cost": "Medium-High ($0.50-8.00/hr)",
            "ease_of_use": "High (web interface)",
            "features": "ML-optimized, Jupyter notebooks, team features",
            "best_for": "ML teams, interactive development"
        }
    }

    for provider, info in providers.items():
        print(f"\n{provider}:")
        print(f"  💰 Cost: {info['cost']}")
        print(f"  🎯 Ease of use: {info['ease_of_use']}")
        print(f"  ⚡ Features: {info['features']}")
        print(f"  🎯 Best for: {info['best_for']}")


def create_cost_estimator():
    """Create a cost estimation tool"""
    print("💰 Creating cost estimation tool...")

    cost_estimator = {
        "training_hours": {
            "small_experiment": 2,
            "medium_training": 8,
            "large_training": 24,
            "production_training": 72
        },
        "provider_costs": {
            "vast_rtx4090": 0.35,
            "vast_a100": 1.25,
            "modal_t4": 0.40,
            "modal_a100": 2.00,
            "jarvis_rtx4090": 0.75,
            "jarvis_a100": 2.50
        },
        "estimates": {
            "small_experiment": {
                "vast_rtx4090": "$0.70",
                "modal_t4": "$0.80",
                "jarvis_rtx4090": "$1.50"
            },
            "medium_training": {
                "vast_rtx4090": "$2.80",
                "modal_t4": "$3.20",
                "jarvis_rtx4090": "$6.00"
            },
            "large_training": {
                "vast_rtx4090": "$8.40",
                "modal_t4": "$9.60",
                "jarvis_rtx4090": "$18.00"
            }
        }
    }

    config_file = Path("cost_estimator.json")
    with open(config_file, 'w') as f:
        json.dump(cost_estimator, f, indent=2)

    print(f"✅ Cost estimator saved to {config_file}")
    print("\n💡 Cost estimates for different training scenarios:")
    for scenario, costs in cost_estimator["estimates"].items():
        print(f"\n{scenario.replace('_', ' ').title()}:")
        for provider, cost in costs.items():
            print(f"  {provider}: {cost}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup Cloud Training for Transformer Summarizer")
    parser.add_argument("--provider", type=str, choices=["vast", "modal", "jarvis", "compare", "costs"],
                        help="Cloud provider to setup or action to perform")
    parser.add_argument("--all", action="store_true",
                        help="Setup all providers")

    args = parser.parse_args()

    print("☁️ Cloud Training Setup for Transformer Summarizer")
    print("=" * 60)

    if args.all:
        print("Setting up all cloud providers...")
        setup_vast_ai()
        print("\n" + "="*60 + "\n")
        setup_modal()
        print("\n" + "="*60 + "\n")
        setup_jarvis()
        print("\n" + "="*60 + "\n")
        compare_providers()
        print("\n" + "="*60 + "\n")
        create_cost_estimator()
    elif args.provider == "vast":
        setup_vast_ai()
    elif args.provider == "modal":
        setup_modal()
    elif args.provider == "jarvis":
        setup_jarvis()
    elif args.provider == "compare":
        compare_providers()
    elif args.provider == "costs":
        create_cost_estimator()
    else:
        print("Please specify a provider or use --all to setup everything")
        print("\nAvailable options:")
        print("  --provider vast    - Setup Vast.ai")
        print("  --provider modal   - Setup Modal")
        print("  --provider jarvis  - Setup JarvisLabs")
        print("  --provider compare - Compare all providers")
        print("  --provider costs   - Create cost estimator")
        print("  --all              - Setup all providers")


if __name__ == "__main__":
    main()
