#!/usr/bin/env python3
"""MPS-optimized training script for Apple Silicon"""
import sys
import os
import argparse
from pathlib import Path
import torch
import time

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    parser = argparse.ArgumentParser(description='Train Transformer with MPS acceleration')
    parser.add_argument('--config', type=str, default='configs/training_config_mps.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--model-config', type=str, default='configs/model_config_mps.yaml',
                        help='Path to model configuration file')
    parser.add_argument('--phase', type=str, choices=['proof_of_concept', 'small_scale', 'production'],
                        default='small_scale', help='Training phase')
    parser.add_argument('--device', type=str, choices=['auto', 'mps', 'cpu'],
                        default='auto', help='Device to use for training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    print("🍎 Apple Silicon Transformer Training")
    print("=" * 50)
    
    # Import device optimization utilities
    try:
        from src.transformer_summarizer.utils.device_optimization import (
            setup_optimal_training_environment, get_optimal_device
        )
    except ImportError:
        print("⚠️ Could not import device optimizations - using basic setup")
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        optimization_settings = {'batch_size_multiplier': 1.0}
    else:
        # Setup optimal training environment
        if args.device == 'auto':
            device, optimization_settings = setup_optimal_training_environment()
        else:
            device = torch.device(args.device)
            from src.transformer_summarizer.utils.device_optimization import optimize_for_device
            optimization_settings = optimize_for_device(device)
    
    print(f"\n🎯 Training Phase: {args.phase}")
    print(f"📋 Config: {args.config}")
    print(f"🖥️ Device: {device}")
    
    # Check if config files exist
    config_path = Path(args.config)
    model_config_path = Path(args.model_config)
    
    if not config_path.exists():
        print(f"❌ Training config not found: {config_path}")
        print("Please create the MPS configuration files first.")
        return 1
        
    if not model_config_path.exists():
        print(f"❌ Model config not found: {model_config_path}")
        print("Please create the MPS configuration files first.")
        return 1
    
    # Test MPS functionality
    if device.type == 'mps':
        print("\n🧪 Testing MPS functionality...")
        try:
            # Test basic tensor operations
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)
            print(f"✅ MPS tensor operations working")
            
            # Test gradient computation
            x.requires_grad_(True)
            loss = (z ** 2).sum()
            loss.backward()
            print(f"✅ MPS gradient computation working")
            
            # Clear cache
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
        except Exception as e:
            print(f"❌ MPS test failed: {e}")
            print("Falling back to CPU...")
            device = torch.device('cpu')
    
    # Import training components (placeholder for now)
    try:
        # from src.transformer_summarizer.training.trainer import TransformerTrainer
        # from src.transformer_summarizer.data.dataset import create_dataloaders
        
        print(f"\n🚀 Training would start here with the following setup:")
        print(f"   • Device: {device}")
        print(f"   • Config: {config_path}")
        print(f"   • Model Config: {model_config_path}")
        print(f"   • Phase: {args.phase}")
        
        if device.type == 'mps':
            print(f"\n🍎 MPS-specific optimizations:")
            print(f"   • Unified memory architecture leveraged")
            print(f"   • Fallback to CPU for unsupported operations")
            print(f"   • Larger batch sizes possible: {optimization_settings.get('batch_size_multiplier', 1.0)}x")
            print(f"   • Expected speedup: 2-3x over CPU")
            
            # Estimate training time
            if args.phase == 'proof_of_concept':
                estimated_time = "30-60 minutes"
            elif args.phase == 'small_scale':
                estimated_time = "2-4 hours"
            else:
                estimated_time = "8-12 hours"
                
            print(f"   • Estimated training time: {estimated_time}")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Implement transformer model architecture")
        print(f"   2. Create training pipeline")
        print(f"   3. Add data loading utilities")
        print(f"   4. Start training with: uv run python scripts/train_mps.py --phase proof_of_concept")
        
        return 0
        
    except ImportError as e:
        print(f"⚠️ Training modules not implemented yet: {e}")
        print("This is expected during initial setup.")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)