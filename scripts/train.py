#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    parser = argparse.ArgumentParser(description='Train Transformer for Text Summarization')
    parser.add_argument('--config', type=str, default='configs/training_config_mps.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--model-config', type=str, default='configs/model_config_mps.yaml',
                        help='Path to model configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with dummy data')
    parser.add_argument('--phase', type=str, 
                        choices=['proof_of_concept', 'small_scale', 'production'],
                        default='small_scale',
                        help='Training phase (affects model size and data amount)')
    
    args = parser.parse_args()
    
    print("🚀 Transformer Summarization Training")
    print("=" * 50)
    print(f"📋 Training config: {args.config}")
    print(f"🏗️ Model config: {args.model_config}")
    print(f"🎯 Phase: {args.phase}")
    
    # Import training components
    try:
        from src.transformer_summarizer.training.trainer import TransformerTrainer, test_training_setup, run_training_test
        from src.transformer_summarizer.data.dataset import create_dataloaders
        from src.transformer_summarizer.utils.device_optimization import setup_optimal_training_environment
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all modules are properly implemented!")
        return 1
    
    # Setup optimal training environment
    device, optimization_settings = setup_optimal_training_environment()
    
    if args.test:
        print("\n🧪 Running in test mode...")
        print("=" * 30)
        
        # Test training setup
        success = test_training_setup()
        if not success:
            print("❌ Training setup test failed!")
            return 1
        
        # Run training test
        success = run_training_test()
        if not success:
            print("❌ Training test failed!")
            return 1
        
        print("\n🎉 All tests passed! Ready for real training.")
        print("\n💡 To start real training, run:")
        print(f"    uv run python {__file__} --phase proof_of_concept")
        return 0
    
    # Check if config files exist
    if not Path(args.config).exists():
        print(f"❌ Training config not found: {args.config}")
        print("Please create the configuration files first!")
        return 1
    
    if not Path(args.model_config).exists():
        print(f"❌ Model config not found: {args.model_config}")
        print("Please create the model configuration files first!")
        return 1
    
    try:
        print("\n🔧 Initializing trainer...")
        
        # Initialize trainer
        trainer = TransformerTrainer(args.config)
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"📂 Resuming from checkpoint: {args.resume}")
            success = trainer.load_checkpoint(args.resume)
            if not success:
                print("❌ Failed to load checkpoint!")
                return 1
        
        print("\n📊 Loading dataset...")
        
        # Load data
        train_loader, val_loader, tokenizer = create_dataloaders(args.config)
        
        print(f"✅ Data loaded successfully!")
        print(f"   📈 Training batches: {len(train_loader)}")
        print(f"   📊 Validation batches: {len(val_loader)}")
        print(f"   🔤 Vocabulary size: {len(tokenizer)}")
        print(f"   📦 Batch size: {train_loader.batch_size}")
        
        # Calculate training estimates
        total_steps = len(train_loader) * trainer.config['training']['num_epochs']
        accumulation_steps = trainer.config['training'].get('gradient_accumulation_steps', 1)
        effective_steps = total_steps // accumulation_steps
        
        print(f"\n⏱️ Training estimates:")
        print(f"   📊 Total steps: {total_steps:,}")
        print(f"   🔄 Effective steps: {effective_steps:,}")
        print(f"   🎯 Target ROUGE-L: >0.30")
        
        if device.type == 'mps':
            print(f"   🍎 Expected time on Apple Silicon: 2-8 hours")
        else:
            print(f"   🖥️ Expected time on CPU: 8-24 hours")
        
        print(f"\n🚀 Starting training...")
        print("=" * 50)
        
        # Start training!
        trainer.train(train_loader, val_loader, tokenizer)
        
        print("\n🎉 Training completed successfully!")
        print(f"🏆 Best ROUGE-L achieved: {trainer.best_rouge:.4f}")
        print(f"💾 Model saved to: experiments/models/best_model.pth")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        print("💾 Latest checkpoint should be saved in experiments/models/")
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)