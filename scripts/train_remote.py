#!/usr/bin/env python3
"""
Remote Training Script for Cloud GPU Instances

This script is designed to run on cloud GPU providers (Vast.ai, Modal, JarvisLabs)
and handles the complete training pipeline including data preparation,
model training, and result upload.
"""

from src.transformer_summarizer.utils.device_optimization import get_optimal_device
from src.transformer_summarizer.data.dataset import SummarizationDataset
from src.transformer_summarizer.training.trainer import TransformerTrainer
import argparse
import os
import sys
import yaml
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RemoteTrainer:
    """Handles remote training on cloud GPU instances"""

    def __init__(self, config_path: str, output_dir: str = "experiments/remote_training"):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.load_config(config_path)
        self.project_root = Path(__file__).parent.parent

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_environment(self):
        """Setup the training environment"""
        logger.info("🔧 Setting up training environment...")

        # Check GPU availability
        device_info = get_optimal_device()
        logger.info(f"🖥️ Using device: {device_info['device_name']}")

        # Create output directories
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        # Update config for remote training
        self.config['training']['output_dir'] = str(self.output_dir)
        self.config['logging']['log_file'] = str(
            self.output_dir / "logs" / "training.log")

        logger.info("✅ Environment setup complete")

    def prepare_data(self):
        """Prepare training data"""
        logger.info("📊 Preparing training data...")

        try:
            # This would typically involve downloading/preparing your dataset
            # For now, we'll create a dummy dataset for testing
            from src.transformer_summarizer.data.dataset import create_dummy_dataset

            train_dataset = create_dummy_dataset(
                size=self.config['data'].get('train_samples', 1000),
                max_input_length=self.config['data'].get(
                    'max_input_length', 256),
                max_target_length=self.config['data'].get(
                    'max_target_length', 64)
            )

            val_dataset = create_dummy_dataset(
                size=self.config['data'].get('val_samples', 100),
                max_input_length=self.config['data'].get(
                    'max_input_length', 256),
                max_target_length=self.config['data'].get(
                    'max_target_length', 64)
            )

            logger.info(
                f"✅ Created datasets: {len(train_dataset)} train, {len(val_dataset)} val")
            return train_dataset, val_dataset

        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            raise

    def train_model(self, train_dataset, val_dataset):
        """Train the model"""
        logger.info("🚀 Starting model training...")

        try:
            # Initialize trainer
            trainer = TransformerTrainer(self.config_path)

            # Create data loaders
            from torch.utils.data import DataLoader

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['data'].get('num_workers', 2),
                pin_memory=self.config['data'].get('pin_memory', False)
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=self.config['data'].get('num_workers', 2),
                pin_memory=self.config['data'].get('pin_memory', False)
            )

            # Create dummy tokenizer for now
            class DummyTokenizer:
                def batch_decode(self, tokens, skip_special_tokens=True):
                    return ["Sample summary"] * len(tokens)

            tokenizer = DummyTokenizer()

            # Start training
            trainer.train(train_loader, val_loader, tokenizer)

            logger.info("✅ Training completed successfully!")

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

    def save_results(self):
        """Save training results and artifacts"""
        logger.info("💾 Saving training results...")

        # Save final model
        model_path = self.output_dir / "models" / "final_model.pth"
        # trainer.save_checkpoint(str(model_path))

        # Save training config
        config_path = self.output_dir / "training_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Create results summary
        results = {
            "training_completed": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_used": self.config_path,
            "output_dir": str(self.output_dir),
            "model_path": str(model_path),
        }

        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✅ Results saved to {self.output_dir}")

    def upload_results(self):
        """Upload results to cloud storage (optional)"""
        logger.info("☁️ Uploading results to cloud storage...")

        # This would typically upload to S3, Google Cloud Storage, etc.
        # For now, we'll just log the results location
        logger.info(f"📁 Results available at: {self.output_dir}")
        logger.info("💡 To upload to cloud storage, implement upload logic here")

    def run(self):
        """Main execution method"""
        logger.info("🚀 Starting remote training pipeline...")

        try:
            # Setup environment
            self.setup_environment()

            # Prepare data
            train_dataset, val_dataset = self.prepare_data()

            # Train model
            self.train_model(train_dataset, val_dataset)

            # Save results
            self.save_results()

            # Upload results (optional)
            self.upload_results()

            logger.info("🎉 Remote training pipeline completed successfully!")

        except Exception as e:
            logger.error(f"❌ Remote training failed: {e}")
            # Save error information
            error_info = {
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config_used": self.config_path,
            }

            error_path = self.output_dir / "error.json"
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2)

            raise


def main():
    parser = argparse.ArgumentParser(
        description="Remote Training for Transformer Summarizer")
    parser.add_argument("--config", type=str, required=True,
                        help="Training configuration file")
    parser.add_argument("--output-dir", type=str, default="experiments/remote_training",
                        help="Output directory for results")

    args = parser.parse_args()

    # Create remote trainer
    trainer = RemoteTrainer(args.config, args.output_dir)
    trainer.run()


if __name__ == "__main__":
    main()
