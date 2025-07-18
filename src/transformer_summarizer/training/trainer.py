import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
import torch.nn.functional as F
import wandb
import yaml
import os
import time
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import math
import numpy as np


from ..models.transformer import Transformer
from ..utils.device_optimization import get_optimal_device


class TransformerTrainer:

    def __init__(self, config_path: str = 'configs/training_config_mps.yaml'):
        """
        Initializes the TransformerTrainer instance with the given configuration path.

        Args:
            config_path (str): The path to the training configuration YAML file. Defaults to 'configs/training_config_mps.yaml'.

        Returns:
            None
        """
        # Load configuration
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_device()

        # Initialize model
        model_config_path = self.config.get(
            'model_config', 'configs/model_config_mps.yaml')
        self.model = Transformer(model_config_path)
        self.model.to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.model.pad_token_id)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_rouge = 0.0
        self.best_loss = float('inf')

        # Setup experiment tracking
        self.setup_wandb()

        print(f"🚀 Trainer initialized!")
        print(f"📋 Device: {self.device}")
        print(
            f"🔧 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(
            f"💾 Model size: {sum(p.numel() * 4 for p in self.model.parameters()) / 1024**2:.1f} MB")

    def load_config(self, config_path: str) -> Dict:
        """
        Loads the training configuration from the specified YAML file.

        Args:
            config_path (str): The path to the training configuration YAML file.

        Returns:
            Dict: The loaded configuration dictionary.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def setup_logging(self):
        """
        Sets up the logging configuration for the trainer.

        Creates the necessary directories for logging, configures the basic logging settings,
        and sets up the logger instance for the trainer.

        Returns:
            None
        """
        os.makedirs('experiments/logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('experiments/logs/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_device(self):
        """
        Sets up the optimal device for training and applies device-specific optimizations.

        Retrieves the optimal device and its information using the get_optimal_device function.
        If the device is of type 'mps', enables MPS optimizations by setting the 
        PYTORCH_ENABLE_MPS_FALLBACK environment variable.

        Prints the device name for training.

        Returns:
            None
        """
        self.device, self.device_info = get_optimal_device()

        # Apply device-specific optimizations
        if self.device.type == 'mps':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            print("🍎 MPS optimizations enabled!")

        print(f"🖥️ Training on: {self.device_info['device_name']}")

    def setup_optimizer(self):
        """
        Sets up the optimizer for the model.

        Args:
            None

        Returns:
            torch.optim.Optimizer: The configured AdamW optimizer
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=self.config['optimizer']['betas'],
            eps=float(self.config['optimizer']['eps']),
            weight_decay=self.config['training']['weight_decay'],
        )

        return optimizer

    def setup_scheduler(self):
        """
        Sets up the scheduler for the optimizer.

        Args:
            None

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The configured learning rate scheduler
        """
        # TODO: Calculate actual steps based on dataset size and batch size
        # For now, use a reasonable estimate
        total_steps = self.config['training']['num_epochs'] * 1000  # Estimate
        warmup_steps = int(
            total_steps * self.config['scheduler']['warmup_ratio'])

        scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        return scheduler

    def setup_wandb(self):
        """
        Sets up the Weights & Biases experiment tracking.

        Args:
            None

        Returns:
            None
        """
        if not self.config['logging'].get('use_wandb', True):
            return

        try:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                config=self.config,
                name=f"transformer-{self.device.type}-{wandb.util.generate_id()}",
                tags=self.config['logging'].get('wandb_tags', [])
            )
            wandb.watch(self.model, log='all', log_freq=100)
            self.use_wandb = True
            print("📊 Weights & Biases tracking enabled!")
        except Exception as e:
            print(f"⚠️ W&B setup failed: {e}. Continuing without tracking.")
            self.use_wandb = False

    def train_epoch(self, train_loader, val_loader, tokenizer) -> float:
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.
            tokenizer (Tokenizer): The tokenizer for text processing.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        # Setup progress bar
        progress_bar = tqdm(
            train_loader,
            desc=f'Epoch {self.current_epoch}',
            leave=True
        )

        # Gradient accumulation setup
        accumulation_steps = self.config['training'].get(
            'gradient_accumulation_steps', 1)
        effective_batch_size = train_loader.batch_size * accumulation_steps

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            decoder_input_ids = batch['decoder_input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(input_ids, decoder_input_ids)

            # Calculate loss
            loss = self.criterion(
                # (batch_size * seq_len, vocab_size)
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1)  # (batch_size * seq_len)
            )

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Clear MPS cache periodically
                if self.device.type == 'mps' and self.global_step % 50 == 0:
                    torch.mps.empty_cache()

            # Update metrics
            epoch_loss += loss.item() * accumulation_steps  # Unscale for logging
            self.global_step += 1

            # Logging
            if self.global_step % self.config['logging']['log_every'] == 0:
                self.log_metrics({
                    'train/loss': loss.item() * accumulation_steps,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/global_step': self.global_step,
                    'train/epoch': self.current_epoch
                })

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'step': self.global_step
            })

            # Validation and checkpointing
            if self.global_step % self.config['training']['eval_every'] == 0:
                val_metrics = self.validate(val_loader, tokenizer)
                self.log_metrics(val_metrics)

                # Save best model
                if val_metrics.get('val/rouge_l', 0) > self.best_rouge:
                    self.best_rouge = val_metrics['val/rouge_l']
                    self.save_checkpoint('best_model.pth', is_best=True)
                    print(f"🏆 New best ROUGE-L: {self.best_rouge:.4f}")

                # Back to training mode
                self.model.train()

            # Regular checkpointing
            if self.global_step % self.config['training']['save_every'] == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pth')

        return epoch_loss / num_batches

    def validate(self, val_loader, tokenizer) -> Dict[str, float]:
        """
        Runs validation and calculates metrics.

        Args:
            val_loader (DataLoader): The validation data loader.
            tokenizer (Tokenizer): The tokenizer for text processing.

        Returns:
            Dict[str, float]: A dictionary containing validation metrics.
        """
        self.model.eval()
        val_loss = 0.0
        # Limit validation batches for speed
        num_batches = min(len(val_loader), 50)

        predictions = []
        references = []

        print(f"\n🔍 Running validation on {num_batches} batches...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= num_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Calculate validation loss
                outputs = self.model(input_ids, decoder_input_ids)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
                val_loss += loss.item()

                # Generate summaries for a subset of the batch
                if batch_idx < 10:  # Only generate for first 10 batches
                    generated = self.model.greedy_generate(
                        input_ids, max_length=64)

                    # Decode predictions and references
                    batch_predictions = tokenizer.batch_decode(
                        generated, skip_special_tokens=True)
                    batch_references = tokenizer.batch_decode(
                        labels, skip_special_tokens=True)

                    predictions.extend(batch_predictions)
                    references.extend(batch_references)

        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge_scores(predictions, references)

        val_metrics = {
            'val/loss': val_loss / num_batches,
            'val/rouge_1': rouge_scores.get('rouge1', 0.0),
            'val/rouge_2': rouge_scores.get('rouge2', 0.0),
            'val/rouge_l': rouge_scores.get('rougeL', 0.0),
            'val/num_examples': len(predictions)
        }

        # Show example predictions
        if predictions:
            print(f"\n📝 Example prediction:")
            print(f"Reference: {references[0]}")
            print(f"Predicted: {predictions[0]}")

        return val_metrics

    def calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculates ROUGE scores.

        Args:
            predictions (List[str]): The predicted summaries.
            references (List[str]): The reference summaries.

        Returns:
            Dict[str, float]: A dictionary containing ROUGE scores.
        """
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []

            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)

            return {
                'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
                'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
                'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
            }
        except ImportError:
            print("⚠️ rouge-score not available, skipping ROUGE calculation")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Saves the model checkpoint.

        Args:
            filename (str): The name of the checkpoint file.
            is_best (bool): Whether the checkpoint is the best model.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_rouge': self.best_rouge,
            'best_loss': self.best_loss,
            'config': self.config,
            'device_info': self.device_info
        }

        # Create checkpoints directory
        os.makedirs('experiments/models', exist_ok=True)
        filepath = f'experiments/models/{filename}'

        # Save checkpoint
        torch.save(checkpoint, filepath)

        if is_best:
            # Also save as best model
            best_path = 'experiments/models/best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"✅ Best model saved: {best_path}")

        self.logger.info(f"✅ Checkpoint saved: {filepath}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Loads the model checkpoint.

        Args:
            checkpoint_path (str): The path to the checkpoint file.

        Returns:
            bool: True if the checkpoint was loaded successfully, False otherwise.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_rouge = checkpoint.get('best_rouge', 0.0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))

            self.logger.info(f"✅ Checkpoint loaded: {checkpoint_path}")
            self.logger.info(
                f"📊 Resuming from epoch {self.current_epoch}, step {self.global_step}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {e}")
            return False

    def log_metrics(self, metrics: Dict[str, float]):
        """
        Logs metrics to W&B and console.

        Args:
            metrics (Dict[str, float]): A dictionary containing the metrics to log.
        """
        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)

        # Also log to our logger occasionally
        if self.global_step % (self.config['logging']['log_every'] * 10) == 0:
            metric_str = ', '.join(
                [f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Step {self.global_step} - {metric_str}")

    def train(self, train_loader, val_loader, tokenizer):
        """
        Main training function

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            tokenizer: Tokenizer for text processing
        """
        self.logger.info("🚀 Starting training...")
        self.logger.info(
            f"📊 Training for {self.config['training']['num_epochs']} epochs")
        self.logger.info(f"📊 {len(train_loader)} batches per epoch")
        self.logger.info(
            f"📊 Effective batch size: {self.config['training']['batch_size'] * self.config['training'].get('gradient_accumulation_steps', 1)}")
        start_time = time.time()

        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss = self.train_epoch(train_loader, val_loader, tokenizer)

            # Log epoch metrics
            epoch_metrics = {
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'train/epoch_time': time.time() - start_time
            }
            self.log_metrics(epoch_metrics)

            # Run final validation for the epoch
            val_metrics = self.validate(val_loader, tokenizer)
            self.log_metrics(val_metrics)

            self.logger.info(f"📊 Epoch {epoch} completed:")
            self.logger.info(f"   Train Loss: {train_loss:.4f}")
            self.logger.info(f"   Val Loss: {val_metrics['val/loss']:.4f}")
            self.logger.info(f"   ROUGE-L: {val_metrics['val/rouge_l']:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch}.pth')

            # Early stopping check (optional)
            if self.early_stopping_check(val_metrics):
                self.logger.info("🛑 Early stopping triggered")
                break

        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"🎉 Training completed!")
        self.logger.info(f"⏱️ Total time: {total_time/3600:.2f} hours")
        self.logger.info(f"🏆 Best ROUGE-L: {self.best_rouge:.4f}")

        # Save final model
        self.save_checkpoint('final_model.pth')

        if self.use_wandb:
            wandb.finish()

    def early_stopping_check(self, val_metrics: Dict[str, float]) -> bool:
        """
        Checks if early stopping should be triggered.

        Args:
            val_metrics (Dict[str, float]): A dictionary containing the validation metrics.

        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        # Simple early stopping based on validation loss
        val_loss = val_metrics.get('val/loss', float('inf'))
        patience = self.config['training'].get('early_stopping_patience', 5)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter = getattr(self, 'patience_counter', 0) + 1

        return self.patience_counter >= patience


def test_training_setup():
    """
    Tests the training setup with a small model and data.

    Args:
        None

    Returns:
        bool: True if the training setup is successful, False otherwise.
    """
    print("🧪 Testing Training Setup...")

    # Create a minimal config for testing
    test_config = {
        'training': {
            'learning_rate': 0.001,
            'num_epochs': 1,
            'gradient_accumulation_steps': 1,
            'gradient_clip_norm': 1.0,
            'weight_decay': 0.01,
            'save_every': 100,
            'eval_every': 50,
            'early_stopping_patience': 3
        },
        'optimizer': {
            'betas': [0.9, 0.999],
            'eps': 1e-8
        },
        'scheduler': {
            'warmup_ratio': 0.1
        },
        'logging': {
            'wandb_project': 'test-transformer',
            'use_wandb': False,  # Disable for testing
            'log_every': 10,
            'wandb_tags': ['test']
        }
    }

    # Save test config
    os.makedirs('configs', exist_ok=True)
    with open('configs/test_training_config.yaml', 'w') as f:
        yaml.dump(test_config, f)

    try:
        # Test trainer initialization
        print("1️⃣ Testing trainer initialization...")
        trainer = TransformerTrainer('configs/test_training_config.yaml')
        print("✅ Trainer initialized successfully")

        # Test optimizer setup
        print("2️⃣ Testing optimizer...")
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        print("✅ Optimizer and scheduler working")

        # Test checkpoint saving/loading
        print("3️⃣ Testing checkpointing...")
        trainer.save_checkpoint('test_checkpoint.pth')
        success = trainer.load_checkpoint(
            'experiments/models/test_checkpoint.pth')
        assert success, "Checkpoint loading failed"
        print("✅ Checkpointing working")

        # Test metrics calculation
        print("4️⃣ Testing metrics...")
        test_predictions = ["The cat sat on the mat", "Dogs are pets"]
        test_references = ["Cat on mat", "Dogs are animals"]
        rouge_scores = trainer.calculate_rouge_scores(
            test_predictions, test_references)
        assert 'rouge1' in rouge_scores
        print("✅ ROUGE calculation working")

        print("🎉 All training setup tests passed!")
        return True

    except Exception as e:
        print(f"❌ Training setup test failed: {e}")
        return False


def create_dummy_data_for_testing():
    """
    Creates dummy data for testing the training loop.

    Args:
        None

    Returns:
        DataLoader: A data loader for the dummy data.
    """
    from torch.utils.data import Dataset, DataLoader

    class DummyDataset(Dataset):
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Create dummy data that looks like real training data
            return {
                'input_ids': torch.randint(3, 1000, (128,)),  # Random tokens
                'attention_mask': torch.ones(128),
                'decoder_input_ids': torch.randint(3, 1000, (64,)),
                'labels': torch.randint(3, 1000, (64,))
            }

    dataset = DummyDataset(50)  # Small dataset for testing
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    return dataloader


def run_training_test():
    """
    Runs a complete but short training test.

    Args:
        None

    Returns:
        bool: True if the training test is successful, False otherwise.
    """
    print("🏋️ Running Training Test...")

    try:
        # Create test trainer
        trainer = TransformerTrainer('configs/test_training_config.yaml')

        # Create dummy data
        train_loader = create_dummy_data_for_testing()
        val_loader = create_dummy_data_for_testing()

        # Create dummy tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.pad_token = '[PAD]'

        # Run one training step
        print("🔄 Running training steps...")
        trainer.model.train()

        batch = next(iter(train_loader))
        input_ids = batch['input_ids'].to(trainer.device)
        decoder_input_ids = batch['decoder_input_ids'].to(trainer.device)
        labels = batch['labels'].to(trainer.device)

        # Forward pass
        outputs = trainer.model(input_ids, decoder_input_ids)
        loss = trainer.criterion(
            outputs.view(-1, outputs.size(-1)), labels.view(-1))

        # Backward pass
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

        print(f"✅ Training step completed! Loss: {loss.item():.4f}")

        # Test validation
        print("🔍 Testing validation...")
        val_metrics = trainer.validate(val_loader, tokenizer)
        print(f"✅ Validation completed! Metrics: {val_metrics}")

        print("🎉 Training test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main_training_script():
    """
    Main script to start training with real data.

    Args:
        None

    Returns:
        None
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Transformer for Summarization')
    parser.add_argument('--config', type=str, default='configs/training_config_mps.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with dummy data')

    args = parser.parse_args()

    if args.test:
        print("🧪 Running in test mode...")
        success = test_training_setup()
        if success:
            run_training_test()
        return

    print("🚀 Starting full training...")

    # Initialize trainer
    trainer = TransformerTrainer(args.config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Load real data
    try:
        from ..data.dataset import create_dataloaders
        train_loader, val_loader, tokenizer = create_dataloaders(args.config)

        print(f"📊 Data loaded:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Vocabulary size: {len(tokenizer)}")

        # Start training
        trainer.train(train_loader, val_loader, tokenizer)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
