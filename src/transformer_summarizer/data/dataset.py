import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional



class SummarizationDataset(Dataset):
    
    def __init__(
        self,
        data,
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 128,
        pad_token_id: int = 0,
        bos_token_id: int = 101,
        eos_token_id: int = 102
    ):
        """
        Initialize the SummarizationDataset.

        Args:
            data: List or iterable containing the dataset examples.
            tokenizer: Tokenizer used for encoding the input and target texts.
            max_input_length (int, optional): Maximum length for input sequences. Defaults to 512.
            max_target_length (int, optional): Maximum length for target sequences. Defaults to 128.
            pad_token_id (int, optional): Token ID used for padding. Defaults to 0.
            bos_token_id (int, optional): Token ID used for the beginning of sequence token. Defaults to 101.
            eos_token_id (int, optional): Token ID used for the end of sequence token. Defaults to 102.

        Prints:
            Dataset creation confirmation with number of examples.
            Maximum input and target sequence lengths.
        """

        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        print(f"📊 Dataset created with {len(self.data)} examples")
        print(f"📏 Max input length: {max_input_length}")
        print(f"📏 Max target length: {max_target_length}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single training example

        Returns:
            Dict with 'input_ids', 'attention_mask', 'labels', 'decoder_input_ids'
        """
        item = self.data[idx]

        # Get article and summary text
        article = item['article']
        summary = item['highlights']

        # Tokenize article (source)
        article_encoding = self.tokenizer(
            article,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize summary (target) - we need to add BOS and EOS tokens
        # First, tokenize without special tokens to get the content
        summary_tokens = self.tokenizer(
            summary,
            max_length=self.max_target_length - 2,  # Leave room for BOS/EOS
            padding=False,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].squeeze()

        # Create decoder input (BOS + summary tokens) and labels (summary tokens + EOS)
        decoder_input_ids = torch.cat([
            torch.tensor([self.bos_token_id]),
            summary_tokens,
            torch.tensor([self.pad_token_id] *
                         (self.max_target_length - len(summary_tokens) - 1))
        ])[:self.max_target_length]

        labels = torch.cat([
            summary_tokens,
            torch.tensor([self.eos_token_id]),
            torch.tensor([self.pad_token_id] *
                         (self.max_target_length - len(summary_tokens) - 1))
        ])[:self.max_target_length]

        return {
            'input_ids': article_encoding['input_ids'].squeeze(),
            'attention_mask': article_encoding['attention_mask'].squeeze(),
            'decoder_input_ids': decoder_input_ids,
            'labels': labels
        }


def load_cnn_dailymail(num_samples: Optional[int] = None, split: str = 'train'):
    
    """
    Load the CNN/DailyMail dataset using the datasets library

    Args:
        num_samples (Optional[int], optional): Number of samples to load. Defaults to None.
        split (str, optional): Split to load (train, validation, or test). Defaults to 'train'.

    Returns:
        Dataset: The loaded dataset
    """
    print(f"📂 Loading CNN/DailyMail dataset ({split} split)...")

    # Load the dataset using the datasets library
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Get the specific split
    split_data = dataset[split]

    # Limit number of samples if specified
    if num_samples is not None:
        split_data = split_data.select(
            range(min(num_samples, len(split_data))))

    print(f"✅ Loaded {len(split_data)} examples from {split} split")

    return split_data


def create_dataloaders(config_path: str = 'configs/training_config_mps.yaml'):
    
    """
    Create data loaders for training and validation.

    Args:
        config_path (str, optional): Path to the YAML configuration file.
            Defaults to 'configs/training_config_mps.yaml'.

    Returns:
        Tuple[DataLoader, DataLoader, AutoTokenizer]: A tuple containing the
            training data loader, validation data loader, and the used tokenizer.
    """
    print("🔧 Creating data loaders...")

    # Load configuration
    with open(config_path, 'r') as f:
        # Handle multi-document YAML files by taking the first document
        configs = list(yaml.safe_load_all(f))
        config = configs[0]  # Use the first document (training config)

    data_config = config['data']
    training_config = config['training']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Add special tokens if they don't exist
    special_tokens = {
        'pad_token': '[PAD]',
        'bos_token': '[CLS]',  # Using CLS as BOS
        'eos_token': '[SEP]',  # Using SEP as EOS
    }

    # Add tokens that don't exist
    for token_type, token in special_tokens.items():
        if getattr(tokenizer, token_type) is None:
            setattr(tokenizer, token_type, token)

    print(f"🔤 Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"🔤 Vocabulary size: {len(tokenizer)}")
    print(f"🔤 Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"🔤 BOS token: {tokenizer.bos_token} (ID: {tokenizer.cls_token_id})")
    print(f"🔤 EOS token: {tokenizer.eos_token} (ID: {tokenizer.sep_token_id})")

    # Load datasets
    train_data = load_cnn_dailymail(data_config['train_samples'], 'train')
    val_data = load_cnn_dailymail(data_config['val_samples'], 'validation')

    # Create dataset instances
    train_dataset = SummarizationDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_input_length=data_config['max_input_length'],
        max_target_length=data_config['max_target_length'],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,  # Using CLS as BOS
        eos_token_id=tokenizer.sep_token_id   # Using SEP as EOS
    )

    val_dataset = SummarizationDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_input_length=data_config['max_input_length'],
        max_target_length=data_config['max_target_length'],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=data_config.get('pin_memory', False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 2),
        pin_memory=data_config.get('pin_memory', False)
    )

    print(f"✅ Train loader: {len(train_loader)} batches")
    print(f"✅ Val loader: {len(val_loader)} batches")
    print(f"✅ Batch size: {training_config['batch_size']}")

    return train_loader, val_loader, tokenizer


def inspect_data(dataloader, tokenizer, num_examples: int = 2):
    """
    Inspect data examples to understand the format

    Args:
        dataloader: DataLoader containing the dataset examples.
        tokenizer: Tokenizer used for decoding the input and target texts.
        num_examples (int, optional): Number of examples to inspect. Defaults to 2.

    Prints:
        Inspecting data examples confirmation.
        Example details including shapes and decoded text.
    """
    print("🔍 Inspecting data examples...")

    for i, batch in enumerate(dataloader):
        if i >= num_examples:
            break

        print(f"\n📋 Example {i+1}:")
        print(f"Batch shape:")
        print(f"  Input IDs: {batch['input_ids'].shape}")
        print(f"  Attention mask: {batch['attention_mask'].shape}")
        print(f"  Decoder input: {batch['decoder_input_ids'].shape}")
        print(f"  Labels: {batch['labels'].shape}")

        # Decode first example in batch
        input_text = tokenizer.decode(
            batch['input_ids'][0], skip_special_tokens=True)
        decoder_input_text = tokenizer.decode(
            batch['decoder_input_ids'][0], skip_special_tokens=True)
        label_text = tokenizer.decode(
            batch['labels'][0], skip_special_tokens=True)

        print(f"\n📄 Article (first 200 chars): {input_text[:200]}...")
        print(f"🎯 Decoder input: {decoder_input_text}")
        print(f"🏷️ Target summary: {label_text}")


def test_data_loading():
    """Test the complete data loading pipeline"""
    print("🧪 Testing Data Loading Pipeline...")

    try:
        # Test with a small sample
        print("\n1️⃣ Testing small dataset load...")
        small_data = load_cnn_dailymail(num_samples=10, split='train')
        print(f"✅ Small dataset loaded: {len(small_data)} examples")

        # Test tokenizer
        print("\n2️⃣ Testing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.pad_token = '[PAD]'
        test_text = "This is a test article about cats."
        tokens = tokenizer(test_text, return_tensors='pt')
        print(f"✅ Tokenizer working: {tokens['input_ids'].shape}")

        # Test dataset
        print("\n3️⃣ Testing dataset class...")
        dataset = SummarizationDataset(
            data=small_data,
            tokenizer=tokenizer,
            max_input_length=128,
            max_target_length=64
        )

        example = dataset[0]
        print(f"✅ Dataset working: {len(example)} keys")
        print(f"   Keys: {list(example.keys())}")

        # Test dataloader
        print("\n4️⃣ Testing dataloader...")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        print(
            f"✅ DataLoader working: batch size {batch['input_ids'].shape[0]}")

        # Inspect examples
        print("\n5️⃣ Inspecting data examples...")
        inspect_data(dataloader, tokenizer, num_examples=1)

        print("\n🎉 All data loading tests passed!")
        return True

    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def get_data_statistics(config_path: str = 'configs/training_config_mps.yaml'):
    """
    Get statistics about the dataset
    """
    print("📊 Analyzing dataset statistics...")

    # Load a sample of data
    sample_data = load_cnn_dailymail(num_samples=1000, split='train')

    article_lengths = []
    summary_lengths = []

    for item in sample_data:
        article_lengths.append(len(item['article'].split()))
        summary_lengths.append(len(item['highlights'].split()))

    print(f"\n📈 Dataset Statistics (1000 samples):")
    print(f"📄 Article lengths:")
    print(f"   Mean: {np.mean(article_lengths):.1f} words")
    print(f"   Median: {np.median(article_lengths):.1f} words")
    print(f"   Max: {np.max(article_lengths)} words")
    print(f"   Min: {np.min(article_lengths)} words")

    print(f"\n📝 Summary lengths:")
    print(f"   Mean: {np.mean(summary_lengths):.1f} words")
    print(f"   Median: {np.median(summary_lengths):.1f} words")
    print(f"   Max: {np.max(summary_lengths)} words")
    print(f"   Min: {np.min(summary_lengths)} words")

    print(
        f"\n💡 Compression ratio: {np.mean(article_lengths) / np.mean(summary_lengths):.1f}x")


if __name__ == "__main__":
    print("🚀 Testing Data Loading Pipeline...")

    # Run comprehensive tests
    success = test_data_loading()

    if success:
        print("\n📊 Getting dataset statistics...")
        get_data_statistics()

        print("\n🎉 Data loading pipeline ready!")
        print("📋 Next: Create training configuration and start training!")
    else:
        print("\n❌ Please fix data loading issues before proceeding.")
