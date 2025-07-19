# Transformer Summarizer

A comprehensive implementation of a transformer-based text summarization model with support for local and cloud training.

## 🚀 Quick Start

### Local Training (MPS/CPU)

```bash
# Install dependencies
uv sync  # or pip install -r requirements.txt

# Train on Apple Silicon GPU (MPS)
python scripts/train.py --config configs/training_config_mps.yaml

# Train on CPU
python scripts/train.py --config configs/training_config_cpu.yaml
```

### Cloud Training (Recommended)

```bash
# Setup cloud training (see docs/setup_guide.md for detailed instructions)
# 1. Copy and configure template files
cp configs/vast_ai_template.json vast_job.json
# Edit vast_job.json with your settings

# 2. Search and select instance
python scripts/deploy.py --action search

# 3. Create instance with selected ID
python scripts/deploy.py --action create --instance-id <selected_instance_id>

# 4. SSH into instance to monitor training
python scripts/deploy.py --action ssh --instance-id <instance_id>
```

## 📊 Cloud Training Options

| Provider       | Cost          | Ease of Use | Best For             |
| -------------- | ------------- | ----------- | -------------------- |
| **Vast.ai**    | $0.20-3.00/hr | Medium      | Cost-conscious users |
| **Modal**      | $0.40-4.00/hr | High        | Serverless workflows |
| **JarvisLabs** | $0.50-8.00/hr | High        | ML teams             |

**💡 Why Cloud Training?**

- **Faster training**: 2-8x speedup vs local
- **Better GPUs**: RTX 4090, A100, H100
- **Cost-effective**: $0.70-18.00 for full training
- **No hardware limitations**: Scale as needed

## 🏗️ Project Structure

```
transformer-summarizer/
├── src/transformer_summarizer/
│   ├── models/              # Transformer model implementation
│   ├── training/            # Training pipeline
│   ├── data/                # Dataset handling
│   ├── evaluation/          # Model evaluation
│   └── utils/               # Utility functions
├── configs/                 # Training configurations & templates
├── scripts/                 # Training and deployment scripts
├── experiments/             # Training results and models (gitignored)
├── data/                    # Datasets (gitignored)
├── docs/                    # Documentation
├── tests/                   # Unit tests
└── .local_config/           # Sensitive configs (gitignored)
```

## 🔧 Features

### Model Architecture

- **Transformer-based**: Attention mechanism for sequence modeling
- **Encoder-Decoder**: Bidirectional encoding, autoregressive decoding
- **Multi-head Attention**: Parallel attention heads
- **Position Encoding**: Learned positional embeddings
- **Layer Normalization**: Stable training

### Training Features

- **Multi-device support**: CPU, MPS (Apple Silicon), CUDA
- **Cloud training**: Vast.ai, Modal, JarvisLabs
- **Mixed precision**: FP16 training for speed
- **Gradient accumulation**: Large effective batch sizes
- **Learning rate scheduling**: Warmup and decay
- **Checkpointing**: Model and training state saving
- **Early stopping**: Prevent overfitting

### Evaluation

- **ROUGE scores**: ROUGE-1, ROUGE-2, ROUGE-L
- **BLEU score**: Bilingual evaluation
- **Custom metrics**: Extensible evaluation framework

## 📈 Performance

| Environment          | Training Time | Cost  | GPU Memory | Batch Size |
| -------------------- | ------------- | ----- | ---------- | ---------- |
| **Local MPS**        | 8 hours       | $0    | 8GB        | 6          |
| **Vast.ai RTX 4090** | 2 hours       | $0.70 | 24GB       | 16         |
| **Modal A100**       | 1 hour        | $2.00 | 40GB       | 32         |
| **JarvisLabs H100**  | 30 min        | $2.00 | 80GB       | 64         |

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.1+
- CUDA (for GPU training) or MPS (Apple Silicon)

### Local Setup

```bash
# Clone repository
git clone https://github.com/yourusername/transformer-summarizer.git
cd transformer-summarizer

# Install dependencies with uv (recommended)
uv sync

# Or install with pip
pip install -r requirements.txt

# Setup development environment
python scripts/setup_dev.py
```

### Cloud Setup

See `docs/setup_guide.md` for detailed cloud setup instructions.

## 🚀 Usage

### Local Training

#### Apple Silicon (MPS)

```bash
python scripts/train.py --config configs/training_config_mps.yaml
```

#### CPU Training

```bash
python scripts/train.py --config configs/training_config_cpu.yaml
```

### Cloud Training

#### Vast.ai (Recommended for cost)

```bash
# Search and select instance
python scripts/deploy.py --action search

# Create instance with selected ID
python scripts/deploy.py --action create --instance-id <selected_instance_id>

# SSH into instance to monitor training
python scripts/deploy.py --action ssh --instance-id <instance_id>
```

#### Modal (Serverless)

```bash
python scripts/cloud_train.py --provider modal --config configs/training_config_cloud.yaml
```

#### JarvisLabs (ML-optimized)

```bash
python scripts/cloud_train.py --provider jarvis --config configs/training_config_cloud.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --model-path experiments/models/best_model.pth
```

### Inference

```bash
python main.py --text "Your text to summarize here"
```

## ⚙️ Configuration

### Training Configurations

- `configs/training_config_cpu.yaml` - CPU-optimized training
- `configs/training_config_mps.yaml` - Apple Silicon training
- `configs/training_config_cloud.yaml` - Cloud GPU training

### Model Configurations

- `configs/model_config_cpu.yaml` - CPU-optimized model
- `configs/model_config_mps.yaml` - Apple Silicon model
- `configs/model_config.yaml` - Default model configuration

### Cloud Templates

- `configs/vast_ai_template.json` - Vast.ai configuration template
- `configs/cloud_setup_template.sh` - Cloud startup script template

## 📊 Monitoring

The project supports Weights & Biases for experiment tracking. Set up your W&B account and configure the project name in the training configs.

## 🔐 Security

Sensitive configuration files (SSH keys, cloud credentials, etc.) are stored in `.local_config/` and are not committed to the repository. See `docs/setup_guide.md` for setup instructions.

## 📚 Documentation

- `docs/setup_guide.md` - Detailed setup instructions
- `docs/training.md` - Training guide
- `docs/cloud_training.md` - Cloud training guide
- `docs/api.md` - API documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- Hugging Face for transformer implementations
- Vast.ai, Modal, and JarvisLabs for cloud GPU access
