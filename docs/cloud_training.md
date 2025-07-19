# Cloud Training Guide for Transformer Summarizer

This guide will help you set up and run training on cloud GPU providers instead of your local machine.

## 🚀 Quick Start

1. **Setup cloud training infrastructure:**

   ```bash
   python scripts/setup_cloud_training.py --all
   ```

2. **Choose your provider and start training:**

   ```bash
   # For Vast.ai (most cost-effective)
   python scripts/cloud_train.py --provider vast --config configs/training_config_cloud.yaml

   # For Modal (serverless)
   python scripts/cloud_train.py --provider modal --config configs/training_config_cloud.yaml

   # For JarvisLabs (ML-optimized)
   python scripts/cloud_train.py --provider jarvis --config configs/training_config_cloud.yaml
   ```

## 📊 Cloud Provider Comparison

| Provider       | Cost          | Ease of Use | Best For                                |
| -------------- | ------------- | ----------- | --------------------------------------- |
| **Vast.ai**    | $0.20-3.00/hr | Medium      | Cost-conscious users, batch training    |
| **Modal**      | $0.40-4.00/hr | High        | Quick experiments, serverless workflows |
| **JarvisLabs** | $0.50-8.00/hr | High        | ML teams, interactive development       |

## 💰 Cost Estimates

### Small Experiment (2 hours)

- **Vast.ai RTX 4090**: $0.70
- **Modal T4**: $0.80
- **JarvisLabs RTX 4090**: $1.50

### Medium Training (8 hours)

- **Vast.ai RTX 4090**: $2.80
- **Modal T4**: $3.20
- **JarvisLabs RTX 4090**: $6.00

### Large Training (24 hours)

- **Vast.ai RTX 4090**: $8.40
- **Modal T4**: $9.60
- **JarvisLabs RTX 4090**: $18.00

## 🔧 Detailed Setup Instructions

### Vast.ai Setup

**Advantages:**

- Lowest cost per hour
- Pay-per-use pricing
- Spot instances available
- Wide range of GPU options
- **Official PyTorch support and documentation**

**Setup Steps:**

1. Sign up at [vast.ai](https://vast.ai/console/account)
2. Add credits to your account
3. Install CLI: `pip install vast-ai`
4. Set API key: `export VAST_API_KEY=your_key`
5. Generate SSH key: `ssh-keygen -t rsa -b 4096 -C 'your_email@example.com'`
6. Search for instances: `vastai search offers 'gpu_name:RTX_4090 num_gpus:1 disk_space:>100'`
7. Run training: `python scripts/cloud_train.py --provider vast`

**Recommended Configuration:**

- GPU: RTX 4090 ($0.30-0.50/hr)
- Disk: 100GB (increased for datasets and checkpoints)
- Price max: $0.50/hr
- PyTorch: 2.1.0 with CUDA 12.1

**Vast.ai PyTorch Best Practices:**

- Use official PyTorch Docker image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- Use uv for fast Python package management
- Minimum 8GB GPU memory
- Minimum 50GB disk space
- Minimum 100Mbps internet speed
- Monitor GPU utilization with `nvidia-smi`
- Use spot instances for cost savings
- Set up proper error handling and logging

**Official Documentation:** [Vast.ai PyTorch Guide](https://docs.vast.ai/pytorch)

### Modal Setup

**Advantages:**

- Serverless - no instance management
- Pay only for actual compute time
- Easy deployment
- Automatic scaling

**Setup Steps:**

1. Sign up at [modal.com](https://modal.com)
2. Install CLI: `pip install modal`
3. Authenticate: `modal token new`
4. Run training: `python scripts/cloud_train.py --provider modal`

**Recommended Configuration:**

- GPU: T4 for development, A100 for production
- Memory: 16GB
- Timeout: 1 hour (adjustable)

### JarvisLabs Setup

**Advantages:**

- Specialized for ML/AI workloads
- Easy-to-use web interface
- Integrated Jupyter notebooks
- Team collaboration features

**Setup Steps:**

1. Sign up at [jarvislabs.ai](https://jarvislabs.ai)
2. Create a new project
3. Choose GPU instance type
4. Upload code or connect via Git
5. Run training: `python scripts/cloud_train.py --provider jarvis`

**Recommended Configuration:**

- Instance: gpu.1x with RTX 4090
- Framework: PyTorch
- Python: 3.11

## 📁 Project Structure for Cloud Training

```
transformer-summarizer/
├── scripts/
│   ├── cloud_train.py          # Main cloud training script
│   ├── train_remote.py         # Remote training execution
│   └── setup_cloud_training.py # Cloud provider setup
├── configs/
│   ├── training_config_cloud.yaml  # Cloud-optimized config
│   ├── training_config_mps.yaml    # Local MPS config
│   └── training_config_cpu.yaml    # Local CPU config
├── docs/
│   └── cloud_training.md       # This guide
└── experiments/
    └── remote_training/        # Cloud training results
```

## ⚙️ Configuration Files

### Cloud-Optimized Configuration (`configs/training_config_cloud.yaml`)

Key differences from local training:

- **Larger datasets**: 50K training samples vs 10K local
- **Bigger batch sizes**: 16 vs 4-6 local
- **More epochs**: 15 vs 5-8 local
- **Mixed precision**: Enabled for faster training
- **Cloud storage**: Results saved to cloud storage

### Provider-Specific Configurations

Each provider creates its own configuration files:

- `vast_job.json` - Vast.ai job configuration
- `modal_app.py` - Modal serverless app
- `jarvis_config.json` - JarvisLabs instance config

## 🔄 Training Workflow

1. **Local Setup** (on your laptop):

   ```bash
   # Setup cloud training infrastructure
   python scripts/setup_cloud_training.py --all

   # Choose provider and start
   python scripts/cloud_train.py --provider vast
   ```

2. **Cloud Execution** (on remote GPU):

   ```bash
   # The remote script automatically:
   # - Sets up environment
   # - Downloads/prepares data
   # - Trains model
   # - Saves results
   # - Uploads to cloud storage
   ```

3. **Results Retrieval**:
   - Results saved to `experiments/remote_training/`
   - Model checkpoints and logs included
   - Training metrics and final model

## 🛠️ Advanced Configuration

### Custom GPU Selection

**Vast.ai:**

```bash
# Edit vast_job.json to change GPU type
{
  "gpu_type": "A100",  # or "H100", "RTX_3090"
  "price_max": 1.50
}
```

**Modal:**

```python
# Edit modal_app.py to change GPU
@stub.function(
    gpu=modal.gpu.A100(),  # or modal.gpu.H100()
    memory=32000,
)
```

**JarvisLabs:**

```json
// Edit jarvis_config.json
{
  "gpu_type": "A100",
  "instance_type": "gpu.2x"
}
```

### Data Management

For large datasets, consider:

1. **Pre-upload to cloud storage** (S3, GCS)
2. **Use cloud datasets** (HuggingFace Datasets)
3. **Stream data** during training

### Cost Optimization

1. **Use spot instances** (Vast.ai)
2. **Monitor usage** with cloud provider dashboards
3. **Set up billing alerts**
4. **Use appropriate GPU for task** (T4 for dev, A100 for production)

## 🐛 Troubleshooting

### Common Issues

**Vast.ai:**

- SSH connection issues → Check SSH key setup
- Instance not starting → Check credit balance
- Training fails → Check startup script

**Modal:**

- Authentication errors → Run `modal token new`
- Timeout issues → Increase timeout in config
- Memory errors → Increase memory allocation

**JarvisLabs:**

- Instance creation fails → Check quota limits
- Code upload issues → Use Git integration
- Training stops → Check instance status

### Debug Commands

```bash
# Check cloud provider status
python scripts/setup_cloud_training.py --provider compare

# Estimate costs
python scripts/setup_cloud_training.py --provider costs

# Test local setup
python scripts/train_remote.py --config configs/training_config_cloud.yaml
```

## 📈 Performance Comparison

| Environment          | Training Time | Cost  | GPU Memory | Batch Size |
| -------------------- | ------------- | ----- | ---------- | ---------- |
| **Local MPS**        | 8 hours       | $0    | 8GB        | 6          |
| **Vast.ai RTX 4090** | 2 hours       | $0.70 | 24GB       | 16         |
| **Modal A100**       | 1 hour        | $2.00 | 40GB       | 32         |
| **JarvisLabs H100**  | 30 min        | $2.00 | 80GB       | 64         |

## 🔐 Security Considerations

1. **API Keys**: Store securely, never commit to Git
2. **SSH Keys**: Use strong keys, rotate regularly
3. **Data Privacy**: Ensure data doesn't contain sensitive information
4. **Access Control**: Use provider-specific access controls

## 📞 Support

- **Vast.ai**: [Documentation](https://vast.ai/docs/)
- **Modal**: [Documentation](https://modal.com/docs)
- **JarvisLabs**: [Documentation](https://jarvislabs.ai/docs)

## 🎯 Next Steps

1. **Start with Vast.ai** for cost-effective training
2. **Experiment with Modal** for serverless workflows
3. **Use JarvisLabs** for team collaboration
4. **Monitor costs** and optimize configurations
5. **Scale up** as your model improves

---

**Happy Cloud Training! 🚀☁️**
