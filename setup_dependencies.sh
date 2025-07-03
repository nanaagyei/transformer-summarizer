#!/bin/bash
# setup_dependencies.sh - Install dependencies for existing UV project

set -e  # Exit on any error

echo "⚡ Installing dependencies for Transformer Summarizer..."
echo "====================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ pyproject.toml not found. Please run this script from your project root.${NC}"
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ UV not found. Please install UV first.${NC}"
    exit 1
fi

# Detect platform and architecture
PLATFORM=$(uname -s)
ARCH=$(uname -m)
print_info "UV version: $(uv --version)"
print_info "Platform: $PLATFORM $ARCH"
print_info "Detected $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) CPU cores"
print_info "Python version: $(python --version 2>/dev/null || echo 'Not activated')"

# Determine CPU optimizations based on platform
INTEL_OPTIMIZATIONS=""
if [[ "$PLATFORM" == "Linux" && "$ARCH" == "x86_64" ]]; then
    INTEL_OPTIMIZATIONS="intel-extension-for-pytorch"
    print_info "Linux x86_64 detected - Intel optimizations available"
elif [[ "$PLATFORM" == "Darwin" && "$ARCH" == "arm64" ]]; then
    print_info "macOS Apple Silicon detected - using alternative optimizations"
elif [[ "$PLATFORM" == "Darwin" && "$ARCH" == "x86_64" ]]; then
    INTEL_OPTIMIZATIONS="intel-extension-for-pytorch"
    print_info "macOS Intel detected - Intel optimizations available"
elif [[ "$PLATFORM" == "MINGW"* || "$PLATFORM" == "MSYS_NT"* || "$PLATFORM" == "CYGWIN"* ]]; then
    print_info "Windows detected - using standard optimizations"
else
    print_info "Platform: $PLATFORM $ARCH - using standard optimizations"
fi

# Update pyproject.toml with dependencies
print_info "Creating platform-specific pyproject.toml..."

# Create base pyproject.toml
cat > pyproject.toml << EOF
[project]
name = "transformer-summarizer"
version = "0.1.0"
description = "CPU-optimized Transformer for text summarization"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    # Core ML libraries (CPU versions)
    "torch>=2.0.0",
    "torchvision",
    "torchaudio", 
    "transformers>=4.30.0",
    "datasets>=2.12.0",
    "tokenizers>=0.13.0",
    
    # Data processing
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    
    # Visualization
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    
    # Jupyter
    "jupyter",
    "notebook",
    "ipykernel",
    "ipywidgets",
    
    # Optimization & deployment
    "onnx>=1.14.0",
    "onnxruntime>=1.15.0",
    "gradio>=3.35.0",
    "streamlit>=1.24.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.22.0",
    
    # Experiment tracking
    "wandb>=0.15.0",
    "mlflow>=2.4.0",
    "tensorboard>=2.13.0",
    
    # Data versioning
    "dvc[all]>=3.0.0",
    
    # Evaluation
    "rouge-score>=0.1.2",
    "nltk>=3.8.1",
    "sacrebleu>=2.3.0",
    "bert-score>=0.3.13",
    
    # Development tools
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.4.0",
    "pre-commit>=3.3.0",
    
    # Documentation
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.1.0",
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
    
    # System utilities
    "psutil>=5.9.0",
    "click>=8.1.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
EOF

# Add platform-specific optimizations
if [[ -n "$INTEL_OPTIMIZATIONS" ]]; then
    cat >> pyproject.toml << EOF
intel = [
    "intel-extension-for-pytorch",
    "openvino-dev",
]
EOF
else
    cat >> pyproject.toml << EOF
# Platform-specific optimizations not available for this system
# Alternative optimizations will be used
accelerate = [
    "accelerate>=0.20.0",
]
EOF
fi

# Add rest of pyproject.toml with proper build configuration
cat >> pyproject.toml << 'EOF'

dev = [
    "pytest-xdist",  # Parallel testing
    "coverage",
    "bandit",        # Security linting
    "safety",        # Dependency security
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/configs",
    "/scripts",
    "/docs",
]

[tool.black]
line-length = 88
target-version = ['py311']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=src --cov-report=html --cov-report=term"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
EOF

print_status "pyproject.toml created"

# Install dependencies with UV based on platform
print_info "Installing CPU-optimized dependencies (this may take 2-3 minutes)..."

# Install base dependencies
uv sync --extra dev

# Try to install platform-specific optimizations
if [[ -n "$INTEL_OPTIMIZATIONS" ]]; then
    print_info "Installing Intel optimizations for x86_64..."
    uv sync --extra intel || print_warning "Intel optimizations failed to install"
else
    print_info "Installing alternative optimizations for Apple Silicon/other platforms..."
    uv sync --extra accelerate || print_warning "Alternative optimizations failed to install"
fi

print_status "Core dependencies installed!"

# Platform-specific verification
print_info "Verifying platform-specific imports..."
if [[ "$PLATFORM" == "Darwin" && "$ARCH" == "arm64" ]]; then
    # macOS Apple Silicon specific verification
    uv run python -c "
import torch
import transformers  
import datasets
print(f'✅ PyTorch {torch.__version__} (CPU - Apple Silicon)')
print(f'✅ Transformers {transformers.__version__}')
print(f'✅ Datasets {datasets.__version__}')
print(f'✅ Device: {torch.device(\"cpu\")}')
print(f'✅ MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
print(f'✅ Threads: {torch.get_num_threads()}')
" || print_warning "Some imports failed - will continue setup"
else
    # Standard verification for other platforms
    uv run python -c "
import torch
import transformers  
import datasets
print(f'✅ PyTorch {torch.__version__} (CPU)')
print(f'✅ Transformers {transformers.__version__}')
print(f'✅ Datasets {datasets.__version__}')
print(f'✅ Device: {torch.device(\"cpu\")}')
print(f'✅ Threads: {torch.get_num_threads()}')
" || print_warning "Some imports failed - will continue setup"
fi

# Create CPU optimization utility
print_info "Creating CPU optimization utilities and project structure..."

# Ensure proper package structure exists
mkdir -p src/transformer_summarizer/{models,training,evaluation,data,optimization,utils}

# Create __init__.py files for proper Python package structure
touch src/__init__.py
touch src/transformer_summarizer/__init__.py
touch src/transformer_summarizer/models/__init__.py
touch src/transformer_summarizer/training/__init__.py
touch src/transformer_summarizer/evaluation/__init__.py
touch src/transformer_summarizer/data/__init__.py
touch src/transformer_summarizer/optimization/__init__.py
touch src/transformer_summarizer/utils/__init__.py

print_status "Package structure created"

# Create utils module
cat > src/transformer_summarizer/__init__.py << 'EOF'
"""Transformer Summarizer Package"""
__version__ = "0.1.0"
EOF

cat > src/transformer_summarizer/utils/__init__.py << 'EOF'
"""Utilities for the transformer summarizer project."""
from .cpu_optimizations import optimize_cpu_performance, print_system_info, get_memory_info

__all__ = ["optimize_cpu_performance", "print_system_info", "get_memory_info"]
EOF

cat > src/transformer_summarizer/utils/cpu_optimizations.py << 'EOF'
"""CPU optimization utilities for PyTorch"""
import torch
import os
import psutil
from typing import Dict, Any

def optimize_cpu_performance() -> Dict[str, Any]:
    """
    Optimize PyTorch for CPU performance
    
    Returns:
        Dict with optimization info
    """
    # Get CPU info
    cpu_count = os.cpu_count() or 4
    physical_cores = psutil.cpu_count(logical=False) or 4
    
    # Set optimal thread count based on platform
    if os.uname().sysname == 'Darwin' and os.uname().machine == 'arm64':
        # Apple Silicon optimization
        optimal_threads = min(6, physical_cores)  # Apple Silicon benefits from more threads
    else:
        # Intel/AMD optimization
        optimal_threads = min(4, physical_cores)
    
    torch.set_num_threads(optimal_threads)
    
    # Set environment variables for CPU libraries
    env_vars = {
        'OMP_NUM_THREADS': str(optimal_threads),
        'MKL_NUM_THREADS': str(optimal_threads), 
        'NUMEXPR_NUM_THREADS': str(optimal_threads),
        'TOKENIZERS_PARALLELISM': 'false',  # Avoid tokenizer warnings
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Enable CPU optimizations
    mkldnn_enabled = False
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
        mkldnn_enabled = torch.backends.mkldnn.enabled
    
    # Platform-specific optimizations
    mps_available = False
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        mps_available = True
        print("🍎 Apple MPS backend available (consider using for acceleration)")
    
    # Disable CUDNN (CPU only)
    torch.backends.cudnn.enabled = False
    
    optimization_info = {
        'threads': torch.get_num_threads(),
        'physical_cores': physical_cores,
        'logical_cores': cpu_count,
        'mkldnn_enabled': mkldnn_enabled,
        'optimal_threads': optimal_threads,
        'platform': f"{os.uname().sysname} {os.uname().machine}",
        'mps_available': mps_available,
    }
    
    print(f"🖥️ CPU optimized for {optimization_info['platform']}:")
    print(f"   • Threads: {optimization_info['threads']}")
    print(f"   • Physical cores: {optimization_info['physical_cores']}")
    print(f"   • Logical cores: {optimization_info['logical_cores']}")
    print(f"   • MKLDNN: {optimization_info['mkldnn_enabled']}")
    if mps_available:
        print(f"   • MPS: Available (Apple Silicon acceleration)")
    
    return optimization_info

def get_memory_info() -> Dict[str, float]:
    """Get current memory usage info"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent': memory.percent
    }

def print_system_info() -> None:
    """Print comprehensive system information"""
    memory = get_memory_info()
    cpu_info = {
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'cpu_freq': psutil.cpu_freq(),
        'platform': f"{os.uname().sysname} {os.uname().machine}",
    }
    
    print(f"💻 System Information:")
    print(f"   🖥️ Platform: {cpu_info['platform']}")
    print(f"   🖥️ CPU: {cpu_info['physical_cores']} physical, {cpu_info['logical_cores']} logical cores")
    if cpu_info['cpu_freq']:
        print(f"   ⚡ Frequency: {cpu_info['cpu_freq'].current:.0f} MHz")
    print(f"   💾 Memory: {memory['total_gb']:.1f} GB total, {memory['available_gb']:.1f} GB available")
    print(f"   📊 Usage: {memory['percent']:.1f}%")
    
    # Platform-specific info
    if os.uname().sysname == 'Darwin' and os.uname().machine == 'arm64':
        print(f"   🍎 Apple Silicon optimizations enabled")
    elif os.uname().sysname == 'Linux' and os.uname().machine == 'x86_64':
        print(f"   🐧 Linux x86_64 optimizations enabled")

if __name__ == "__main__":
    optimize_cpu_performance()
    print_system_info()
EOF

print_status "CPU optimization utilities created"

# Download NLTK data
print_info "Downloading NLTK data..."
uv run python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ NLTK data downloaded')
except Exception as e:
    print(f'⚠️ NLTK download failed: {e}')
" || print_warning "NLTK download failed, will continue"

# Create training script
cat > scripts/train_cpu.py << 'EOF'
#!/usr/bin/env python3
"""CPU-optimized training script with UV"""
import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """Main training function"""
    try:
        from src.transformer_summarizer.utils.cpu_optimizations import optimize_cpu_performance, print_system_info
    except ImportError:
        print("⚠️ Could not import CPU optimizations - using basic setup")
        return
    
    print("🖥️ Starting CPU-optimized training...")
    print("=" * 50)
    
    # Print system info
    print_system_info()
    print()
    
    # Optimize CPU performance
    opt_info = optimize_cpu_performance()
    print()
    
    # Import after optimization
    try:
        # from src.transformer_summarizer.training.trainer import TransformerTrainer
        # from src.transformer_summarizer.data.dataset import create_dataloaders
        
        config_path = project_root / 'configs' / 'training_config_cpu.yaml'
        
        print(f"📋 Using config: {config_path}")
        
        if not config_path.exists():
            print(f"⚠️ Config file not found: {config_path}")
            print("Please create the configuration files first.")
            return
        
        print("🚀 Training will start here once model implementation is complete!")
        print("💡 For now, this script validates the setup.")
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        print("This is expected until model implementation is complete.")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/train_cpu.py
print_status "Training script created"

# Create development setup script
cat > scripts/setup_dev.py << 'EOF'
#!/usr/bin/env python3
"""Development setup script"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    """Setup development environment"""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🛠️ Setting up development environment...")
    
    # Install pre-commit hooks
    run_command("uv run pre-commit install", "Installing pre-commit hooks")
    
    # Create initial test
    test_dir = project_root / "tests"
    test_dir.mkdir(exist_ok=True)
    
    with open(test_dir / "__init__.py", "w") as f:
        f.write("# Test package\n")
    
    with open(test_dir / "test_setup.py", "w") as f:
        f.write('''"""Test basic setup"""
import torch
import sys
from pathlib import Path

def test_torch_cpu():
    """Test PyTorch CPU functionality"""
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.mm(x, y)
    assert z.shape == (3, 3)
    assert not z.is_cuda

def test_project_structure():
    """Test project structure exists"""
    project_root = Path(__file__).parent.parent
    required_dirs = ["src", "configs", "scripts", "tests"]
    
    for dir_name in required_dirs:
        assert (project_root / dir_name).exists(), f"Missing {dir_name} directory"

def test_imports():
    """Test basic imports work"""
    import torch
    import transformers
    import datasets
    assert torch.__version__
    print(f"✅ PyTorch {torch.__version__} (CPU)")
''')
    
    # Run initial test
    run_command("uv run pytest tests/test_setup.py -v", "Running setup tests")
    
    print("🎉 Development environment ready!")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/setup_dev.py

# Final setup verification
print_info "Running final verification..."
echo ""
echo "🔍 Dependency Check:"
uv run python -c "
import sys
packages = ['torch', 'transformers', 'datasets', 'gradio', 'wandb', 'pytest']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  ✅ {pkg}')
    except ImportError:
        print(f'  ❌ {pkg} - failed to import')

print()
print('🖥️ System Info:')
import torch
print(f'  • PyTorch: {torch.__version__}')
print(f'  • CPU threads: {torch.get_num_threads()}')
print(f'  • Python: {sys.version.split()[0]}')
"

echo ""
echo "🎉 Platform-specific dependency installation complete!"
echo "=============================================="
echo ""
echo "🖥️ Platform detected: $PLATFORM $ARCH"

if [[ "$PLATFORM" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "🍎 Apple Silicon optimizations:"
    echo "   • Increased thread count for M-series chips"
    echo "   • MPS backend detection (for potential GPU acceleration)"
    echo "   • Alternative optimization packages"
elif [[ -n "$INTEL_OPTIMIZATIONS" ]]; then
    echo "⚡ Intel optimizations installed:"
    echo "   • Intel Extension for PyTorch"
    echo "   • OpenVINO development tools"
else
    echo "🔧 Standard CPU optimizations:"
    echo "   • Platform-specific thread optimization"
    echo "   • Standard PyTorch CPU backend"
fi

echo ""
echo "📦 Installed packages:"
echo "   • PyTorch (CPU optimized for your platform)"
echo "   • Transformers & Datasets"
echo "   • Gradio & FastAPI (deployment)"
echo "   • Pytest & development tools"
echo "   • Experiment tracking (wandb, mlflow)"
echo "   • Documentation tools"
echo ""
echo "🚀 Next steps:"
echo "   1. Create configuration files:"
echo "      mkdir -p configs"
echo "      # Copy the config files provided"
echo ""
echo "   2. Test your setup:"
echo "      uv run python scripts/train_cpu.py"
echo ""
echo "   3. Start implementing the model:"
echo "      # Follow the implementation guide"
echo ""
echo "   4. Run development setup:"
echo "      uv run python scripts/setup_dev.py"
echo ""

if [[ "$PLATFORM" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "💡 Apple Silicon tips:"
    echo "   • Your M-series chip is excellent for ML development"
    echo "   • Consider using MPS for some operations (if supported)"
    echo "   • Memory is unified - you have more effective RAM"
    echo ""
fi

echo "💡 Remember: Use 'uv run python script.py' for all commands!"
print_status "Ready to start building your Transformer! 🤖"