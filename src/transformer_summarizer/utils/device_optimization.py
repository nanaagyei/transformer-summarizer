# src/transformer_summarizer/utils/device_optimization.py
"""Device optimization utilities for Apple Silicon and other platforms"""
import torch
import os
import psutil
from typing import Dict, Any, Tuple

def get_optimal_device() -> Tuple[torch.device, Dict[str, Any]]:
    """
    Determine the optimal device for training/inference
    
    Returns:
        Tuple of (device, device_info)
    """
    device_info = {
        'device_type': 'cpu',
        'device_name': 'CPU',
        'memory_gb': 0,
        'optimization_notes': []
    }
    
    # Check for CUDA (unlikely on Mac, but good to check)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_info.update({
            'device_type': 'cuda',
            'device_name': torch.cuda.get_device_name(0),
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'optimization_notes': ['CUDA GPU acceleration available']
        })
        return device, device_info
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_info.update({
            'device_type': 'mps',
            'device_name': 'Apple Silicon GPU (MPS)',
            'memory_gb': get_apple_silicon_memory(),
            'optimization_notes': [
                'Apple Silicon GPU acceleration available',
                'Unified memory architecture',
                'Excellent for transformer models'
            ]
        })
        return device, device_info
    
    # Fallback to CPU
    device = torch.device('cpu')
    memory = psutil.virtual_memory()
    device_info.update({
        'device_type': 'cpu',
        'device_name': f'CPU ({psutil.cpu_count(logical=False)} cores)',
        'memory_gb': memory.total / 1024**3,
        'optimization_notes': ['CPU-only training (still very capable!)']
    })
    
    return device, device_info

def get_apple_silicon_memory() -> float:
    """Estimate available memory for MPS on Apple Silicon"""
    total_memory = psutil.virtual_memory().total / 1024**3
    # Apple Silicon has unified memory, estimate ~70% available for GPU tasks
    return total_memory * 0.7

def optimize_for_device(device: torch.device) -> Dict[str, Any]:
    """
    Apply device-specific optimizations
    
    Args:
        device: Target device for training
        
    Returns:
        Optimization settings
    """
    if device.type == 'mps':
        return optimize_for_mps()
    elif device.type == 'cuda':
        return optimize_for_cuda()
    else:
        return optimize_for_cpu()

def optimize_for_mps() -> Dict[str, Any]:
    """Optimizations specific to Apple Silicon MPS"""
    # Set environment variables for MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Fallback to CPU for unsupported ops
    
    # MPS-specific settings
    optimization_settings = {
        'device_type': 'mps',
        'batch_size_multiplier': 1.5,  # MPS can handle slightly larger batches
        'gradient_accumulation_steps': 2,  # Reduce accumulation since batches can be larger
        'memory_fraction': 0.8,  # Use 80% of available memory
        'mixed_precision': False,  # MPS doesn't support AMP yet
        'compile_model': False,  # torch.compile not fully supported on MPS
        'dataloader_workers': 4,  # MPS benefits from more workers
        'pin_memory': False,  # Not needed for MPS
        'optimization_notes': [
            'Using Apple Silicon GPU acceleration',
            'Unified memory allows larger effective batch sizes',
            'Automatic fallback to CPU for unsupported operations'
        ]
    }
    
    print("🍎 MPS (Apple Silicon GPU) optimizations enabled:")
    for note in optimization_settings['optimization_notes']:
        print(f"   • {note}")
    
    return optimization_settings

def optimize_for_cuda() -> Dict[str, Any]:
    """Optimizations specific to CUDA GPUs"""
    optimization_settings = {
        'device_type': 'cuda',
        'batch_size_multiplier': 2.0,
        'gradient_accumulation_steps': 2,
        'memory_fraction': 0.9,
        'mixed_precision': True,  # CUDA supports AMP
        'compile_model': True,  # torch.compile works well on CUDA
        'dataloader_workers': 4,
        'pin_memory': True,
        'optimization_notes': ['CUDA GPU acceleration with mixed precision']
    }
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    return optimization_settings

def optimize_for_cpu() -> Dict[str, Any]:
    """Optimizations specific to CPU training"""
    # Get CPU info
    cpu_count = os.cpu_count() or 4
    physical_cores = psutil.cpu_count(logical=False) or 4
    
    # Apple Silicon CPUs can handle more threads
    if os.uname().sysname == 'Darwin' and os.uname().machine == 'arm64':
        optimal_threads = min(6, physical_cores)
    else:
        optimal_threads = min(4, physical_cores)
    
    torch.set_num_threads(optimal_threads)
    
    # Set environment variables
    env_vars = {
        'OMP_NUM_THREADS': str(optimal_threads),
        'MKL_NUM_THREADS': str(optimal_threads),
        'NUMEXPR_NUM_THREADS': str(optimal_threads),
        'TOKENIZERS_PARALLELISM': 'false'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    optimization_settings = {
        'device_type': 'cpu',
        'batch_size_multiplier': 1.0,
        'gradient_accumulation_steps': 4,
        'memory_fraction': 0.7,
        'mixed_precision': False,
        'compile_model': False,
        'dataloader_workers': 2,
        'pin_memory': False,
        'threads': optimal_threads,
        'optimization_notes': [f'CPU optimized with {optimal_threads} threads']
    }
    
    return optimization_settings

def print_device_info(device: torch.device, device_info: Dict[str, Any], 
                     optimization_settings: Dict[str, Any]) -> None:
    """Print comprehensive device information"""
    print(f"🖥️ Device Configuration:")
    print(f"   • Device: {device}")
    print(f"   • Type: {device_info['device_name']}")
    print(f"   • Memory: {device_info['memory_gb']:.1f} GB")
    
    if device.type == 'mps':
        print(f"   • Unified Memory: Shared CPU/GPU memory pool")
        print(f"   • Fallback: CPU for unsupported operations")
    
    print(f"   • Batch size multiplier: {optimization_settings['batch_size_multiplier']}x")
    print(f"   • Workers: {optimization_settings['dataloader_workers']}")
    
    for note in optimization_settings['optimization_notes']:
        print(f"   • {note}")

def setup_optimal_training_environment():
    """
    Complete setup for optimal training environment
    
    Returns:
        Tuple of (device, optimization_settings)
    """
    print("🔧 Setting up optimal training environment...")
    
    # Get optimal device
    device, device_info = get_optimal_device()
    
    # Apply device-specific optimizations
    optimization_settings = optimize_for_device(device)
    
    # Print configuration
    print_device_info(device, device_info, optimization_settings)
    
    print(f"✅ Training environment optimized for {device.type.upper()}!")
    
    return device, optimization_settings

# Convenience function for quick device setup
def get_device():
    """Simple function to get the best available device"""
    device, _ = get_optimal_device()
    return device

if __name__ == "__main__":
    # Test the device optimization
    device, settings = setup_optimal_training_environment()
    
    # Test tensor operations
    print(f"\n🧪 Testing tensor operations on {device}...")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.mm(x, y)
    print(f"✅ Matrix multiplication successful on {device}")
    print(f"   Result shape: {z.shape}")
    print(f"   Result device: {z.device}")