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
