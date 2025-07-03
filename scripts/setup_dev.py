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
