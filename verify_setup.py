"""
Setup Verification Script for 3D CycleGAN
Checks environment, dependencies, and data directories
"""

import sys
import torch
import numpy as np
from pathlib import Path

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_python_version():
    """Check Python version"""
    print_section("Python Version")
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python: {version}")
    
    if sys.version_info >= (3, 8):
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python version too old! Requires 3.8+")
        return False

def check_pytorch():
    """Check PyTorch installation"""
    print_section("PyTorch Installation")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU(s): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - {torch.cuda.get_device_name(i)}")
            print(f"    VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print("✓ GPU acceleration available")
    else:
        print("⚠ GPU not available - training will use CPU (slow)")
    
    return True

def check_dependencies():
    """Check required packages"""
    print_section("Dependencies")
    
    required = {
        'numpy': 'numpy',
        'tifffile': 'tifffile',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
        'skimage': 'scikit-image',
    }
    
    all_ok = True
    for module_name, package_name in required.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name}: {version}")
        except ImportError:
            print(f"✗ {package_name}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check project directories"""
    print_section("Project Directories")
    
    repo_path = Path(__file__).parent
    dirs_to_check = {
        'real': repo_path / 'real',
        'syn': repo_path / 'syn',
        'checkpoints': repo_path / 'checkpoints',
        'logs': repo_path / 'logs',
    }
    
    all_ok = True
    for dir_name, dir_path in dirs_to_check.items():
        if dir_path.exists():
            print(f"✓ {dir_name}/: exists")
        else:
            print(f"⚠ {dir_name}/: does not exist (creating...)")
            dir_path.mkdir(exist_ok=True)
    
    return all_ok

def check_data():
    """Check data in real and syn folders"""
    print_section("Data Verification")
    
    repo_path = Path(__file__).parent
    real_dir = repo_path / 'real'
    syn_dir = repo_path / 'syn'
    
    real_files = sorted(list(real_dir.glob('*.tif')) + list(real_dir.glob('*.tiff')))
    syn_files = sorted(list(syn_dir.glob('*.tif')) + list(syn_dir.glob('*.tiff')))
    
    print(f"Real domain images: {len(real_files)} found")
    if real_files:
        for f in real_files[:3]:
            print(f"  - {f.name}")
        if len(real_files) > 3:
            print(f"  ... and {len(real_files) - 3} more")
    else:
        print("  ⚠ No images found - add TIFF files to real/ folder")
    
    print(f"\nSynthetic domain images: {len(syn_files)} found")
    if syn_files:
        for f in syn_files[:3]:
            print(f"  - {f.name}")
        if len(syn_files) > 3:
            print(f"  ... and {len(syn_files) - 3} more")
    else:
        print("  ⚠ No images found - add TIFF files to syn/ folder")
    
    if real_files and syn_files:
        print(f"\n✓ Ready for training! ({len(real_files)} real + {len(syn_files)} syn images)")
        return True
    else:
        print(f"\n⚠ Need more data! Minimum recommended: 50+ images per domain")
        return False

def check_scripts():
    """Check if required scripts exist"""
    print_section("Script Files")
    
    repo_path = Path(__file__).parent
    scripts = {
        'train_3d_cyclegan.py': 'Training script',
        'inference_3d_cyclegan.py': 'Inference script',
        'README.md': 'Documentation',
    }
    
    all_ok = True
    for script_name, description in scripts.items():
        script_path = repo_path / script_name
        if script_path.exists():
            print(f"✓ {script_name}: {description}")
        else:
            print(f"✗ {script_name}: NOT FOUND")
            all_ok = False
    
    return all_ok

def print_summary(checks):
    """Print summary of all checks"""
    print_section("System Summary")
    
    total = len(checks)
    passed = sum(1 for v in checks.values() if v)
    
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All checks passed! Ready to train.")
        return True
    else:
        print(f"\n⚠ {total - passed} issue(s) detected. Please fix above warnings.")
        return False

def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("  3D CycleGAN Setup Verification")
    print("="*60)
    
    checks = {
        'Python Version': check_python_version(),
        'PyTorch Installation': check_pytorch(),
        'Dependencies': check_dependencies(),
        'Directories': check_directories(),
        'Script Files': check_scripts(),
        'Data': check_data(),
    }
    
    success = print_summary(checks)
    
    print("\n" + "="*60)
    if success:
        print("  Next Steps:")
        print("  1. (Optional) Edit Config in train_3d_cyclegan.py")
        print("  2. Run: python train_3d_cyclegan.py")
        print("="*60 + "\n")
    else:
        print("  Please fix the issues above and run this script again")
        print("="*60 + "\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
