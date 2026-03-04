#!/usr/bin/env python3
"""
Environment Validation Script
=============================
Quick test to verify all dependencies are working correctly.
"""

import sys
from typing import Dict, Tuple

def test_imports() -> Dict[str, Tuple[bool, str]]:
    """Test importing all required packages"""
    results = {}
    
    packages = [
        ('gymnasium', 'Gymnasium'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('scipy', 'SciPy'),
        ('torch', 'PyTorch'),
        ('stable_baselines3', 'Stable-Baselines3'),
        ('tensorboard', 'TensorBoard')
    ]
    
    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'Unknown')
            results[display_name] = (True, version)
        except ImportError as e:
            results[display_name] = (False, str(e))
    
    return results

def test_gpu_availability() -> Dict[str, bool]:
    """Test GPU availability"""
    import torch
    
    return {
        'CUDA (NVIDIA)': torch.cuda.is_available(),
        'MPS (Apple Silicon)': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }

def main():
    print("="*80)
    print("SUPRA-PPO ENVIRONMENT VALIDATION")
    print("="*80)
    print(f"Python Version: {sys.version}")
    print("="*80)
    
    # Test package imports
    print("\nTESTING PACKAGE IMPORTS:")
    print("-"*80)
    
    results = test_imports()
    all_passed = True
    
    for package, (success, info) in results.items():
        status = "OK" if success else "FAIL"
        print(f"{status:>4} {package:<25} {'v' + info if success else 'FAILED: ' + info}")
        if not success:
            all_passed = False
    
    # Test GPU
    print("\n"+"="*80)
    print("TESTING GPU AVAILABILITY:")
    print("-"*80)
    
    gpu_results = test_gpu_availability()
    for gpu_type, available in gpu_results.items():
        status = "Available" if available else "Not Available"
        print(f"{gpu_type:<25} {status}")
    
    # Test basic functionality
    print("\n"+"="*80)
    print("TESTING BASIC FUNCTIONALITY:")
    print("-"*80)
    
    try:
        import numpy as np
        import pandas as pd
        
        # Test NumPy
        arr = np.array([1, 2, 3, 4, 5])
        print(f"OK: NumPy array creation: {arr.shape}")
        
        # Test Pandas
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"OK: Pandas DataFrame creation: {df.shape}")
        
        # Test PyTorch
        import torch
        tensor = torch.tensor([1.0, 2.0, 3.0])
        print(f"OK: PyTorch tensor creation: {tensor.shape}")
        
        # Test Gymnasium
        import gymnasium as gym
        env = gym.make('CartPole-v1')
        print(f"OK: Gymnasium environment creation: {env.spec.id}")
        env.close()
        
        # Test Stable-Baselines3
        from stable_baselines3 import PPO
        print(f"OK: Stable-Baselines3 PPO import successful")
        
    except Exception as e:
        print(f"FAIL: Functionality test failed: {e}")
        all_passed = False
    
    # Final verdict
    print("\n"+"="*80)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nYour environment is ready for:")
        print("   - Phase I: MCDM Evaluation (python src/mcdm_evaluation.py)")
        print("   - Phase II: RL Experiments (python src/run_experiment.py)")
        print("   - Phase III: Visualisation (python src/visualize_results.py)")
        print("\nTip: Use GPU acceleration for faster training:")
        if gpu_results.get('MPS (Apple Silicon)', False):
            print("   Your M-series Mac supports Metal Performance Shaders (MPS)")
            print("   PyTorch will automatically use it for training!")
        elif gpu_results.get('CUDA (NVIDIA)', False):
            print("   Your NVIDIA GPU is ready for CUDA acceleration!")
        else:
            print("   Training will use CPU (slower but still functional)")
        print("="*80)
        return 0
    else:
        print("SOME TESTS FAILED")
        print("="*80)
        print("\nPlease check the errors above and reinstall if needed:")
        print("   pip install -r requirements.txt")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
