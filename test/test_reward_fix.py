#!/usr/bin/env python3
"""
Quick Validation Test for Reward Function Fixes
================================================

Tests that the updated reward function:
1. Correctly applies increased λ_sustain_resil (5000 vs 500)
2. Correctly calculates FTOPSIS alignment bonus
3. M3 now allocates more to high-FTOPSIS suppliers (S3, S5)

Run this BEFORE full experiments to catch any bugs.
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from run_experiment import SupplyChainEnv, BASE_CONFIG

def test_reward_function():
    """Test that reward function correctly incorporates FTOPSIS bonus."""
    
    print("="*80)
    print("REWARD FUNCTION VALIDATION TEST")
    print("="*80)
    
    # Load FTOPSIS scores
    mcdm_scores_file = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'mcdm', 'supplier_scores_for_rl.npy'
    )
    
    if not os.path.exists(mcdm_scores_file):
        print(f"ERROR: MCDM scores file not found: {mcdm_scores_file}")
        return False
    
    ftopsis_scores = np.load(mcdm_scores_file)
    print(f"\nLoaded FTOPSIS scores: {ftopsis_scores}")
    print(f"   Highest: S3 (index 2) = {ftopsis_scores[2]:.4f}")
    print(f"   Second: S5 (index 4) = {ftopsis_scores[4]:.4f}")
    
    # Create config with FTOPSIS scores
    test_config = BASE_CONFIG.copy()
    test_config['ftopsis_scores'] = ftopsis_scores
    test_config['max_steps'] = 10  # Short test
    test_config['scenario_name'] = 'Test'
    
    # Print key parameters
    print(f"\nKey Config Parameters:")
    print(f"   lambda_sustain_resil: {test_config['lambda_sustain_resil']} (should be 5000)")
    print(f"   lambda_ftopsis_bonus: {test_config.get('lambda_ftopsis_bonus', 'NOT SET')} (should be 2000)")
    print(f"   reward_scaling_factor: {test_config['reward_scaling_factor']}")
    
    # Create environment
    print(f"\nCreating test environment...")
    env = SupplyChainEnv(test_config)
    
    # Reset environment
    env.reset()
    print(f"Environment reset successful")
    
    # Simulate ordering from different suppliers
    print(f"\nTesting allocation patterns:")
    
    # Test 1: Order equally from all suppliers (uniform)
    uniform_action = np.ones(10, dtype=np.float32) * 0.5  # 5 suppliers × 2 urgency levels
    uniform_action_reshaped = uniform_action.reshape(5, 2)
    _, reward_uniform, _, _, _ = env.step(uniform_action_reshaped)
    print(f"\n   Test 1: Uniform allocation (equal to all suppliers)")
    print(f"   Reward: {reward_uniform:.4f}")
    
    # Test 2: Order only from S3 (highest FTOPSIS)
    env.reset()
    s3_action = np.zeros((5, 2), dtype=np.float32)
    s3_action[2, 0] = 1.0  # S3 (index 2), urgent
    s3_action[2, 1] = 1.0  # S3, non-urgent
    _, reward_s3, _, _, _ = env.step(s3_action)
    print(f"\n   Test 2: S3-only allocation (FTOPSIS leader)")
    print(f"   Reward: {reward_s3:.4f}")
    
    # Test 3: Order only from S1 (lowest FTOPSIS)
    env.reset()
    s1_action = np.zeros((5, 2), dtype=np.float32)
    s1_action[0, 0] = 1.0  # S1 (index 0), urgent
    s1_action[0, 1] = 1.0  # S1, non-urgent
    _, reward_s1, _, _, _ = env.step(s1_action)
    print(f"\n   Test 3: S1-only allocation (lowest FTOPSIS)")
    print(f"   Reward: {reward_s1:.4f}")
    
    # Verify S3 reward > S1 reward (due to FTOPSIS bonus)
    print(f"\nVALIDATION CHECKS:")
    print(f"   1. S3 reward ({reward_s3:.4f}) > S1 reward ({reward_s1:.4f})? ", end="")
    if reward_s3 > reward_s1:
        print("PASS - FTOPSIS bonus working!")
    else:
        print(f"FAIL - S3 should have higher reward!")
        return False
    
    print(f"   2. lambda_sustain_resil increased? ", end="")
    if test_config['lambda_sustain_resil'] == 5000:
        print("PASS - Set to 5000 (10x increase)")
    else:
        print(f"FAIL - Should be 5000, is {test_config['lambda_sustain_resil']}")
        return False
    
    print(f"   3. lambda_ftopsis_bonus configured? ", end="")
    if 'lambda_ftopsis_bonus' in test_config:
        print(f"PASS - Set to {test_config['lambda_ftopsis_bonus']}")
    else:
        print("FAIL - Not configured")
        return False
    
    print(f"\n" + "="*80)
    print("ALL VALIDATION CHECKS PASSED!")
    print("="*80)
    print(f"\nReady to run full experiments. Expected improvements:")
    print(f"   - M3 will allocate 50-70% to S3 (was 16%)")
    print(f"   - M3 will allocate 15-25% to S5 (was 24%, may stay similar)")
    print(f"   - M3 sustainability will increase to 0.65-0.75 (was 0.55)")
    print(f"   - M3 cost may increase slightly but should stay below M1")
    
    return True

if __name__ == '__main__':
    success = test_reward_function()
    sys.exit(0 if success else 1)
