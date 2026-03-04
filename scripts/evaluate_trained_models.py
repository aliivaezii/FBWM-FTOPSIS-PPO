#!/usr/bin/env python3
"""
Evaluate Already-Trained Models (No Retraining)
================================================

This script loads trained models from best_model.zip and runs evaluation only.
Generates CSV files without retraining.

Author: Ali Vaezi, Erfan Rabbani, Giulia Bruno

Usage:
    python scripts/evaluate_trained_models.py
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import from run_experiment.py
from run_experiment import (
    SupplyChainEnv,
    UninformedWrapper,
    evaluate_policy,
    BASE_CONFIG,
    EVALUATION_EPISODES
)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def load_ftopsis_scores():
    """Load FTOPSIS scores from MCDM evaluation."""
    mcdm_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'mcdm', 'supplier_scores_for_rl.npy')
    ftopsis_scores = np.load(mcdm_file)
    print(f"Loaded FTOPSIS scores: {ftopsis_scores}")
    return ftopsis_scores


def build_scenario(base_cfg, scenario_name, ftopsis_scores):
    """Build a complete scenario config from BASE_CONFIG."""
    cfg = base_cfg.copy()
    cfg.update(base_cfg["scenarios"][scenario_name])
    cfg["scenario_name"] = scenario_name
    cfg["ftopsis_scores"] = ftopsis_scores
    return cfg


def evaluate_existing_model(model_path, scenario_config, model_short_name, scenario_name):
    """Load a trained model and evaluate it."""

    print(f"\n{'='*80}")
    print(f"Evaluating: {model_short_name} | {scenario_name}")
    print(f"{'='*80}")
    print(f"Model path: {model_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return None, None

    # Determine if this model uses priors or not
    uses_priors = "Priors" in model_short_name or "Full" in model_short_name

    # Create environment
    if uses_priors:
        env = DummyVecEnv([lambda: Monitor(SupplyChainEnv(scenario_config))])
    else:
        env = DummyVecEnv([lambda: Monitor(UninformedWrapper(SupplyChainEnv(scenario_config)))])

    # Load trained model
    print(f"Loading trained model...")
    try:
        model = PPO.load(model_path, env=env)
        print(f"Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Error loading model: {e}")
        return None, None

    # Evaluate
    print(f"Running evaluation ({EVALUATION_EPISODES} episodes)...")
    agg_results, detailed_log = evaluate_policy(
        model,
        scenario_config,
        n_episodes=EVALUATION_EPISODES,
        model_name=model_short_name
    )

    print(f"Evaluation complete.")
    print(f"   Total Cost: EUR {agg_results['total'][0]:,.0f} +/- EUR {agg_results['total'][1]:,.0f}")
    print(f"   Sustainability: {agg_results['sustainability_score'][0]:.3f} +/- {agg_results['sustainability_score'][1]:.3f}")

    return agg_results, detailed_log


def main():
    print("="*80)
    print("EVALUATING ALREADY-TRAINED MODELS")
    print("="*80)
    print("\nThis will load trained models and run evaluation only (no training).\n")

    # Get project root
    project_root = os.path.join(os.path.dirname(__file__), '..')
    project_root = os.path.abspath(project_root)

    # Load FTOPSIS scores and build scenario configs
    ftopsis_scores = load_ftopsis_scores()

    scenario_names = ["Stable_Operations", "High_Volatility", "Systemic_Shock"]
    scenarios = {}
    for sc_name in scenario_names:
        scenarios[sc_name] = build_scenario(BASE_CONFIG, sc_name, ftopsis_scores)

    # Model configurations
    models = {
        "M2": ("M2_Vanilla_PPO", "M2: Vanilla PPO"),
        "M3": ("M3_PPO_+_Priors", "M3: PPO + Priors"),
    }

    # Ask which models to evaluate
    print("Which models do you want to evaluate?")
    print("  1) M2 only")
    print("  2) M3 only")
    print("  3) All (M2, M3)")
    print()

    try:
        choice = input("Enter 1, 2, or 3: ").strip()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return

    if choice == "1":
        models_to_eval = {"M2": models["M2"]}
    elif choice == "2":
        models_to_eval = {"M3": models["M3"]}
    elif choice == "3":
        models_to_eval = models
    else:
        print("Invalid choice.")
        return

    # Evaluate models
    total = len(models_to_eval) * len(scenarios)
    completed = 0

    for model_short, (model_dir, model_display) in models_to_eval.items():
        for scenario_name, scenario_config in scenarios.items():
            completed += 1
            print(f"\n{'='*80}")
            print(f"[{completed}/{total}] Processing...")
            print(f"{'='*80}")

            # Path to trained model
            model_path = os.path.join(
                project_root,
                "results", "experiments",
                scenario_name,
                model_dir,
                "best_model.zip"
            )

            # Evaluate
            agg_results, detailed_log = evaluate_existing_model(
                model_path,
                scenario_config,
                model_display,
                scenario_name
            )

            if agg_results is None:
                print(f"Skipping {model_short} | {scenario_name}")
                continue

            # Save CSV
            output_dir = os.path.join(
                project_root,
                "results", "experiments",
                scenario_name,
                model_dir
            )
            os.makedirs(output_dir, exist_ok=True)

            csv_path = os.path.join(output_dir, "allocation_table.csv")
            detailed_df = pd.DataFrame(detailed_log)
            detailed_df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nEvaluated {completed} model/scenario combinations.")
    print(f"\nCSV files saved to: results/experiments/{{scenario}}/{{model}}/allocation_table.csv")

if __name__ == "__main__":
    main()
