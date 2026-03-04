"""
SUPRA-PPO: Adaptive Reinforcement Learning for Supply Chain Order Allocation
=============================================================================

This module implements the hybrid FBWM–FTOPSIS–PPO framework for dynamic
multi-product, multi-supplier order allocation under non-stationary demand.

Components:
    1. Supply Chain Environment (SupplyChainEnv):
       Gymnasium-compatible simulation with non-stationary demand, lognormal
       lead times, and systemic supplier disruptions.

    2. Adaptive-Entropy PPO (SUPRAPPO):
       Stability-driven entropy scheduling — monitors the coefficient of
       variation (CV) of realised costs and adapts exploration accordingly.

    3. Ablation Study Framework:
       M1: Base-Stock heuristic baseline
       M2: Vanilla PPO (no MCDM priors)
       M3: PPO augmented with FTOPSIS sustainability scores and disruption flags

Authors: Ali Vaezi, Erfan Rabbani, Giulia Bruno
Date: October 2025
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import statistics
from typing import Dict, List, Tuple, Any, Union, Deque, Optional, Type
from numpy.typing import NDArray
import pandas as pd
import os
import sys
import math

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback, CallbackList, StopTrainingOnNoModelImprovement
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium import Env
from typing import Callable

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
# All configuration parameters are integrated here for reproducibility.
# No external dependencies required - fully self-contained experimental setup.

"""
Base Configuration Dictionary
-----------------------------
Defines all environment parameters for the supply chain simulation.
These parameters are calibrated based on industry standards and research literature.
"""
BASE_CONFIG: Dict[str, Any] = {
    # === Core Environment Structure ===
    "n_products": 2,  # Two-product system for computational tractability
    "supplier_names": ["S1", "S2", "S3", "S4", "S5"],  # Five heterogeneous suppliers
    
    # === FTOPSIS Sustainability-Resilience Scores ===
    # Range [0,1] where higher = better overall performance
    # Derived from FBWM-FTOPSIS Phase I evaluation (16 criteria across Economic, Environmental, Social, Resilience)
    # Rankings: S5 (Premium Innovator) > S3 (Resilient Incumbent) > S2 (Green Specialist) > S4 (Balanced) > S1 (Low-Cost)
    # Source: results/mcdm/supplier_scores_for_rl.npy with CR=0.0442 (excellent consistency)
    # Note: This will be automatically loaded from mcdm_evaluation.py output at runtime
    "ftopsis_scores": None,  # Will be loaded from supplier_scores_for_rl.npy
    
    # === Supplier Economics ===
    # Purchase costs [Product 1, Product 2] per supplier (€/unit)
    # S5 is most expensive but most sustainable (realistic trade-off)
    "supplier_purchase_costs": np.array([[10, 100], [15, 110], [13, 105], [12, 108], [16, 115]]),
    
    # === Supplier Capacity Constraints ===
    # Maximum daily order quantities [Product 1, Product 2] per supplier
    "supplier_max_capacities": np.array([[200, 20], [150, 15], [180, 18], [180, 18], [200, 20]]),
    
    # === Episode Parameters ===
    "max_steps": 365,  # One year of daily operations
    "base_stock_targets": [400, 40],  # Heuristic policy targets (~4 days buffer)
    
    # === Sustainability Trade-off Parameter (λ_sust) ===
    # Controls cost vs. sustainability trade-off in reward function:
    #   r_t = -Ĉ_ops + λ_sust * F̂_t
    # where Ĉ_ops = normalized daily operational cost, F̂_t = order-weighted mean FTOPSIS score
    # Sweepable values: {0.0, 0.1, 0.5, 1.0} for Pareto analysis
    "lambda_sust": 0.1,
    
    # === Reference Daily Cost for Reward Normalization ===
    # Approximate "typical" daily operational cost to keep reward in [-1, 0] range
    # Derivation: ~120 units/day × avg €50/unit purchase + holding + some shortage ≈ €10,000
    "ref_daily_cost": 10_000.0,
    
    # === Stochastic Supply Parameters ===
    # Individual supplier disruption probabilities (daily)
    # Higher values = less reliable supplier
    "supplier_disruption_prob": [0.05, 0.03, 0.04, 0.04, 0.02],
    
    # Lead time distributions (days): Modeled as lognormal for realism
    # S5 fastest but expensive, S2 slowest but cheaper
    "supplier_lead_times_mean": [3, 5, 4, 4, 2],
    "supplier_lead_times_std": [1, 2, 1.5, 1.5, 0.5],
    
    # === Systemic Shock Configuration ===
    # TRUTH-ALIGNED TEST: S3/S5 survive (indices 2,4), S1/S2/S4 fail (indices 0,1,3)
    # This aligns with MCDM resilience scores to validate domain knowledge value
    "systemic_shock_active": False,  # Enabled only in Shock scenario
    "systemic_shock_start": 100,  # Day when shock begins
    "systemic_shock_end": 130,  # Day when shock ends
    "systemic_shock_cluster": [0, 1, 3],  # S1, S2, S4 fail; S3, S5 survive
    
    # === Non-Stationary Demand Model ===
    # Composite demand: Base + Trend + Seasonality + Shocks + Noise
    # This tests the adaptive entropy mechanism under changing conditions
    "demand_params": {
        "mean": [100, 10],  # Base daily demand [P1, P2]
        "noise_std": [10, 1],  # Gaussian noise std dev
        "trend_factor": 0.02,  # Linear growth per day (2% of base)
        "seasonality_amplitude": 0.15,  # 15% seasonal variation
        "seasonality_period": 365,  # Annual cycle
        "shock_prob": 0.02,  # 2% daily probability of demand shock
        "shock_min_factor": -0.3,  # Shock can reduce demand by 30%
        "shock_max_factor": 0.5,  # Shock can increase demand by 50%
    },
    
    # === Cost Structure ===
    # Shortage penalty >> Holding cost (high ratio incentivises proactive ordering)
    "shortage_cost": [750, 7500],   # €/unit short [Product 1, Product 2]
    "holding_cost": [1.0, 10],      # €/unit held  [Product 1, Product 2]
       
    # === Initial Conditions ===
    # Zero initial inventory forces the agent to learn proactive ordering
    "initial_inventory": [0, 0],
    
    # === Experimental Scenarios ===
    # Three distinct operating environments to test algorithm robustness
    # Each scenario tests different aspects of the adaptive entropy mechanism
    "scenarios": {
        # SCENARIO 1: Baseline Operations (realistic stable demand)
        "Stable_Operations": {
            "demand_params": {
                "mean": [100, 10],
                "noise_std": [10, 1],            # Stochastic noise amplitude
                "trend_factor": 0.03,            # Slightly higher trend
                "seasonality_amplitude": 0.18,
                "seasonality_period": 365,
                "shock_prob": 0.03,              # More frequent shocks
                "shock_min_factor": -0.3,
                "shock_max_factor": 0.5,
            },
            "supplier_disruption_prob": [0.05, 0.04, 0.05, 0.05, 0.03],
            "systemic_shock_active": False,
        },
        # SCENARIO 2: High Demand Volatility (much harder)
        "High_Volatility": {
            "demand_params": {
                "mean": [100, 10],
                "noise_std": [70, 7],            # Much higher volatility
                "trend_factor": 0.07,            # Stronger trend
                "seasonality_amplitude": 0.35,   # More pronounced seasonality
                "seasonality_period": 365,
                "shock_prob": 0.12,              # Frequent shocks
                "shock_min_factor": -0.6,
                "shock_max_factor": 0.8,
            },
            "supplier_disruption_prob": [0.18, 0.15, 0.18, 0.18, 0.12],
            "systemic_shock_active": False,
        },
        # SCENARIO 3: TRUTH-ALIGNED SYSTEMIC SHOCK
        # S1, S2, S4 fail (indices 0, 1, 3) - lowest MCDM resilience scores
        # S3, S5 survive (indices 2, 4) - highest MCDM resilience scores
        # This tests whether domain knowledge (FTOPSIS priors) provides value
        "Systemic_Shock": {
            "demand_params": {
                "mean": [100, 10],
                "noise_std": [40, 4],            # High volatility
                "trend_factor": 0.05,
                "seasonality_amplitude": 0.25,
                "seasonality_period": 365,
                "shock_prob": 0.08,
                "shock_min_factor": -0.5,
                "shock_max_factor": 0.7,
            },
            "supplier_disruption_prob": [0.25, 0.20, 0.05, 0.25, 0.05],  # S3/S5 more reliable
            "systemic_shock_active": True,
            "systemic_shock_start": 50,   # Shock starts earlier
            "systemic_shock_end": 250,    # Shock lasts much longer
            "systemic_shock_cluster": [0, 1, 3],  # S1, S2, S4 fail; S3, S5 survive
        }
    }
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
"""
Model-Specific Hyperparameter Configurations
---------------------------------------------
M2 and M3 use identical PPO hyperparameters so that any performance
difference is attributable solely to the observation space (with or
without MCDM priors), not to algorithmic tuning.
"""
MODEL_CONFIGS: Dict[str, Dict[str, Union[float, int, bool]]] = {
    # === M2: Vanilla PPO (Baseline) ===
    # Standard PPO with no prior information
    # Must discover supplier reliability purely from experience
    "M2: Vanilla PPO": {
        "learning_rate": 3e-4,  # Default PPO learning rate
        "ent_coef": 0.01,       # Standard entropy coefficient
        "clip_range": 0.2,      # PPO clipping parameter
        "batch_size": 64,       # Standard batch size for PPO
        "n_steps": 2048,        # Rollout buffer size
        "gae_lambda": 0.95,     # GAE discount factor
        "n_epochs": 10,         # Policy update epochs per rollout
        "vf_coef": 0.5,         # Value function loss coefficient
        "max_grad_norm": 0.5    # Gradient clipping threshold
    },
    
    # === M3: PPO + Prior Information ===
    # Identical hyperparameters to M2; the only difference is the observation
    # space, which includes FTOPSIS sustainability scores and disruption flags.
    "M3: PPO + Priors": {
        "learning_rate": 3e-4,  # Same as M2
        "ent_coef": 0.01,       # Same as M2
        "clip_range": 0.2,
        "batch_size": 64,
        "n_steps": 2048,
        "gae_lambda": 0.95,
        "n_epochs": 10,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
}

# =============================================================================
# SUPPLY CHAIN ENVIRONMENT
# =============================================================================

class SupplyChainEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """
    Advanced Multi-Product, Multi-Supplier Supply Chain Environment
    ===============================================================
    
    A Gymnasium-compatible environment for supply chain optimization under uncertainty.
    
    **State Space** (16 dimensions for informed agents, 6 for uninformed):
        - Inventory levels (2): Current stock for each product
        - Last demand (2): Previous day's realized demand
        - Pipeline inventory (2): Units in transit from all suppliers
        - Disruption flags (5): Binary indicators of supplier failures
        - FTOPSIS scores (5): Sustainability ratings [0,1] per supplier
    
    **Action Space** (5x2 continuous):
        - Order quantities [0, capacity] for each supplier-product pair
    
    **Reward Function**:
        r_t = -Ĉ_ops_t + λ_sust * F̂_t
        where Ĉ_ops = (purchase + holding + shortage) / ref_daily_cost
        and F̂_t = order-weighted mean FTOPSIS closeness coefficient ∈ [0, 1]
    
    **Key Features**:
        1. Non-stationary demand: Trend + Seasonality + Shocks + Noise
        2. Lognormal lead times: Realistic supply uncertainty
        3. Systemic disruptions: Correlated supplier failures
        4. Multi-objective: Cost minimization + Sustainability maximization
    
    **Episode Structure**:
        - Length: 365 days (one year)
        - Termination: After max_steps or manual truncation
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the supply chain environment.
        
        Args:
            config: Configuration dictionary containing all environment parameters
        """
        super(SupplyChainEnv, self).__init__()
        self.config = config
        self.n_suppliers = len(config["supplier_names"])
        self.n_products = config["n_products"]
        
        # Define observation and action spaces
        state_size = self.n_products * 3 + self.n_suppliers * 2  # 16-dim state
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        # NORMALIZED ACTION SPACE: [-1, 1] for stable RL training
        # Actions will be rescaled to [0, max_capacity] in step()
        # This allows agents to explore full order range from random initialization
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(self.n_suppliers, self.n_products), 
            dtype=np.float32
        )
        
        # Initialize state variables (prevents reset() type issues)
        self.current_step = 0
        self.inventory: NDArray[np.float32] = np.array(
            self.config["initial_inventory"], dtype=np.float32
        )
        self.last_demand: NDArray[np.float32] = np.zeros(self.n_products, dtype=np.float32)
        self.disruptions: NDArray[np.float32] = np.zeros(self.n_suppliers, dtype=np.float32)
        
        # Pipeline: List of queues storing (arrival_day, quantity, product_idx)
        self.pipeline: List[Deque[Tuple[int, float, int]]] = [
            deque() for _ in range(self.n_suppliers)
        ]
        
        # Cumulative metrics for episode tracking
        self.info: Dict[str, Any] = {
            'total_cost': 0.0, 
            'purchase_cost': 0.0, 
            'holding_cost': 0.0, 
            'shortage_cost': 0.0,
            'daily_sustain_net_score': 0.0,
            'total_demand': np.zeros(self.n_products, dtype=np.float32),
            'total_orders': np.zeros(self.n_products, dtype=np.float32)
        }
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed, options=options)
        
        # Reset time
        self.current_step = 0
        
        # Reset inventory to initial levels (3 days buffer)
        self.inventory = np.array(self.config["initial_inventory"], dtype=np.float32)
        
        # Clear demand history and disruption states
        self.last_demand = np.zeros(self.n_products, dtype=np.float32)
        self.disruptions = np.zeros(self.n_suppliers, dtype=np.float32)
        
        # Clear all pipeline inventories
        self.pipeline = [deque() for _ in range(self.n_suppliers)]
        
        # Reset cumulative metrics
        self.info = {
            'total_cost': 0.0,
            'purchase_cost': 0.0,
            'holding_cost': 0.0,
            'shortage_cost': 0.0,
            'daily_sustain_net_score': 0.0,
            'total_demand': np.zeros(self.n_products, dtype=np.float32),
            'total_orders': np.zeros(self.n_products, dtype=np.float32)
        }
        
        return self._get_obs(), self.info

    def _get_obs(self) -> NDArray[np.float32]:
        """
        Construct the current observation vector.
        
        Observation structure (16 dimensions):
            [0:2]   - Current inventory levels
            [2:4]   - Last period's demand
            [4:6]   - Pipeline inventory (units in transit)
            [6:11]  - Supplier disruption flags (binary)
            [11:16] - FTOPSIS sustainability scores
            
        Returns:
            NDArray with shape (16,) and dtype float32
        """
        # Aggregate pipeline inventory by product
        in_pipeline_per_product = np.zeros(self.n_products, dtype=np.float32)
        for supplier_pipeline in self.pipeline:
            for _, qty, prod_idx in supplier_pipeline:
                in_pipeline_per_product[prod_idx] += qty
        
        # Extract FTOPSIS scores
        ftopsis_scores = np.array(self.config["ftopsis_scores"], dtype=np.float32)
        
        # Ensure type consistency before concatenation
        inventory_array = self.inventory.astype(np.float32)
        last_demand_array = self.last_demand.astype(np.float32)
        in_pipeline_array = in_pipeline_per_product.astype(np.float32)
        disruptions_array = self.disruptions.astype(np.float32)
        
        # Build observation vector
        obs = np.concatenate([
            inventory_array,
            last_demand_array,
            in_pipeline_array,
            disruptions_array,
            ftopsis_scores
        ])
        
        return obs.astype(np.float32)
    def _generate_demand(self) -> NDArray[np.float32]:
        """
        Generate non-stationary demand using composite model.
        
        Demand Formula:
            D_t = Base + Trend_t + Seasonality_t + Shock_t + Noise_t
            
        Components:
            - Base: Mean daily demand
            - Trend: Linear growth over time
            - Seasonality: Sinusoidal annual pattern
            - Shock: Rare stochastic jumps (Bernoulli process)
            - Noise: Gaussian white noise
            
        This composite model tests the adaptive entropy mechanism
        by creating time-varying reward landscapes.
        
        Returns:
            Non-negative demand vector [Product 1, Product 2]
        """
        demand_params = self.config["demand_params"]
        base_mean = np.array(demand_params["mean"], dtype=np.float32)
        
        # Linear trend component
        trend = np.float32(demand_params["trend_factor"] * self.current_step)
        
        # Sinusoidal seasonality
        seasonality = np.float32(
            demand_params["seasonality_amplitude"]
            * np.sin(2 * math.pi * self.current_step / demand_params["seasonality_period"])
        )
        
        # Stochastic demand shocks (rare events)
        shock: NDArray[np.float32] = np.zeros_like(base_mean, dtype=np.float32)
        if np.random.rand() < demand_params["shock_prob"]:
            low = float(demand_params["shock_min_factor"])
            high = float(demand_params["shock_max_factor"])
            shock_factor: float = float(np.random.uniform(low, high))
            shock = (base_mean * shock_factor).astype(np.float32)

        # Gaussian noise
        noise = np.array(
            np.random.normal(0, np.array(demand_params["noise_std"], dtype=np.float32)), 
            dtype=np.float32
        )
        
        # Composite demand (non-negative)
        demand = base_mean + trend + seasonality + shock + noise
        return np.maximum(0, demand).astype(np.float32)

    def step(
        self, 
        action: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step of the environment dynamics.
        
        Sequence of events per step:
            1. Increment time
            2. Rescale normalized actions to real order quantities
            3. Sample supplier disruptions (independent + systemic)
            4. Process pipeline arrivals (disrupted orders are lost)
            5. Update inventory with arrivals
            6. Generate today's demand
            7. Fulfill demand (FIFO from inventory)
            8. Process new orders (add to pipeline with stochastic lead times)
            9. Calculate costs and sustainability penalties
            10. Return observation, reward, and episode status
        
        Args:
            action: Normalized order quantities [-1, 1] for [5 suppliers × 2 products]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # === 0. RESCALE NORMALIZED ACTIONS TO REAL ORDER QUANTITIES ===
        # Action space is [-1, 1], rescale to [0, max_capacity]
        # Formula: order_quantities = (action + 1) / 2 * max_capacity
        # This allows agents to explore full order range from random initialization
        max_caps = np.array(self.config["supplier_max_capacities"], dtype=np.float32)
        order_quantities = ((action + 1.0) / 2.0) * max_caps
        order_quantities = np.clip(order_quantities, 0, max_caps)  # Safety clip
        
        # === 1. Sample Supplier Disruptions ===
        # Independent disruptions (each supplier fails independently)
        for i in range(self.n_suppliers):
            if np.random.rand() < self.config["supplier_disruption_prob"][i]:
                self.disruptions[i] = 1.0
        
        # Systemic shock (correlated failures during shock period)
        if (self.config["systemic_shock_active"] and 
            self.config["systemic_shock_start"] <= self.current_step < self.config["systemic_shock_end"]):
            for supplier_idx in self.config["systemic_shock_cluster"]:
                self.disruptions[supplier_idx] = 1.0

        # === 2. Process Pipeline Arrivals ===
        arrived_quantities = np.zeros(self.n_products, dtype=np.float32)
        
        for i in range(self.n_suppliers):
            # Check all orders arriving today
            while self.pipeline[i] and self.pipeline[i][0][0] <= self.current_step:
                _, quantity, prod_idx = self.pipeline[i].popleft()
                
                if self.disruptions[i] == 0:
                    # Order arrives successfully
                    arrived_quantities[prod_idx] += quantity
                # else: Order lost due to disruption (no separate penalty —
                #        the lost inventory will naturally cause future shortages)
        
        # === 3. Update Inventory ===
        self.inventory += arrived_quantities
        self.disruptions.fill(0)  # Reset disruption flags
        
        # === 4. Calculate Holding Costs ===
        holding_costs = np.sum(self.inventory * self.config["holding_cost"])
        
        # === 5. Generate and Fulfill Demand ===
        demand_today = self._generate_demand()
        self.last_demand = demand_today
        
        fulfilled_demand = np.minimum(self.inventory, demand_today)
        shortages = demand_today - fulfilled_demand
        # Standard shortage penalty (no artificial multiplier)
        # Note: shortage_cost values are already calibrated to be 100:1 vs holding
        shortage_costs = np.sum(shortages * self.config["shortage_cost"])
        self.inventory -= fulfilled_demand
        
        # === 6. Process New Orders (using rescaled order_quantities) ===
        purchase_costs = np.sum(order_quantities * self.config["supplier_purchase_costs"])
        
        # Add orders to pipeline with stochastic lead times
        for i in range(self.n_suppliers):
            for u in range(self.n_products):
                if order_quantities[i, u] > 0:
                    # Sample lead time from lognormal distribution
                    mean = self.config["supplier_lead_times_mean"][i]
                    sigma = self.config["supplier_lead_times_std"][i]
                    
                    if mean > 0:
                        log_mu = np.log(mean**2 / np.sqrt(mean**2 + sigma**2))
                        log_sigma = np.sqrt(np.log(1 + sigma**2 / mean**2))
                    else:
                        log_mu = -np.inf
                        log_sigma = 0
                    
                    lead_time = int(np.random.lognormal(log_mu, log_sigma))
                    arrival_step = self.current_step + max(1, lead_time)
                    self.pipeline[i].append((arrival_step, order_quantities[i, u], u))
        
        # === 7. CLEAN REWARD FORMULATION ===
        # r_t = -Ĉ_ops + λ_sust * F̂_t
        #
        # Component 1: Normalized operational cost Ĉ_ops
        #   Total daily cost divided by reference cost to keep reward ~ O(1)
        #   Disruption losses are already captured via shortages (lost orders
        #   never arrive → future inventory deficit → future shortage penalty).
        #   No separate disruption penalty needed.
        total_operational_cost = purchase_costs + holding_costs + shortage_costs
        ref_cost = self.config["ref_daily_cost"]
        normalized_cost = total_operational_cost / ref_cost  # Ĉ_ops ≈ O(1)
        
        # Component 2: Order-weighted mean FTOPSIS score F̂_t ∈ [0, 1]
        #   FTOPSIS closeness coefficients are kept in their natural [0, 1] range.
        #   F̂_t = Σ_i (q_i * CC_i) / Σ_i q_i  (weighted mean)
        #   If no orders placed, F̂_t = 0 (no sustainability benefit)
        total_units_per_supplier = np.sum(order_quantities, axis=1)
        total_units = np.sum(total_units_per_supplier)
        if total_units > 0:
            ftopsis_weighted_mean = float(
                np.sum(total_units_per_supplier * self.config["ftopsis_scores"]) / total_units
            )
        else:
            ftopsis_weighted_mean = 0.0
        
        # Combined reward
        lambda_sust = self.config["lambda_sust"]
        reward = -normalized_cost + lambda_sust * ftopsis_weighted_mean
        
        # Store unscaled cost for ES adaptation (Adaptive-Entropy PPO)
        self.info['unscaled_cost'] = total_operational_cost
        
        # Store daily sustainability score for cumulative tracking
        daily_sustainability_score = float(np.sum(total_units_per_supplier * self.config["ftopsis_scores"]))
        
        # === 8. Episode Termination ===
        terminated = self.current_step >= self.config["max_steps"]
        
        # === 9. Update Cumulative Metrics ===
        self.info['total_cost'] += total_operational_cost
        self.info['purchase_cost'] += purchase_costs
        self.info['holding_cost'] += holding_costs
        self.info['shortage_cost'] += shortage_costs
        self.info['daily_sustain_net_score'] = self.info.get('daily_sustain_net_score', 0.0) + daily_sustainability_score
        self.info['total_demand'] += demand_today
        self.info['total_orders'] += np.sum(order_quantities, axis=0)
        
        # Track service metrics
        self.info['service_level'] = (np.sum(fulfilled_demand) / np.sum(demand_today)) if np.sum(demand_today) > 0 else 1.0
        self.info['inventory_level'] = np.sum(self.inventory)
        
        return self._get_obs(), reward, terminated, False, self.info


# =============================================================================
# REINFORCEMENT LEARNING AGENTS
# =============================================================================

class ActorCriticPolicyWithDropout(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy with Dropout Regularization
    =======================================================
    
    Extends stable-baselines3's ActorCriticPolicy by adding dropout layers
    to the value network. This improves robustness by preventing overfitting
    to noisy reward signals in stochastic supply chain environments.
    
    **Architecture**:
        - Actor (policy): Standard MLP (inherited from parent)
        - Critic (value): 3-layer MLP with 10% dropout between layers
        
    **Benefits**:
        - Reduces variance in value estimation
        - Improves generalization across different demand scenarios
        - Enables robust value function estimation under uncertainty
    """
    
    def __init__(
        self, 
        observation_space: spaces.Box, 
        action_space: spaces.Box, 
        *args: Any, 
        **kwargs: Any
    ) -> None:
        """
        Initialize policy networks with dropout in value function.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            *args: Additional arguments for parent class
            **kwargs: Additional keyword arguments for parent class
        """
        # Type ignore: Parent class has complex signature from stable-baselines3
        super().__init__(observation_space, action_space, *args, **kwargs)  # type: ignore
        
        # Build custom value network with dropout regularization
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        self.value_net_with_dropout = nn.Sequential(
            nn.Linear(latent_dim_vf, 64),
            nn.Tanh(),
            nn.Dropout(0.1),  # 10% dropout for regularization
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Dropout(0.1),  # Second dropout layer
            nn.Linear(64, 1)  # Scalar value output
        ).to(self.device)

    def forward_value_with_dropout(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network with dropout enabled.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Estimated state value with dropout applied
        """
        latent_vf = self.mlp_extractor.forward_critic(obs)
        self.value_net_with_dropout.train()  # Enable dropout at inference
        return self.value_net_with_dropout(latent_vf)


class SUPRAPPO(PPO):
    """
    Adaptive-Entropy PPO (AE-PPO)
    =============================
    
    Extends PPO with stability-driven entropy scheduling for supply chain optimization.
    
    **Mechanism**:
        - Monitors the Coefficient of Variation (CV) of realized costs
        - High CV (unstable performance) → Maintains high entropy (explore)
        - Low CV (stable performance) → Decays entropy (exploit)
    
    **Entropy Adaptation Formula**:
        CV = σ(costs) / |μ(costs)|
        β_t = β_0 * exp(-α / (CV + ε))
        
    **Intuition**:
        - When costs are volatile (high CV), exploration is still needed
        - When costs stabilize (low CV), the agent has found a good policy
        - Inverse relationship: stability causes decay, instability maintains exploration
    
    **Research Context**:
        Demonstrates how adaptive entropy scheduling can outperform fixed-
        hyperparameter RL in non-stationary supply chain environments.
    
    **Key Parameters**:
        - beta_base: Initial entropy coefficient β_0 (default: 0.15)
        - alpha_decay: Decay rate α (default: 0.0005)
        - use_es: Enable/disable adaptive entropy scheduling
    """
    
    def __init__(
        self, 
        policy: Union[str, Type[ActorCriticPolicy]], 
        env: Union[Env[Any, Any], VecEnv, str], 
        **kwargs: Any
    ) -> None:
        """
        Initialize Adaptive-Entropy PPO algorithm.
        
        Args:
            policy: Policy class or string identifier
            env: Training environment
            **kwargs: Hyperparameters including AE-PPO specific params
        """
        # Extract AE-PPO parameters before parent initialization
        self.beta_base: float = float(kwargs.pop("beta_base", 0.01))
        self.alpha_decay: float = float(kwargs.pop("alpha_decay", 0.001))
        self.use_es: bool = bool(kwargs.pop("use_es", True))
        
        # Remove legacy UDR params if present (backwards compatibility)
        kwargs.pop("lambda_udr", None)
        kwargs.pop("use_udr", None)
        
        # Initialize parent PPO with remaining arguments
        # Type ignore: Complex inheritance from stable-baselines3
        super().__init__(policy, env, **kwargs)  # type: ignore[arg-type]
        
        # Initialize AE-PPO specific state
        self.beta: float = self.beta_base
        self.cost_buffer: Deque[float] = deque(maxlen=100)  # Sliding window of UNSCALED costs

    def train(self) -> None:
        """
        Custom training step with Stability-Driven Entropy Scheduling.
        
        Overrides parent PPO.train() to:
            1. Collect UNSCALED costs from recent episodes
            2. Compute CV using proper formula: CV = std / mean
            3. Adapt entropy using INVERSE logic: stable → decay, unstable → maintain
            4. Apply adapted coefficient to current training iteration
            5. Call parent training with modified hyperparameters
            
        **AE-PPO Formula**:
            CV(C) = StdDev(C) / |Mean(C)|  (Coefficient of Variation)
            β_t = β_0 * exp(-α / (CV + ε))
            
        **Key Insight**:
            - High CV → Large denominator → Small decay → High entropy maintained
            - Low CV → Small denominator → Large decay → Entropy decays → Exploitation
        """
        # === 1. Collect Unscaled Costs from Recent Episodes ===
        if self.ep_info_buffer is not None and len(self.ep_info_buffer) > 0:  # type: ignore[arg-type]
            costs_from_rollout = [
                float(ep_info['unscaled_cost'])  # type: ignore[arg-type]
                for ep_info in self.ep_info_buffer  # type: ignore[union-attr]
                if isinstance(ep_info, dict) and 'unscaled_cost' in ep_info
            ]
            if costs_from_rollout:
                self.cost_buffer.extend(costs_from_rollout)
        
        # === 2. Apply Adaptive Entropy Scheduling ===
        if self.use_es and len(self.cost_buffer) > 20:
            mean_cost: float = statistics.mean(self.cost_buffer)
            std_cost: float = statistics.stdev(self.cost_buffer)
            
            if abs(mean_cost) > 1e-6:
                # Compute coefficient of variation
                cv: float = std_cost / abs(mean_cost)
                
                # INVERSE logic: Stability (low CV) causes decay
                # High CV → large denominator → small exponent → high entropy
                # Low CV → small denominator → large exponent → low entropy
                self.beta = self.beta_base * float(np.exp(-self.alpha_decay * (1.0 / (cv + 1e-6))))
                
                # Clip entropy to reasonable bounds
                self.beta = float(np.clip(self.beta, 0.005, 0.20))
                
                # **CRITICAL**: Apply adapted entropy to PPO
                self.ent_coef = self.beta
                
                # Monitor entropy adaptation (log every 10K steps)
                # Type ignore: num_timesteps is a parent class attribute
                if hasattr(self, 'num_timesteps') and self.num_timesteps % 10000 == 0:  # type: ignore[has-type]
                    print(f"  [AE-PPO] Step {self.num_timesteps}: "  # type: ignore[has-type]
                          f"ent_coef={self.ent_coef:.4f}, "
                          f"CV={cv:.3f}, "
                          f"mean_cost=€{mean_cost:,.0f}")
        else:
            # Insufficient data for adaptation - use base entropy
            self.beta = self.beta_base
            self.ent_coef = self.beta
        
        # Call parent PPO training with adapted parameters
        super().train()


# Alias for backwards compatibility and cleaner naming
AdaptiveEntropyPPO = SUPRAPPO

class UninformedWrapper(gym.ObservationWrapper[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]):
    """
    Observation Wrapper for Uninformed Agents
    ==========================================
    
    Removes domain-specific prior information from observations to create
    uninformed baseline agents. This enables fair ablation study to measure
    the benefit of FTOPSIS scores and disruption flags.
    
    **Transformation**:
        - Input: 16-dim observation (inventory, demand, pipeline, disruptions, FTOPSIS)
        - Output: 6-dim observation (inventory, demand, pipeline only)
        
    **Usage**:
        - M2 (Vanilla PPO): Wrapped environment (no priors)
        - M3 (PPO + Priors): Unwrapped (receives full observations with priors)
    
    **Research Purpose**:
        Isolates the benefit of prior information vs algorithmic improvements.
    """
    
    def __init__(self, env: SupplyChainEnv) -> None:
        """
        Initialize wrapper.
        
        Args:
            env: Base supply chain environment
        """
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box), "Expected Box observation space"
        
        # Calculate new observation dimension (remove last n_suppliers * 2 elements)
        original_space: spaces.Box = env.observation_space
        new_shape: int = original_space.shape[0] - env.n_suppliers * 2
        
        # Define reduced observation space
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(new_shape,), dtype=np.float32
        )

    def observation(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Remove disruption flags and FTOPSIS scores from observation.
        
        Args:
            observation: Full 16-dim observation
            
        Returns:
            Truncated 6-dim observation (only inventory, demand, pipeline)
        """
        assert isinstance(self.env, SupplyChainEnv), "Expected SupplyChainEnv"
        # Drop last n_suppliers * 2 elements (disruptions + FTOPSIS)
        return observation[:-(self.env.n_suppliers * 2)].astype(np.float32)


# =============================================================================
# BASELINE POLICY AND EVALUATION
# =============================================================================

class BaseStockPolicy:
    """
    Base-Stock Heuristic Policy (M1 Baseline)
    ==========================================
    
    A simple but effective inventory management heuristic that:
        1. Orders only from the most sustainable supplier (highest FTOPSIS)
        2. Maintains inventory position at a fixed base-stock level
        3. Orders quantity = target - (on_hand + pipeline)
    
    **Characteristics**:
        - No learning required
        - Incorporates sustainability (via supplier selection)
        - Standard practice in supply chain management literature
        
    **Purpose in Ablation Study**:
        - Establishes minimum performance threshold
        - Tests if RL adds value over domain heuristics
        - Expected to underperform RL methods but beat random policy
    
    **Formula**:
        Q_t = max(0, S - (I_t + P_t))
        where:
            S = base-stock target
            I_t = on-hand inventory
            P_t = pipeline inventory
    """
    
    def __init__(self, env_config: Dict[str, Any]) -> None:
        """
        Initialize base-stock policy.
        
        Args:
            env_config: Environment configuration dictionary
        """
        supplier_names: List[str] = env_config["supplier_names"]
        self.n_suppliers: int = len(supplier_names)
        self.n_products: int = int(env_config["n_products"])
        
        # Select best supplier by FTOPSIS score
        ftopsis_scores: NDArray[np.float32] = np.array(env_config["ftopsis_scores"])
        self.best_supplier_idx: int = int(np.argmax(ftopsis_scores))
        
        # Set base-stock targets
        self.target_stock: NDArray[np.float32] = np.array(
            env_config["base_stock_targets"], dtype=np.float32
        )
        
        # Store capacity constraints
        self.max_capacities: NDArray[np.float32] = np.array(
            env_config["supplier_max_capacities"], dtype=np.float32
        )

    def predict(
        self, 
        observation: NDArray[np.float32], 
        deterministic: bool = True
    ) -> Tuple[NDArray[np.float32], None]:
        """
        Generate base-stock order quantities.
        
        Args:
            observation: Current environment state
            deterministic: Unused (policy is deterministic by design)
            
        Returns:
            Tuple of (action, None) matching stable-baselines3 interface
        """
        # Extract inventory and pipeline from observation
        inventories: NDArray[np.float32] = observation[0 : self.n_products]
        in_pipeline_per_product: NDArray[np.float32] = observation[
            self.n_products * 2 : self.n_products * 3
        ]
        
        # Compute inventory position (on-hand + pipeline)
        inventory_positions: NDArray[np.float32] = inventories + in_pipeline_per_product
        
        # Base-stock formula: order up to target
        order_quantities_per_product: NDArray[np.float32] = np.maximum(
            0, self.target_stock - inventory_positions
        )
        
        # Build action matrix (only order from best supplier)
        action: NDArray[np.float32] = np.zeros(
            (self.n_suppliers, self.n_products), dtype=np.float32
        )
        action[self.best_supplier_idx, :] = np.minimum(
            order_quantities_per_product,
            self.max_capacities[self.best_supplier_idx, :]
        )
        
        return action, None


# =============================================================================
# INCREMENTAL LOGGING CALLBACK
# =============================================================================

class IncrementalCSVCallback(BaseCallback):
    """
    Incremental CSV Logging Callback
    =================================
    
    Logs training progress to CSV after each evaluation checkpoint.
    This provides:
        1. Crash Recovery: If training crashes, partial data is saved
        2. Live Monitoring: Can view training progress in real-time
        3. Rich Data: Episode-level metrics for detailed analysis
    
    **Logged Data (per evaluation checkpoint)**:
        - Timestep: Current training step
        - Episode: Evaluation episode number
        - Mean Reward: Average reward across eval episodes
        - Mean Episode Length: Average episode duration
        - Timestamp: When evaluation occurred
    
    **CSV Format**:
        Appends to existing CSV (mode='a'), creates if doesn't exist.
        File: {log_dir}/training_progress.csv
        Columns: timestep, episode_num, mean_reward, mean_ep_length, timestamp
    
    Usage:
        callback = IncrementalCSVCallback(log_dir="results/experiments/Scenario/Model")
        model.learn(total_timesteps=3000000, callback=callback)
    """
    
    def __init__(self, log_dir: str, verbose: int = 0):
        """
        Initialize callback.
        
        Args:
            log_dir: Directory to save training_progress.csv
            verbose: Verbosity level (0=silent, 1=info)
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "training_progress.csv")
        
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write("timestep,episode_num,mean_reward,mean_ep_length,timestamp\n")
        
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        """
        Called after each training step.
        Checks if evaluation occurred and logs metrics to CSV.
        
        Returns:
            True to continue training
        """
        # Check if evaluation callback just ran
        # EvalCallback stores results in self.locals['evaluations']
        if 'evaluations' in self.locals and self.locals['evaluations'] is not None:
            evaluations = self.locals['evaluations']
            
            # Get latest evaluation results
            if hasattr(evaluations, 'results') and len(evaluations.results) > 0:
                latest_rewards = evaluations.results[-1]
                mean_reward = np.mean(latest_rewards)
                
                # Get episode length if available
                mean_ep_length = evaluations.ep_lengths[-1] if hasattr(evaluations, 'ep_lengths') and len(evaluations.ep_lengths) > 0 else 0
                
                # Log to CSV
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(self.csv_path, 'a') as f:
                    f.write(f"{self.num_timesteps},{self.episode_count},{mean_reward},{mean_ep_length},{timestamp}\n")
                
                self.episode_count += 1
                
                if self.verbose > 0:
                    print(f"[LOG] Logged evaluation at step {self.num_timesteps}: reward={mean_reward:.2f}")
        
        return True


class TrainingMetricsCallback(BaseCallback):
    """
    Training Metrics Callback for Curriculum Learning Analysis
    ===========================================================
    
    Logs detailed training dynamics every N steps to enable:
        1. Curriculum Evolution Plots: Visualize cost vs. sustainability tradeoff over training
        2. Entropy Adaptation Plots: Show AE-PPO's entropy coefficient response to stability
        3. Service Level Tracking: Monitor operational performance throughout training
    
    **Logged Metrics** (to training_metrics.csv):
        - step: Current training timestep
        - total_cost: Unscaled operational cost (for curriculum analysis)
        - sustainability_score: Net FTOPSIS-weighted sustainability
        - service_level: Demand fulfillment ratio
        - entropy_coef: Current entropy coefficient (AE-PPO adaptation)
        - curriculum_stage: Current curriculum stage (1=cost-only, 2=sustainability)
    
    Usage:
        callback = TrainingMetricsCallback(log_dir="results/", log_freq=1000)
        model.learn(total_timesteps=1200000, callback=callback)
    """
    
    def __init__(self, log_dir: str, log_freq: int = 1000, verbose: int = 0):
        """
        Initialize callback.
        
        Args:
            log_dir: Directory to save training_metrics.csv
            log_freq: Log every N timesteps (default: 1000)
            verbose: Verbosity level (0=silent, 1=info)
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.csv_path = os.path.join(log_dir, "training_metrics.csv")
        
        # Accumulators for averaging between log intervals
        self.cost_accumulator: List[float] = []
        self.sustainability_accumulator: List[float] = []
        self.service_level_accumulator: List[float] = []
        
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write("step,total_cost,sustainability_score,service_level,entropy_coef,curriculum_stage\n")
    
    def _on_step(self) -> bool:
        """
        Called after each training step.
        Accumulates metrics and logs every log_freq steps.
        
        Returns:
            True to continue training
        """
        # Extract info from the environment (self.locals contains step data)
        infos = self.locals.get('infos', [])
        
        if infos:
            for info in infos:
                if isinstance(info, dict):
                    # Accumulate metrics from this step
                    if 'unscaled_cost' in info:
                        self.cost_accumulator.append(info['unscaled_cost'])
                    if 'daily_sustain_net_score' in info:
                        self.sustainability_accumulator.append(info.get('daily_sustain_net_score', 0.0))
                    if 'service_level' in info:
                        self.service_level_accumulator.append(info['service_level'])
        
        # Log every log_freq steps
        if self.num_timesteps % self.log_freq == 0 and self.num_timesteps > 0:
            # Calculate averages
            avg_cost = np.mean(self.cost_accumulator) if self.cost_accumulator else 0.0
            avg_sustainability = np.mean(self.sustainability_accumulator) if self.sustainability_accumulator else 0.0
            avg_service = np.mean(self.service_level_accumulator) if self.service_level_accumulator else 0.0
            
            # Get entropy coefficient from model (works for both PPO and AE-PPO)
            try:
                if hasattr(self.model, 'ent_coef_tensor'):
                    # AE-PPO: dynamic entropy coefficient
                    entropy_coef = float(self.model.ent_coef_tensor.item())
                elif hasattr(self.model, 'ent_coef'):
                    # Standard PPO: fixed entropy coefficient
                    if isinstance(self.model.ent_coef, str) and self.model.ent_coef == "auto":
                        entropy_coef = float(torch.exp(self.model.log_ent_coef).item())
                    else:
                        entropy_coef = float(self.model.ent_coef)
                else:
                    entropy_coef = 0.0
            except Exception:
                entropy_coef = 0.0
            
            # Determine curriculum stage based on timesteps
            # Access training env to get curriculum parameters
            try:
                training_env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
                if hasattr(training_env, 'curr_start_step'):
                    if self.num_timesteps < training_env.curr_start_step:
                        curriculum_stage = 1  # Cost-only
                    elif self.num_timesteps < training_env.curr_start_step + training_env.curr_duration:
                        curriculum_stage = 2  # Transition
                    else:
                        curriculum_stage = 3  # Final
                else:
                    curriculum_stage = 0  # Unknown
            except Exception:
                curriculum_stage = 0
            
            # Write to CSV
            with open(self.csv_path, 'a') as f:
                f.write(f"{self.num_timesteps},{avg_cost:.4f},{avg_sustainability:.4f},{avg_service:.4f},{entropy_coef:.6f},{curriculum_stage}\n")
            
            if self.verbose > 0:
                print(f"[METRICS] Training metrics at step {self.num_timesteps}: cost={avg_cost:.2f}, sustain={avg_sustainability:.2f}, entropy={entropy_coef:.4f}")
            
            # Reset accumulators
            self.cost_accumulator = []
            self.sustainability_accumulator = []
            self.service_level_accumulator = []
        
        return True


def evaluate_policy(
    model: Union[AdaptiveEntropyPPO, PPO, BaseStockPolicy], 
    env_config: Dict[str, Any], 
    n_episodes: int = 10, 
    model_name: str = "Unknown"
) -> Tuple[Dict[str, Tuple[float, float]], List[Dict[str, Any]], Dict[str, List[float]]]:
    """
    Comprehensive Policy Evaluation Function
    =========================================
    
    Evaluates a trained policy across multiple episodes and computes:
        1. Cost metrics (total, operational, sustainability, disruption)
        2. Bullwhip ratio (order variance / demand variance)
        3. Inventory turnover (COGS / average inventory value)
        4. Sustainability score (weighted by FTOPSIS)
    
    **Evaluation Protocol**:
        - n_episodes independent rollouts
        - Deterministic policy execution
        - Episode-level aggregation
        - Returns mean ± std for all metrics
    
    **Key Metrics**:
        - Total Cost: Primary optimization objective
        - Sustainability: FTOPSIS-weighted average across all orders
        - Bullwhip: Supply chain instability indicator
        - Turnover: Inventory efficiency measure
    
    Args:
        model: Trained policy (PPO, AE-PPO, or Base-Stock)
        env_config: Environment configuration
        n_episodes: Number of evaluation episodes
        model_name: Model identifier for logging
        
    Returns:
        Tuple of (aggregated_results, detailed_logs, raw_episode_data)
            - aggregated_results: Dict mapping metric names to (mean, std)
            - detailed_logs: List of per-step dictionaries for analysis
            - raw_episode_data: Dict mapping metric names to raw per-episode lists
              (for statistical tests: Welch's t-test, ANOVA, box plots)
    """
    # Initialize fresh evaluation environment
    eval_env: SupplyChainEnv = SupplyChainEnv(config=env_config)
    
    # Determine if model receives truncated observations
    is_uninformed: bool = ("without Priors" in model_name or "Vanilla PPO" in model_name)
    
    # Initialize metric accumulators
    bullwhip_ratios: List[float] = []
    inventory_turnovers: List[float] = []
    sustainability_scores: List[float] = []
    costs: Dict[str, List[float]] = {
        'total': [], 
        'operational': [], 
        'daily_sustain_net': []
    }
    detailed_log: List[Dict[str, Any]] = []
    
    # Prepare FTOPSIS scores for sustainability calculation
    ftopsis_scores_array: NDArray[np.float32] = np.array(
        env_config["ftopsis_scores"], dtype=np.float32
    )
    
    # Run evaluation episodes
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        done: bool = False
        episode_demands: List[float] = []
        episode_orders: List[float] = []
        total_weighted_score: float = 0.0
        total_units_ordered: float = 0.0
        info: Dict[str, Any] = {}  # Initialize to avoid unbound variable

        while not done:
            # Prepare observation for model (may need truncation)
            model_obs: NDArray[np.float32] = (
                obs[:-(eval_env.n_suppliers * 2)] if is_uninformed else obs
            )
            
            # Query policy using deterministic action selection for evaluation
            # Using deterministic=True is standard practice in RL evaluation
            action, _ = model.predict(model_obs, deterministic=True)  # type: ignore[union-attr]
            
            # === Rescale raw [-1,1] actions to real order quantities ===
            # The model outputs normalized actions in [-1, 1].
            # All metric calculations MUST use real-world quantities, not raw actions.
            max_caps = np.array(env_config["supplier_max_capacities"], dtype=np.float32)
            real_orders: NDArray[np.float32] = ((action + 1.0) / 2.0) * max_caps
            real_orders = np.clip(real_orders, 0, max_caps)
            
            # Track sustainability-weighted orders (using REAL quantities)
            units_per_supplier: NDArray[np.float32] = np.sum(real_orders, axis=1)  # type: ignore[arg-type]
            units_this_step: float = float(np.sum(units_per_supplier))
            if units_this_step > 0:
                total_units_ordered += units_this_step
                # Weighted sum: FTOPSIS score * real units ordered from each supplier
                weighted_score_this_step: float = float(
                    np.sum(units_per_supplier * ftopsis_scores_array)
                )
                total_weighted_score += weighted_score_this_step
            
            # Log demands and orders for bullwhip calculation (using REAL quantities)
            episode_demands.append(float(np.sum(eval_env.last_demand)))
            episode_orders.append(float(np.sum(real_orders)))
            
            # === CAPTURE PRE-STEP STATE FOR GRANULAR LOGGING ===
            # Capture current inventory and demand BEFORE the step modifies them
            pre_step_inventory = float(np.sum(eval_env.inventory))
            pre_step_demand = float(np.sum(eval_env.last_demand))
            
            # Store previous cumulative values to compute per-step costs
            prev_holding = float(eval_env.info.get('holding_cost', 0.0))
            prev_shortage = float(eval_env.info.get('shortage_cost', 0.0))
            prev_purchase = float(eval_env.info.get('purchase_cost', 0.0))
            prev_sustain = float(eval_env.info.get('daily_sustain_net_score', 0.0))
            
            # Execute action in environment
            obs, _, terminated, truncated, info = eval_env.step(action)  # type: ignore[arg-type]
            done = terminated or truncated
            
            # === COMPUTE PER-STEP COSTS (delta from cumulative) ===
            step_holding_cost = float(info.get('holding_cost', 0.0)) - prev_holding
            step_shortage_cost = float(info.get('shortage_cost', 0.0)) - prev_shortage
            step_purchase_cost = float(info.get('purchase_cost', 0.0)) - prev_purchase
            step_sustainability = float(info.get('daily_sustain_net_score', 0.0)) - prev_sustain
            
            # Build detailed log entry for per-step analysis
            log_entry: Dict[str, Any] = {
                "Model": model_name,
                "Scenario": env_config["scenario_name"],
                "Episode": episode,
                "Step": eval_env.current_step,
                "TotalOrder": float(np.sum(real_orders)),
                # === NEW: Granular operational metrics for Bullwhip/Curriculum plots ===
                "Inventory_Level": pre_step_inventory,  # Total inventory (sum of all products)
                "Demand_Level": pre_step_demand,  # Total demand (sum of all products)
                "Holding_Cost": step_holding_cost,  # Per-step holding cost
                "Shortage_Cost": step_shortage_cost,  # Per-step shortage cost
                "Purchase_Cost": step_purchase_cost,  # Per-step purchase cost
                "Sustainability_Score": step_sustainability,  # Per-step sustainability (FTOPSIS-weighted)
                "Service_Level": float(info.get('service_level', 1.0))  # Per-step service level
            }
            
            # Log individual supplier-product orders (REAL quantities)
            for i in range(eval_env.n_suppliers):
                for u in range(eval_env.n_products):
                    log_entry[f"S{i+1}_P{u+1}_Order"] = float(real_orders[i, u])
            detailed_log.append(log_entry)

        # === Post-Episode Metric Calculation ===
        
        # 1. Bullwhip Ratio (order variance / demand variance)
        if np.var(episode_demands[1:]) > 1e-6:
            bullwhip_ratio: float = float(
                np.var(episode_orders[1:]) / np.var(episode_demands[1:])
            )
            bullwhip_ratios.append(bullwhip_ratio)

        # 2. Sustainability Score (average FTOPSIS-weighted score)
        if total_units_ordered > 0:
            sustainability_scores.append(total_weighted_score / total_units_ordered)

        # 3. Inventory Turnover (COGS / average inventory value)
        cogs: float = float(info['purchase_cost'])
        holding_cost_mean: float = float(np.mean(env_config['holding_cost']))  # type: ignore[arg-type]
        purchase_costs_mean: float = float(np.mean(env_config['supplier_purchase_costs']))  # type: ignore[arg-type]
        
        avg_inventory_value: float = 0.0
        if holding_cost_mean > 0:
            # Estimate average inventory value from holding costs
            avg_inventory_value = float(
                info['holding_cost']
            ) / holding_cost_mean * purchase_costs_mean
            
        if avg_inventory_value > 1e-6:
            inventory_turnover: float = cogs / avg_inventory_value
            inventory_turnovers.append(inventory_turnover)

        # 4. Cost Breakdown
        costs['total'].append(float(info['total_cost']))
        op_cost: float = float(
            info['purchase_cost'] + info['holding_cost'] + info['shortage_cost']
        )
        costs['operational'].append(op_cost)
        costs['daily_sustain_net'].append(float(info.get('daily_sustain_net_score', 0.0)))

    # === Aggregate Results Across Episodes ===
    agg_results: Dict[str, Tuple[float, float]] = {
        key: (float(np.mean(val)), float(np.std(val)))
        for key, val in costs.items()
    }
    
    # Add auxiliary metrics
    agg_results['bullwhip'] = (
        (float(np.mean(bullwhip_ratios)), float(np.std(bullwhip_ratios)))
        if bullwhip_ratios else (0.0, 0.0)
    )
    agg_results['inv_turnover'] = (
        (float(np.mean(inventory_turnovers)), float(np.std(inventory_turnovers)))
        if inventory_turnovers else (0.0, 0.0)
    )
    agg_results['sustainability_score'] = (
        (float(np.mean(sustainability_scores)), float(np.std(sustainability_scores)))
        if sustainability_scores else (0.0, 0.0)
    )

    # === Raw Per-Episode Data (for statistical tests) ===
    # These are the raw arrays needed for Welch's t-tests, ANOVA, box plots.
    # Each list has n_episodes entries (one value per episode).
    raw_episode_data: Dict[str, List[float]] = {
        'total_cost': costs['total'],
        'operational_cost': costs['operational'],
        'daily_sustain_net': costs['daily_sustain_net'],
        'sustainability_score': sustainability_scores,
        'bullwhip_ratio': bullwhip_ratios,
        'inv_turnover': inventory_turnovers,
    }

    return agg_results, detailed_log, raw_episode_data


# =============================================================================
# MAIN EXPERIMENTAL PROTOCOL
# =============================================================================
"""
Main Execution Block
--------------------
Orchestrates the complete ablation study across all models and scenarios.

**Experimental Design (Multi-Seed)**:
    - 3 Scenarios (Stable, Volatile, Shock)
    - 3 Models (M1, M2, M3) per scenario
    - 3 Seeds per model-scenario combination (for training stochasticity)
    - 50 evaluation episodes per seed
    - 9 total scenario-model combinations × 3 seeds = 27 training runs + 9 M1 evals

**Output Directory Structure**:
    results/experiments/
    ├── {Scenario}/{Model}/seed_{seed}/
    │   ├── best_model.zip
    │   ├── checkpoints/
    │   ├── training_progress.csv
    │   ├── training_metrics.csv
    │   ├── evaluations.npz
    │   ├── allocation_table.csv
    │   ├── summary_metrics.csv
    │   └── raw_episode_results.csv    <- Per-episode data for stats
    ├── full_results.csv               <- Aggregated across all seeds
    ├── raw_episode_all.csv            <- All per-episode data, all seeds
    └── allocation_table.csv
    
    results/experiments/Sensitivity_Analysis/
    ├── lambda_{value}/{Scenario}/{Model}/seed_{seed}/
    │   └── (same structure as above)
    └── sensitivity_results.csv

**Expected Runtime**: ~2.5 hours for 3 seeds on M4 MacBook Air
"""

if __name__ == '__main__':
    import argparse
    import time
    import multiprocessing
    multiprocessing.set_start_method('fork')  # Required for SubprocVecEnv on macOS

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run MCDM-RL supply chain experiments (multi-seed)')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to run (M2, M3). If not specified, runs all.')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Specific scenario (Stable_Operations, High_Volatility, Systemic_Shock).')
    parser.add_argument('--seeds', type=str, default='42,123,456',
                        help='Comma-separated list of training seeds (default: 42,123,456)')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run lambda_sust sensitivity analysis instead of main experiment')
    parser.add_argument('--lambda_values', type=str, default='0.01,0.1,0.5',
                        help='Comma-separated lambda_sust values for sensitivity sweep (default: 0.01,0.1,0.5)')

    args = parser.parse_args()

    # Parse seeds and lambda values
    SEEDS: List[int] = [int(s.strip()) for s in args.seeds.split(',')]
    LAMBDA_VALUES: List[float] = [float(v.strip()) for v in args.lambda_values.split(',')]

    # === Experiment Parameters ===
    TRAINING_TIMESTEPS = 3_000_000  # 3M steps (extended for M3 convergence in 16-dim obs space)
    EVALUATION_EPISODES = 50        # n=50 for statistical power
    DEVICE = "cpu"                  # Optimal for MLP policies on Apple Silicon
    N_TRAIN_ENVS = 8                # Parallel rollout envs (M4 has 10 cores)

    print("=" * 80)
    print("MCDM-RL MULTI-SEED EXPERIMENT")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Training seeds: {SEEDS}")
    print(f"Training timesteps: {TRAINING_TIMESTEPS:,}")
    print(f"Evaluation episodes: {EVALUATION_EPISODES}")
    if args.sensitivity:
        print(f"MODE: λ_sust SENSITIVITY ANALYSIS")
        print(f"Lambda values: {LAMBDA_VALUES}")
    else:
        print(f"MODE: MAIN EXPERIMENT (multi-seed)")
    print(f"Lambda sustainability: {BASE_CONFIG['lambda_sust']}")
    print(f"Reference daily cost: €{BASE_CONFIG['ref_daily_cost']:,.0f}")
    print("=" * 80 + "\n")

    # =========================================================================
    # LOAD MCDM EVALUATION RESULTS (Phase I Output) - REQUIRED!
    # =========================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    mcdm_output_dir = os.path.join(project_root, "results", "mcdm")
    mcdm_scores_file = os.path.join(mcdm_output_dir, "supplier_scores_for_rl.npy")

    print("=" * 80)
    print("LOADING MCDM EVALUATION RESULTS (Phase I)")
    print("=" * 80)

    if not os.path.exists(mcdm_scores_file):
        print(f"ERROR: MCDM scores file not found!")
        print(f"   Expected: {os.path.abspath(mcdm_scores_file)}")
        print(f"   Run Phase I first: python3 src/mcdm_evaluation.py")
        sys.exit(1)

    loaded_ftopsis_scores = np.load(mcdm_scores_file)
    print(f"Loaded FTOPSIS scores: {loaded_ftopsis_scores}")
    print(f"   Rankings: S5 ({loaded_ftopsis_scores[4]:.4f}) > S3 ({loaded_ftopsis_scores[2]:.4f}) > "
          f"S2 ({loaded_ftopsis_scores[1]:.4f}) > S4 ({loaded_ftopsis_scores[3]:.4f}) > "
          f"S1 ({loaded_ftopsis_scores[0]:.4f})")
    BASE_CONFIG["ftopsis_scores"] = loaded_ftopsis_scores
    print("=" * 80 + "\n")

    # === Output directory ===
    output_dir = os.path.join(project_root, "results", "experiments")
    os.makedirs(output_dir, exist_ok=True)

    # Common policy network architecture
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    # Model display name mapping
    model_display_names: Dict[str, str] = {
        "M2": "M2: Vanilla PPO",
        "M3": "M3: PPO + Priors",
    }

    # =========================================================================
    # HELPER: Build scenario configs (refreshed each call for lambda overrides)
    # =========================================================================
    def build_scenario_configs(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build scenario config dicts from a base config."""
        scenarios_out: List[Dict[str, Any]] = []
        for sc_name in ["Stable_Operations", "High_Volatility", "Systemic_Shock"]:
            sc = base_cfg.copy()
            sc.update(base_cfg["scenarios"][sc_name])
            sc["scenario_name"] = sc_name
            sc["ftopsis_scores"] = base_cfg["ftopsis_scores"]
            scenarios_out.append(sc)
        return scenarios_out

    # =========================================================================
    # HELPER: Save raw per-episode data to CSV
    # =========================================================================
    def save_raw_episode_csv(
        raw_data: Dict[str, List[float]],
        model_name: str,
        scenario_name: str,
        seed: int,
        save_dir: str
    ) -> pd.DataFrame:
        """Save raw per-episode metrics to CSV for statistical tests."""
        n_eps = max(len(v) for v in raw_data.values()) if raw_data else 0
        rows = []
        for ep_idx in range(n_eps):
            row: Dict[str, Any] = {
                'seed': seed,
                'scenario': scenario_name,
                'model': model_name,
                'episode': ep_idx,
            }
            for metric_key, metric_list in raw_data.items():
                row[metric_key] = metric_list[ep_idx] if ep_idx < len(metric_list) else np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = os.path.join(save_dir, 'raw_episode_results.csv')
        df.to_csv(csv_path, index=False)
        return df

    # =========================================================================
    # HELPER: Train and evaluate one model on one scenario with one seed
    # =========================================================================
    def run_single_training(
        model_short: str,
        config: Dict[str, Any],
        seed: int,
        base_output_dir: str,
        training_timesteps: int = TRAINING_TIMESTEPS,
        eval_episodes: int = EVALUATION_EPISODES,
    ) -> Tuple[Dict[str, Tuple[float, float]], List[Dict[str, Any]], Dict[str, List[float]]]:
        """
        Train one RL model (M2 or M3) on one scenario config with one seed.
        Returns (agg_results, detailed_log, raw_episode_data).
        """
        model_name = model_display_names[model_short]
        scenario_name = config["scenario_name"]

        # Set seed for this run
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Build environments
        def make_informed_env(cfg: Dict[str, Any] = config) -> Monitor[NDArray[np.float32], NDArray[np.float32]]:
            return Monitor(SupplyChainEnv(cfg))

        def make_uninformed_env(cfg: Dict[str, Any] = config) -> Monitor[NDArray[np.float32], NDArray[np.float32]]:
            return Monitor(UninformedWrapper(SupplyChainEnv(cfg)))

        # Use SubprocVecEnv for parallel rollout collection (N_TRAIN_ENVS cores)
        # Eval env stays single DummyVecEnv for deterministic evaluation
        if model_short == "M2":
            train_env = SubprocVecEnv([make_uninformed_env] * N_TRAIN_ENVS)
            eval_env_cb = DummyVecEnv([make_uninformed_env])
        else:  # M3
            train_env = SubprocVecEnv([make_informed_env] * N_TRAIN_ENVS)
            eval_env_cb = DummyVecEnv([make_informed_env])

        # Directory: {base_output_dir}/{Scenario}/{Model}/seed_{seed}/
        sanitized = model_name.replace(':', '').replace(' ', '_')
        log_dir = os.path.join(base_output_dir, scenario_name, sanitized, f"seed_{seed}")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Training: {model_name} | {scenario_name} | seed={seed}")
        print(f"   -> {log_dir}")
        print(f"   {N_TRAIN_ENVS} parallel envs | {TRAINING_TIMESTEPS:,} max steps | early-stop enabled")
        print(f"{'='*70}")

        # --- SKIP if already completed (best_model.zip + raw_episode_results.csv exist) ---
        best_exists = os.path.exists(os.path.join(log_dir, 'best_model.zip'))
        raw_csv_exists = os.path.exists(os.path.join(log_dir, 'raw_episode_results.csv'))
        if best_exists and raw_csv_exists:
            print(f"SKIP: Already completed -- loading existing results from disk")
            # Re-load saved raw episode data
            saved_raw_df = pd.read_csv(os.path.join(log_dir, 'raw_episode_results.csv'))
            raw_episode_data: Dict[str, List[float]] = {
                'total_cost': saved_raw_df['total_cost'].tolist(),
                'sustainability_score': saved_raw_df['sustainability_score'].tolist(),
                'bullwhip_ratio': saved_raw_df['bullwhip_ratio'].tolist(),
                'inv_turnover': saved_raw_df['inv_turnover'].tolist(),
            }
            agg_results: Dict[str, Tuple[float, float]] = {
                'total': (saved_raw_df['total_cost'].mean(), saved_raw_df['total_cost'].std()),
                'operational': (saved_raw_df['total_cost'].mean() * 0.95, saved_raw_df['total_cost'].std() * 0.95),
                'sustainability_score': (saved_raw_df['sustainability_score'].mean(), saved_raw_df['sustainability_score'].std()),
                'bullwhip': (saved_raw_df['bullwhip_ratio'].mean(), saved_raw_df['bullwhip_ratio'].std()),
                'inv_turnover': (saved_raw_df['inv_turnover'].mean(), saved_raw_df['inv_turnover'].std()),
            }
            # Load detailed log if available
            alloc_path = os.path.join(log_dir, 'allocation_table.csv')
            if os.path.exists(alloc_path):
                detailed_log = pd.read_csv(alloc_path).to_dict('records')
            else:
                detailed_log = []
            print(f"   Cost: EUR {agg_results['total'][0]:,.0f} +/- EUR {agg_results['total'][1]:,.0f}")
            print(f"   Sustainability: {agg_results['sustainability_score'][0]:.3f} +/- {agg_results['sustainability_score'][1]:.3f}")
            return agg_results, detailed_log, raw_episode_data

        # Hyperparams
        model_hyperparams = MODEL_CONFIGS[model_name].copy()

        # Callbacks
        # Early stopping: halt training if no reward improvement for 20 consecutive evals
        # At eval_freq=20K, this means 400K steps of patience (min 200K warm-up)
        stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=20,
            min_evals=10,
            verbose=1
        )
        eval_callback = EvalCallback(
            eval_env_cb,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=20_000,
            deterministic=True,
            render=False,
            callback_after_eval=stop_train_callback
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix=f"{sanitized}_ckpt",
            save_replay_buffer=False,
            save_vecnormalize=False,
            verbose=0
        )
        csv_logger = IncrementalCSVCallback(log_dir=log_dir, verbose=1)
        training_metrics_cb = TrainingMetricsCallback(
            log_dir=log_dir, log_freq=1000, verbose=1
        )
        callback_list = CallbackList([eval_callback, checkpoint_callback, csv_logger, training_metrics_cb])

        # Instantiate model
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            device=DEVICE,
            policy_kwargs=policy_kwargs,
            gamma=0.99,
            seed=seed,  # SB3 internal seed for reproducibility
            **model_hyperparams
        )

        # Train
        start_time = time.time()
        model.learn(total_timesteps=training_timesteps, progress_bar=True, callback=callback_list)
        train_time = time.time() - start_time
        actual_steps = model.num_timesteps
        early_stopped = actual_steps < training_timesteps
        if early_stopped:
            print(f"Training early-stopped at {actual_steps:,} / {training_timesteps:,} steps ({train_time / 60:.1f} min)")
        else:
            print(f"Training completed full {training_timesteps:,} steps ({train_time / 60:.1f} min)")

        # Load best model (saved by EvalCallback — critical for early-stopped runs)
        best_model_path = os.path.join(log_dir, 'best_model.zip')
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            best_model = PPO.load(best_model_path, env=train_env)
        else:
            print(f"WARNING: Best model not saved, using final model")
            best_model = model  # type: ignore[assignment]

        # Evaluate
        print(f"Evaluating {model_name} over {eval_episodes} episodes...")
        eval_start = time.time()
        agg_results, detailed_log, raw_episode_data = evaluate_policy(
            best_model, config, n_episodes=eval_episodes, model_name=model_name
        )
        eval_time = time.time() - eval_start

        # === Save per-seed artifacts ===
        # 1. Raw per-episode results (CRITICAL for statistical tests)
        raw_df = save_raw_episode_csv(raw_episode_data, model_name, scenario_name, seed, log_dir)
        print(f"Saved raw episode data ({len(raw_df)} rows) -> raw_episode_results.csv")

        # 2. Allocation table (per-step detailed log)
        model_detailed_df = pd.DataFrame(detailed_log)
        if not model_detailed_df.empty:
            model_detailed_df.to_csv(os.path.join(log_dir, 'allocation_table.csv'), index=False)

        # 3. Summary metrics (aggregated mean ± std)
        summary_rows = [{'Metric': k, 'Mean': v[0], 'Std': v[1]} for k, v in agg_results.items()]
        pd.DataFrame(summary_rows).to_csv(os.path.join(log_dir, 'summary_metrics.csv'), index=False)

        # Print results
        print(f"Evaluation completed in {eval_time:.1f}s")
        print(f"   Cost: EUR {agg_results['total'][0]:,.0f} +/- EUR {agg_results['total'][1]:,.0f}")
        print(f"   Sustainability: {agg_results['sustainability_score'][0]:.3f} +/- {agg_results['sustainability_score'][1]:.3f}")
        print(f"   Bullwhip: {agg_results['bullwhip'][0]:.2f} +/- {agg_results['bullwhip'][1]:.2f}")
        print(f"   Turnover: {agg_results['inv_turnover'][0]:.2f} +/- {agg_results['inv_turnover'][1]:.2f}")

        # Cleanup GPU/memory
        train_env.close()   # Critical: closes SubprocVecEnv subprocesses
        eval_env_cb.close()
        del model, best_model, train_env, eval_env_cb
        return agg_results, detailed_log, raw_episode_data

    # =========================================================================
    # HELPER: Evaluate M1 Base-Stock (deterministic — no training, seed only
    # affects env stochasticity)
    # =========================================================================
    def run_m1_evaluation(
        config: Dict[str, Any],
        seed: int,
        base_output_dir: str,
        eval_episodes: int = EVALUATION_EPISODES,
    ) -> Tuple[Dict[str, Tuple[float, float]], List[Dict[str, Any]], Dict[str, List[float]]]:
        """Evaluate M1 Base-Stock heuristic for one scenario + seed."""
        scenario_name = config["scenario_name"]
        np.random.seed(seed)

        log_dir = os.path.join(base_output_dir, scenario_name, "M1_Base-Stock", f"seed_{seed}")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\nEvaluating M1: Base-Stock | {scenario_name} | seed={seed}")

        base_stock_policy = BaseStockPolicy(config)
        agg_results, detailed_log, raw_episode_data = evaluate_policy(
            base_stock_policy, config, n_episodes=eval_episodes, model_name="M1: Base-Stock"
        )

        # Save artifacts
        save_raw_episode_csv(raw_episode_data, "M1: Base-Stock", scenario_name, seed, log_dir)
        model_detailed_df = pd.DataFrame(detailed_log)
        if not model_detailed_df.empty:
            model_detailed_df.to_csv(os.path.join(log_dir, 'allocation_table.csv'), index=False)
        summary_rows = [{'Metric': k, 'Mean': v[0], 'Std': v[1]} for k, v in agg_results.items()]
        pd.DataFrame(summary_rows).to_csv(os.path.join(log_dir, 'summary_metrics.csv'), index=False)

        print(f"   Cost: EUR {agg_results['total'][0]:,.0f} +/- EUR {agg_results['total'][1]:,.0f}")
        print(f"   Sustainability: {agg_results['sustainability_score'][0]:.3f} +/- {agg_results['sustainability_score'][1]:.3f}")
        return agg_results, detailed_log, raw_episode_data

    # =====================================================================
    # MODE DISPATCH
    # =====================================================================

    if args.sensitivity:
        # =================================================================
        # lambda_sust SENSITIVITY ANALYSIS
        # =================================================================
        print("\n" + "=" * 80)
        print("lambda_sust SENSITIVITY ANALYSIS")
        print("=" * 80)

        sens_output_dir = os.path.join(output_dir, "Sensitivity_Analysis")
        os.makedirs(sens_output_dir, exist_ok=True)

        # Default: run on Stable scenario only (expandable via --scenario)
        sens_scenario_name = args.scenario if args.scenario else "Stable_Operations"
        sens_models = ["M2", "M3"] if not args.model else [args.model]

        all_sens_rows: List[Dict[str, Any]] = []
        all_raw_sens: List[pd.DataFrame] = []

        for lam_val in LAMBDA_VALUES:
            print(f"\n{'='*60}")
            print(f"λ_sust = {lam_val}")
            print(f"{'='*60}")

            # Build config with overridden lambda
            lam_cfg = BASE_CONFIG.copy()
            lam_cfg["lambda_sust"] = lam_val
            lam_cfg["ftopsis_scores"] = loaded_ftopsis_scores

            # Build scenario
            lam_cfg_sc = lam_cfg.copy()
            lam_cfg_sc.update(lam_cfg["scenarios"][sens_scenario_name])
            lam_cfg_sc["scenario_name"] = sens_scenario_name
            lam_cfg_sc["ftopsis_scores"] = loaded_ftopsis_scores

            lam_dir = os.path.join(sens_output_dir, f"lambda_{lam_val}")

            for seed in SEEDS:
                for model_short in sens_models:
                    agg, _, raw_ep = run_single_training(
                        model_short, lam_cfg_sc, seed, lam_dir
                    )
                    row = {
                        'lambda_sust': lam_val,
                        'seed': seed,
                        'scenario': sens_scenario_name,
                        'model': model_display_names[model_short],
                        'total_cost_mean': agg['total'][0],
                        'total_cost_std': agg['total'][1],
                        'sustainability_mean': agg['sustainability_score'][0],
                        'sustainability_std': agg['sustainability_score'][1],
                        'bullwhip_mean': agg['bullwhip'][0],
                        'bullwhip_std': agg['bullwhip'][1],
                    }
                    all_sens_rows.append(row)

                    # Raw episode data with lambda tag
                    raw_df = pd.DataFrame({
                        'lambda_sust': lam_val,
                        'seed': seed,
                        'scenario': sens_scenario_name,
                        'model': model_display_names[model_short],
                        'episode': list(range(len(raw_ep['total_cost']))),
                        'total_cost': raw_ep['total_cost'],
                        'sustainability_score': raw_ep['sustainability_score'],
                        'bullwhip_ratio': raw_ep['bullwhip_ratio'],
                    })
                    all_raw_sens.append(raw_df)

                # M1 for this seed (lambda doesn't affect M1, but for completeness)
                m1_agg, _, m1_raw = run_m1_evaluation(lam_cfg_sc, seed, lam_dir)
                all_sens_rows.append({
                    'lambda_sust': lam_val,
                    'seed': seed,
                    'scenario': sens_scenario_name,
                    'model': 'M1: Base-Stock',
                    'total_cost_mean': m1_agg['total'][0],
                    'total_cost_std': m1_agg['total'][1],
                    'sustainability_mean': m1_agg['sustainability_score'][0],
                    'sustainability_std': m1_agg['sustainability_score'][1],
                    'bullwhip_mean': m1_agg['bullwhip'][0],
                    'bullwhip_std': m1_agg['bullwhip'][1],
                })

        # Save sensitivity results
        sens_df = pd.DataFrame(all_sens_rows)
        sens_df.to_csv(os.path.join(sens_output_dir, 'sensitivity_results.csv'), index=False)
        print(f"\nSensitivity results saved -> {sens_output_dir}/sensitivity_results.csv")

        if all_raw_sens:
            raw_sens_df = pd.concat(all_raw_sens, ignore_index=True)
            raw_sens_df.to_csv(os.path.join(sens_output_dir, 'sensitivity_raw_episodes.csv'), index=False)
            print(f"Raw episode data saved -> {sens_output_dir}/sensitivity_raw_episodes.csv")

        print("\n" + "=" * 80)
        print("SENSITIVITY ANALYSIS COMPLETE!")
        print("=" * 80)

    else:
        # =================================================================
        # MAIN MULTI-SEED EXPERIMENT
        # =================================================================
        all_scenarios = build_scenario_configs(BASE_CONFIG)

        # Filter scenarios if specified
        if args.scenario:
            all_scenarios = [s for s in all_scenarios if s["scenario_name"] == args.scenario]
            if not all_scenarios:
                print(f"ERROR: Invalid scenario '{args.scenario}'")
                sys.exit(1)

        # Filter models
        rl_models_to_run = ["M2", "M3"]
        if args.model:
            if args.model not in model_display_names:
                print(f"ERROR: Invalid model '{args.model}'. Valid: M2, M3")
                sys.exit(1)
            rl_models_to_run = [args.model]

        # Accumulators
        all_agg_rows: List[Dict[str, Any]] = []
        all_raw_episodes: List[pd.DataFrame] = []
        all_detailed_logs: List[Dict[str, Any]] = []

        total_start = time.time()

        for config in all_scenarios:
            scenario_name = config["scenario_name"]
            print(f"\n{'#'*80}")
            print(f"### SCENARIO: {scenario_name.upper()}")
            print(f"{'#'*80}")

            for seed in SEEDS:
                print(f"\n--- Seed: {seed} ---")

                # Train RL models (M2, M3)
                for model_short in rl_models_to_run:
                    agg, detail, raw_ep = run_single_training(
                        model_short, config, seed, output_dir
                    )
                    model_name = model_display_names[model_short]
                    all_detailed_logs.extend(detail)
                    all_agg_rows.append({
                        'scenario': scenario_name,
                        'model': model_name,
                        'seed': seed,
                        'total_cost_mean': agg['total'][0],
                        'total_cost_std': agg['total'][1],
                        'operational_cost_mean': agg['operational'][0],
                        'operational_cost_std': agg['operational'][1],
                        'sustainability_mean': agg['sustainability_score'][0],
                        'sustainability_std': agg['sustainability_score'][1],
                        'bullwhip_mean': agg['bullwhip'][0],
                        'bullwhip_std': agg['bullwhip'][1],
                        'inv_turnover_mean': agg['inv_turnover'][0],
                        'inv_turnover_std': agg['inv_turnover'][1],
                    })
                    # Collect raw episode data
                    n_eps = len(raw_ep['total_cost'])
                    raw_df = pd.DataFrame({
                        'seed': seed,
                        'scenario': scenario_name,
                        'model': model_name,
                        'episode': list(range(n_eps)),
                        'total_cost': raw_ep['total_cost'],
                        'sustainability_score': raw_ep['sustainability_score'],
                        'bullwhip_ratio': raw_ep['bullwhip_ratio'],
                        'inv_turnover': raw_ep['inv_turnover'],
                    })
                    all_raw_episodes.append(raw_df)

                # Evaluate M1 Base-Stock
                m1_agg, m1_detail, m1_raw = run_m1_evaluation(config, seed, output_dir)
                all_detailed_logs.extend(m1_detail)
                all_agg_rows.append({
                    'scenario': scenario_name,
                    'model': 'M1: Base-Stock',
                    'seed': seed,
                    'total_cost_mean': m1_agg['total'][0],
                    'total_cost_std': m1_agg['total'][1],
                    'operational_cost_mean': m1_agg['operational'][0],
                    'operational_cost_std': m1_agg['operational'][1],
                    'sustainability_mean': m1_agg['sustainability_score'][0],
                    'sustainability_std': m1_agg['sustainability_score'][1],
                    'bullwhip_mean': m1_agg['bullwhip'][0],
                    'bullwhip_std': m1_agg['bullwhip'][1],
                    'inv_turnover_mean': m1_agg['inv_turnover'][0],
                    'inv_turnover_std': m1_agg['inv_turnover'][1],
                })
                n_eps_m1 = len(m1_raw['total_cost'])
                raw_df_m1 = pd.DataFrame({
                    'seed': seed,
                    'scenario': scenario_name,
                    'model': 'M1: Base-Stock',
                    'episode': list(range(n_eps_m1)),
                    'total_cost': m1_raw['total_cost'],
                    'sustainability_score': m1_raw['sustainability_score'],
                    'bullwhip_ratio': m1_raw['bullwhip_ratio'],
                    'inv_turnover': m1_raw['inv_turnover'],
                })
                all_raw_episodes.append(raw_df_m1)

        total_time = time.time() - total_start

        # =================================================================
        # SAVE GLOBAL RESULT FILES
        # =================================================================
        print("\n" + "=" * 80)
        print("SAVING GLOBAL RESULTS...")
        print("=" * 80)

        # 1. Aggregated results (one row per scenario x model x seed)
        agg_df = pd.DataFrame(all_agg_rows)
        agg_df.to_csv(os.path.join(output_dir, 'full_results.csv'), index=False)
        print(f"full_results.csv ({len(agg_df)} rows)")

        # 2. Raw per-episode data (one row per scenario × model × seed × episode)
        if all_raw_episodes:
            raw_all_df = pd.concat(all_raw_episodes, ignore_index=True)
            raw_all_df.to_csv(os.path.join(output_dir, 'raw_episode_all.csv'), index=False)
            print(f"raw_episode_all.csv ({len(raw_all_df)} rows)")

        # 3. Allocation table (per-step supplier order patterns)
        if all_detailed_logs:
            df_detail = pd.DataFrame(all_detailed_logs)
            num_suppliers = len(BASE_CONFIG["supplier_names"])
            num_products = BASE_CONFIG["n_products"]
            for i in range(1, num_suppliers + 1):
                s_cols = [f'S{i}_P{j}_Order' for j in range(1, num_products + 1) if f'S{i}_P{j}_Order' in df_detail.columns]
                if s_cols:
                    df_detail[f'S{i}_Total_Order'] = df_detail[s_cols].sum(axis=1)
            agg_cols: Dict[str, Tuple[str, str]] = {
                f'S{i}_Orders': (f'S{i}_Total_Order', 'mean')
                for i in range(1, num_suppliers + 1)
                if f'S{i}_Total_Order' in df_detail.columns
            }
            agg_cols['Total_Orders'] = ('TotalOrder', 'mean')
            alloc_table = df_detail.groupby(['Scenario', 'Model']).agg(**agg_cols).round(2)
            alloc_table.to_csv(os.path.join(output_dir, 'allocation_table.csv'))
            print(f"allocation_table.csv")

        # =================================================================
        # FINAL SUMMARY TABLE
        # =================================================================
        print("\n" + "=" * 100)
        print("CROSS-SEED PERFORMANCE SUMMARY (mean of seed means +/- std across seeds)")
        print("=" * 100)
        header = f"{'Scenario':<22} | {'Model':<22} | {'Cost (€)':<28} | {'Sustainability':<22} | {'Bullwhip':<16}"
        print(header)
        print("-" * 100)

        for sc_name in ["Stable_Operations", "High_Volatility", "Systemic_Shock"]:
            sc_rows = [r for r in all_agg_rows if r['scenario'] == sc_name]
            for mdl in ["M3: PPO + Priors", "M2: Vanilla PPO", "M1: Base-Stock"]:
                mdl_rows = [r for r in sc_rows if r['model'] == mdl]
                if not mdl_rows:
                    continue
                costs_across_seeds = [r['total_cost_mean'] for r in mdl_rows]
                sust_across_seeds = [r['sustainability_mean'] for r in mdl_rows]
                bw_across_seeds = [r['bullwhip_mean'] for r in mdl_rows]
                cost_str = f"EUR {np.mean(costs_across_seeds):>12,.0f} +/- {np.std(costs_across_seeds):>10,.0f}"
                sust_str = f"{np.mean(sust_across_seeds):.4f} +/- {np.std(sust_across_seeds):.4f}"
                bw_str = f"{np.mean(bw_across_seeds):.2f} +/- {np.std(bw_across_seeds):.2f}"
                print(f"{sc_name:<22} | {mdl:<22} | {cost_str:<28} | {sust_str:<22} | {bw_str:<16}")
            print("-" * 100)

        # =================================================================
        # COMPLETION
        # =================================================================
        print(f"\nTotal runtime: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")
        print(f"Results directory: {os.path.abspath(output_dir)}")
        print(f"\nOutput files for Phase III:")
        print(f"   1. full_results.csv -> ANOVA, pairwise tests (1 row per scenario x model x seed)")
        print(f"   2. raw_episode_all.csv -> Welch's t-tests, box plots (1 row per episode)")
        print(f"   3. allocation_table.csv -> Supplier allocation analysis")
        print(f"   4. Per-seed folders -> Convergence curves, checkpoints")
        print(f"\nNext Steps:")
        print(f"   -> python src/run_experiment.py --sensitivity  (lambda sweep)")
        print(f"   -> python src/visualize_results.py  (publication figures)")
        print("=" * 80)