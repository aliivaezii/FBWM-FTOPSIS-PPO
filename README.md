<div align="center">

# SUPRA-PPO

### A Hybrid FBWM–FTOPSIS–PPO Framework for Dynamic Supplier Order Allocation

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Env-Gymnasium-red.svg)](https://gymnasium.farama.org/)
[![Code style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-yellow.svg)](https://peps.python.org/pep-0008/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

</div>

---

> **Paper:** *"When Do Sustainability Priors Help Reinforcement Learning? A Hybrid FBWM–FTOPSIS–PPO Framework for Dynamic Supplier Order Allocation"*
>
> **Authors:** Ali Vaezi · Erfan Rabbani · Giulia Bruno
>
> **Affiliation:** Politecnico di Torino, Italy

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Core Features](#core-features)
- [Experimental Design](#experimental-design)
- [Results Preview](#results-preview)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository provides the complete implementation for a **two-phase hybrid framework** that integrates Multi-Criteria Decision Making (MCDM) with Deep Reinforcement Learning (DRL) for sustainable supplier order allocation under demand uncertainty.

**Phase I** applies Fuzzy Best-Worst Method (FBWM) and Fuzzy TOPSIS to evaluate suppliers across economic, environmental, and resilience dimensions. **Phase II** embeds these sustainability priors into a custom PPO agent (SUPRA-PPO) operating within a stochastic supply chain simulation.

The key research question: *When and how do sustainability priors improve RL-based order allocation?*

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Phase I: MCDM                        │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │   FBWM   │───▸│  Criteria │───▸│     FTOPSIS      │  │
│  │ (weights) │    │  Weights  │    │ (supplier ranks) │  │
│  └──────────┘    └───────────┘    └────────┬─────────┘  │
└────────────────────────────────────────────┼────────────┘
                                             │
                            Sustainability Priors (scores)
                                             │
┌────────────────────────────────────────────▼────────────┐
│                 Phase II: RL Training                    │
│  ┌──────────────────┐    ┌───────────────────────────┐  │
│  │  SupplyChainEnv  │◂──▸│       SUPRA-PPO           │  │
│  │  (gymnasium)     │    │  - Adaptive Entropy (ES)   │  │
│  │  - 5 suppliers   │    │  - FTOPSIS reward shaping  │  │
│  │  - 3 products    │    │  - CV-based scheduling     │  │
│  │  - 3 scenarios   │    └───────────────────────────┘  │
│  └──────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Recompute MCDM supplier scores — pre-computed scores are provided in `results/mcdm/`:

```bash
python src/mcdm_evaluation.py
```

4. Run the full ablation study (3 scenarios × 3 models × 3 seeds = 27 runs):

```bash
bash run_all_experiments.sh
# or individually:
python src/run_experiment.py
```

5. Generate publication-quality figures:

```bash
python src/visualize_results.py
```

### Docker (Alternative)

Build and run the entire experiment suite in a container:

```bash
# Build the image
docker build -t supra-ppo .

# Run environment validation
docker run --rm supra-ppo

# Run experiments (results are saved inside the container)
docker run --rm -v $(pwd)/results:/app/results supra-ppo \
    python src/run_experiment.py

# Run visualisation
docker run --rm -v $(pwd)/results:/app/results supra-ppo \
    python src/visualize_results.py
```

---

## Repository Structure

```
├── src/
│   ├── run_experiment.py          # Main experiment runner (3 scenarios × 3 models × 3 seeds)
│   ├── mcdm_evaluation.py         # FBWM + FTOPSIS supplier scoring
│   ├── visualize_results.py       # Publication-quality figure generator
│   └── generate_supplier_table.py # Supplier evaluation LaTeX table
├── scripts/
│   ├── check_status.py            # Experiment progress monitor
│   └── evaluate_trained_models.py # Post-training model evaluation
├── test/
│   ├── test_environment.py        # Supply chain environment unit tests
│   └── test_reward_fix.py         # Reward function validation tests
├── results/
│   └── mcdm/                      # Pre-computed MCDM scores (tracked in Git)
│       ├── fbwm_weights.csv       # Fuzzy BWM criteria weights
│       ├── ftopsis_rankings.csv   # Fuzzy TOPSIS supplier rankings
│       ├── dimensional_scores.csv # Per-dimension supplier scores
│       └── supplier_scores_for_rl.npy  # NumPy array used by RL agent
├── Report/                        # Manuscript LaTeX source
│   ├── main.tex                   # Main manuscript
│   ├── preamble.tex               # LaTeX preamble & packages
│   ├── front_matter.tex           # Title, authors, abstract, keywords
│   ├── bibliography.bib           # BibTeX references (46 entries)
│   ├── highlights.tex             # Research highlights (standalone)
│   └── figures/                   # Paper figures (PNG)
├── requirements.txt               # Pinned Python dependencies
├── Dockerfile                     # Container-based reproducibility
├── .dockerignore                  # Docker build exclusions
├── run_all_experiments.sh         # Shell script to run full experiment suite
├── LICENSE                        # CC BY-NC-ND 4.0 (pre-publication)
└── README.md
```

> **Note:** Trained model checkpoints (~160 MB) and generated figures are excluded from Git.
> Run the experiment scripts to reproduce them locally.

---

## Core Features

* **SUPRA-PPO Algorithm**: A PPO variant featuring:
    * **Adaptive Entropy Scheduling (ES)** for dynamic exploration–exploitation balance
    * **Uncertainty-Driven Regularisation (UDR)** for promoting robust policies
* **Supply Chain Environment**: A `gymnasium`-compatible simulation with:
    * Non-stationary demand (trend, seasonality, stochastic shocks)
    * Lognormal lead times and systemic supplier disruptions
    * Three market scenarios: Stable Operations, High Volatility, Systemic Shock
* **MCDM Integration**: **Fuzzy Best-Worst Method (FBWM)** + **Fuzzy TOPSIS (FTOPSIS)** to generate sustainability–resilience priors
* **Ablation Study**: Three models (M1 = Base-Stock, M2 = Vanilla PPO, M3 = PPO + FTOPSIS priors) across 3 scenarios × 3 seeds

---

## Experimental Design

| Factor | Levels | Details |
|--------|--------|---------|
| **Models** | 3 | M1: Base-Stock heuristic, M2: Vanilla PPO, M3: SUPRA-PPO (PPO + FTOPSIS priors) |
| **Scenarios** | 3 | Stable Operations, High Volatility, Systemic Shock |
| **Seeds** | 3 | 42, 123, 456 (for statistical robustness) |
| **Timesteps** | 3M | Per model-scenario combination |
| **Sensitivity** | 5 | λ_sust ∈ {0.0, 0.1, 0.2, 0.3, 0.5} |
| **Total runs** | 27 | Full factorial: 3 models × 3 scenarios × 3 seeds |

---

## Results Preview

<div align="center">

| Metric | M1 (Base-Stock) | M2 (Vanilla PPO) | M3 (SUPRA-PPO) |
|--------|:---:|:---:|:---:|
| Cost minimisation | Baseline | Improved | **Best** |
| Sustainability score | Low | Moderate | **Highest** |
| Bullwhip reduction | None | Moderate | **Significant** |
| Disruption resilience | Fixed | Adaptive | **Most robust** |

*Full quantitative results are available after running the experiments.*

</div>

---

## Reproducibility

- All configuration parameters are embedded in `run_experiment.py` — no external config files needed
- Scenarios, hyperparameters, and random seeds are explicitly declared in configuration dictionaries
- Pre-computed MCDM scores are provided in `results/mcdm/` so the RL experiments can run without recomputing them
- Large outputs (trained models, generated figures, TensorBoard logs) are excluded from Git via `.gitignore`
- To fully reproduce: install dependencies, run experiments, run visualisation

---

## Citation

If you use this code in your research, please cite using the **"Cite this repository"** button on GitHub, or:

```bibtex
@software{vaezi2026suprappo,
  author    = {Vaezi, Ali and Rabbani, Erfan and Bruno, Giulia},
  title     = {{SUPRA-PPO}: A Hybrid {FBWM}--{FTOPSIS}--{PPO} Framework
               for Dynamic Supplier Order Allocation},
  year      = {2026},
  url       = {https://github.com/aliivaezii/FBWM-FTOPSIS-PPO},
  version   = {1.0.0},
  license   = {CC-BY-NC-ND-4.0}
}
```

---

## Contributing

Contributions are welcome. Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) during the pre-publication period. You may view and share the code with attribution, but commercial use and derivative works are not permitted. After the associated paper is published, this repository will be re-licensed under the MIT License.
