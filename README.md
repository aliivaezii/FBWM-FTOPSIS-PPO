# SUPRA-PPO: A Hybrid FBWM–FTOPSIS–PPO Framework for Dynamic Supplier Order Allocation

> **Paper:** "When Do Sustainability Priors Help Reinforcement Learning? A Hybrid FBWM–FTOPSIS–PPO Framework for Dynamic Supplier Order Allocation"
>
> **Authors:** Ali Vaezi · Erfan Rabbani · Giulia Bruno
>
> **Submitted to:** *Journal of Cleaner Production*

This repository contains the full source code, MCDM data, and manuscript LaTeX source for reproducing all results in the paper. The framework addresses multi-product, multi-supplier order allocation under non-stationary demand, integrating multi-criteria decision-making (MCDM) priors with Proximal Policy Optimization (PPO).

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
│   ├── highlights.tex             # Journal highlights (standalone)
│   └── figures/                   # Paper figures (PNG)
├── requirements.txt               # Python dependencies
├── run_all_experiments.sh         # Shell script to run full experiment suite
├── LICENSE                        # MIT License
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
* **Ablation Study**: Three models (M1 = Baseline PPO, M2 = MCDM-weighted, M3 = Full SUPRA-PPO) across 3 scenarios × 3 seeds

---

## Reproducibility

- All configuration parameters are embedded in `run_experiment.py` — no external config files needed
- Scenarios, hyperparameters, and random seeds are explicitly declared in configuration dictionaries
- Pre-computed MCDM scores are provided in `results/mcdm/` so the RL experiments can run without recomputing them
- Large outputs (trained models, generated figures, TensorBoard logs) are excluded from Git via `.gitignore`
- To fully reproduce: install dependencies, run experiments, run visualisation

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{vaezi2025supra,
  author  = {Vaezi, Ali and Rabbani, Erfan and Bruno, Giulia},
  title   = {When Do Sustainability Priors Help Reinforcement Learning? A Hybrid FBWM--FTOPSIS--PPO Framework for Dynamic Supplier Order Allocation},
  journal = {Journal of Cleaner Production},
  year    = {2025}
}
```

---

## License

This project uses the MIT License. See the `LICENSE` file for full text.
