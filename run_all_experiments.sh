#!/bin/bash
################################################################################
# SUPRA-PPO Experiment Runner
#
# Runs all experiments sequentially: 3 scenarios x 2 RL models x 3 seeds.
# M1 (Base-Stock) is evaluated within each run automatically.
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
#
# Or run individual commands from this file.
################################################################################

# Ensure we're in the virtual environment
source venv/bin/activate

echo "=================================================================================="
echo "Starting SUPRA-PPO Experiment Suite"
echo "=================================================================================="
echo "Total experiments: 6 (2 RL models x 3 scenarios)"
echo "Estimated time: ~45 hours (per seed)"
echo "Started at: $(date)"
echo ""
echo "TIP: This will run for a long time. Consider running in screen/tmux:"
echo "   screen -S supra_experiments"
echo "   ./run_all_experiments.sh"
echo "   # Press Ctrl+A then D to detach"
echo "   # Later: screen -r supra_experiments to reattach"
echo "=================================================================================="
echo ""
sleep 5

# Counter for progress
COMPLETED=0
TOTAL=6

################################################################################
# STABLE OPERATIONS SCENARIO
################################################################################

echo ""
echo "=========================================="
echo "SCENARIO 1/3: Stable Operations"
echo "=========================================="
echo ""

echo "[$((++COMPLETED))/$TOTAL] Running M2 on Stable Operations..."
python src/run_experiment.py --model M2 --scenario Stable_Operations --seed 42
echo "M2 Stable Operations completed at $(date)"
echo ""

echo "[$((++COMPLETED))/$TOTAL] Running M3 on Stable Operations..."
python src/run_experiment.py --model M3 --scenario Stable_Operations --seed 42
echo "M3 Stable Operations completed at $(date)"
echo ""

################################################################################
# HIGH VOLATILITY SCENARIO
################################################################################

echo ""
echo "=========================================="
echo "SCENARIO 2/3: High Volatility"
echo "=========================================="
echo ""

echo "[$((++COMPLETED))/$TOTAL] Running M2 on High Volatility..."
python src/run_experiment.py --model M2 --scenario High_Volatility --seed 42
echo "M2 High Volatility completed at $(date)"
echo ""

echo "[$((++COMPLETED))/$TOTAL] Running M3 on High Volatility..."
python src/run_experiment.py --model M3 --scenario High_Volatility --seed 42
echo "M3 High Volatility completed at $(date)"
echo ""

################################################################################
# SYSTEMIC SHOCK SCENARIO
################################################################################

echo ""
echo "=========================================="
echo "SCENARIO 3/3: Systemic Shock"
echo "=========================================="
echo ""

echo "[$((++COMPLETED))/$TOTAL] Running M2 on Systemic Shock..."
python src/run_experiment.py --model M2 --scenario Systemic_Shock --seed 42
echo "M2 Systemic Shock completed at $(date)"
echo ""

echo "[$((++COMPLETED))/$TOTAL] Running M3 on Systemic Shock..."
python src/run_experiment.py --model M3 --scenario Systemic_Shock --seed 42
echo "M3 Systemic Shock completed at $(date)"
echo ""

################################################################################
# COMPLETION
################################################################################

echo ""
echo "=================================================================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=================================================================================="
echo "Started at: $(cat .experiment_start_time 2>/dev/null || echo 'unknown')"
echo "Completed at: $(date)"
echo ""
echo "Results saved to: results/experiments/"
echo ""
echo "Next steps:"
echo "   1. python src/visualize_results.py"
echo ""
echo "=================================================================================="
