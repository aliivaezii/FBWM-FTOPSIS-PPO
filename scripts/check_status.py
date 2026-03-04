#!/usr/bin/env python3
"""
Check Experiment Status
======================
Shows which experiments are complete, which need CSV regeneration, etc.
Does NOT modify anything - just reports status.
"""

import os
import datetime

def main():
    project_root = os.path.join(os.path.dirname(__file__), '..')
    
    models = ["M1", "M2", "M3"]
    scenarios = ["Stable_Operations", "High_Volatility", "Systemic_Shock"]
    
    model_display = {
        "M1": "M1_Base-Stock",
        "M2": "M2_Vanilla_PPO",
        "M3": "M3_PPO_+_Priors",
    }
    
    print("="*80)
    print("EXPERIMENT STATUS CHECK")
    print("="*80)
    print("\nChecking which experiments are complete (training + CSV)...")
    print()
    
    total_complete = 0
    total_need_csv = 0
    total_missing = 0
    
    for model in models:
        model_name = model_display[model]
        print(f"\n{model}: {model_name}")
        print("-" * 80)
        
        for scenario in scenarios:
            model_path = os.path.join(
                project_root, 
                "results", "experiments", 
                scenario, 
                model_name, 
                "best_model.zip"
            )
            
            csv_path = os.path.join(
                project_root,
                "results", "experiments",
                scenario,
                model_name,
                "allocation_table.csv"
            )
            
            has_model = os.path.exists(model_path)
            has_csv = os.path.exists(csv_path)
            
            if has_model:
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(model_path))
                is_recent = mod_time.date() == datetime.date.today()
                date_str = mod_time.strftime("%b %d %H:%M")
            else:
                is_recent = False
                date_str = "N/A"
            
            # Status determination
            if has_model and has_csv and is_recent:
                status = "COMPLETE"
                total_complete += 1
            elif has_model and not has_csv and is_recent:
                status = "NEED CSV (model trained)"
                total_need_csv += 1
            elif has_model and not is_recent:
                status = "OLD (from previous run)"
                total_missing += 1
            else:
                status = "NOT STARTED"
                total_missing += 1
            
            print(f"  {scenario:20} | {date_str:15} | Model: {has_model:5} | CSV: {has_csv:5} | {status}")
    
    total_experiments = len(models) * len(scenarios)
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Complete (training + CSV):     {total_complete:2}/{total_experiments} ({total_complete/total_experiments*100:.0f}%)")
    print(f"  Need CSV regeneration:         {total_need_csv:2}/{total_experiments} ({total_need_csv/total_experiments*100:.0f}%)")
    print(f"  Not started or old:            {total_missing:2}/{total_experiments} ({total_missing/total_experiments*100:.0f}%)")
    print()
    print(f"  Total experiments needed: {total_experiments}")
    print(f"  Work remaining: {total_need_csv + total_missing} experiments")
    print()
    
    if total_need_csv > 0:
        print(f"TIP: Run 'python3 scripts/regenerate_csv_files.py' to fix the {total_need_csv} experiments")
        print(f"     that need CSV regeneration (~{total_need_csv} minutes)")
    
    if total_missing > 0:
        print(f"TIP: {total_missing} experiments still need to be run (training + evaluation)")
    
    print()

if __name__ == "__main__":
    main()
