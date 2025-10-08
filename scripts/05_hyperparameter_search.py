"""
Automated hyperparameter search for T5 finance chatbot.
Runs multiple training configurations and tracks results.

This script systematically explores the hyperparameter space to find
the optimal configuration, ensuring ≥10% improvement over baseline.
"""

import subprocess
import json
from pathlib import Path
from itertools import product
import pandas as pd
import time

# Define hyperparameter search space
SEARCH_SPACE = {
    'lr': [1e-4, 3e-4, 5e-4],
    'batch_size': [8, 16, 32],
    'epochs': [5, 8],
    'label_smoothing': [0.0, 0.1, 0.2],
    'warmup_ratio': [0.0, 0.05, 0.1],
    'dropout': [0.1, 0.2],
    'weight_decay': [0.0, 0.01],
}

# Predefined configurations for systematic exploration
CONFIGURATIONS = [
    # Baseline configuration
    {'lr': 3e-4, 'batch_size': 16, 'epochs': 5, 'label_smoothing': 0.0, 
     'warmup_ratio': 0.0, 'dropout': 0.1, 'weight_decay': 0.01,
     'notes': 'Baseline configuration'},
    
    # Higher learning rate
    {'lr': 5e-4, 'batch_size': 16, 'epochs': 5, 'label_smoothing': 0.1,
     'warmup_ratio': 0.05, 'dropout': 0.1, 'weight_decay': 0.01,
     'notes': 'Higher learning rate'},
    
    # Larger batch size
    {'lr': 3e-4, 'batch_size': 32, 'epochs': 5, 'label_smoothing': 0.1,
     'warmup_ratio': 0.05, 'dropout': 0.1, 'weight_decay': 0.01,
     'notes': 'Larger batch size'},
    
    # More epochs with regularization
    {'lr': 3e-4, 'batch_size': 16, 'epochs': 8, 'label_smoothing': 0.2,
     'warmup_ratio': 0.1, 'dropout': 0.2, 'weight_decay': 0.01,
     'notes': 'More epochs + strong regularization'},
    
    # Lower learning rate, more training
    {'lr': 1e-4, 'batch_size': 16, 'epochs': 8, 'label_smoothing': 0.1,
     'warmup_ratio': 0.1, 'dropout': 0.1, 'weight_decay': 0.01,
     'notes': 'Lower LR, longer training'},
    
    # Aggressive configuration
    {'lr': 5e-4, 'batch_size': 32, 'epochs': 8, 'label_smoothing': 0.1,
     'warmup_ratio': 0.05, 'dropout': 0.15, 'weight_decay': 0.01,
     'notes': 'Aggressive training'},
    
    # Conservative configuration
    {'lr': 1e-4, 'batch_size': 8, 'epochs': 5, 'label_smoothing': 0.05,
     'warmup_ratio': 0.1, 'dropout': 0.1, 'weight_decay': 0.0,
     'notes': 'Conservative approach'},
]

def run_baseline():
    """Run baseline evaluation."""
    print("\n" + "="*80)
    print("STEP 1: Evaluating Baseline Model")
    print("="*80)
    
    cmd = [
        'python', 'scripts/02_train_t5_tf.py',
        '--baseline',
        '--track'
    ]
    
    subprocess.run(cmd, check=True)
    print("\n✓ Baseline evaluation complete\n")

def run_training(config: dict, run_num: int, total_runs: int):
    """Run training with specific configuration."""
    print("\n" + "="*80)
    print(f"STEP 2.{run_num}: Training Configuration {run_num}/{total_runs}")
    print("="*80)
    print(f"Configuration: {config['notes']}")
    print(f"Parameters: {json.dumps({k: v for k, v in config.items() if k != 'notes'}, indent=2)}")
    print("="*80 + "\n")
    
    cmd = [
        'python', 'scripts/02_train_t5_tf.py',
        '--lr', str(config['lr']),
        '--batch_size', str(config['batch_size']),
        '--epochs', str(config['epochs']),
        '--label_smoothing', str(config['label_smoothing']),
        '--warmup_ratio', str(config['warmup_ratio']),
        '--dropout', str(config['dropout']),
        '--weight_decay', str(config['weight_decay']),
        '--track',
        '--notes', config['notes'],
    ]
    
    # Save to different directory for each run
    save_dir = f"models/t5-small-finance-run{run_num}"
    cmd.extend(['--save_dir', save_dir])
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Run {run_num} complete\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Run {run_num} failed: {e}\n")
        return False

def generate_report():
    """Generate comprehensive experiment report."""
    print("\n" + "="*80)
    print("GENERATING EXPERIMENT REPORT")
    print("="*80)
    
    csv_path = Path("experiments/runs.csv")
    if not csv_path.exists():
        print("No experiments found.")
        return
    
    df = pd.read_csv(csv_path)
    
    # Separate baseline and fine-tuned
    baseline = df[df['model'] == 't5-small-baseline']
    finetuned = df[df['model'] != 't5-small-baseline']
    
    if len(baseline) == 0 or len(finetuned) == 0:
        print("Incomplete experiments. Need both baseline and fine-tuned runs.")
        return
    
    baseline_ppl = baseline.iloc[0]['val_ppl']
    
    # Find best model
    best_idx = finetuned['val_ppl'].idxmin()
    best_run = finetuned.loc[best_idx]
    
    # Calculate improvements
    improvements = ((baseline_ppl - finetuned['val_ppl']) / baseline_ppl * 100)
    
    # Create report
    report = {
        'baseline': {
            'perplexity': float(baseline_ppl),
            'loss': float(baseline.iloc[0]['val_loss']),
        },
        'best_model': {
            'run_id': int(best_run['run_id']),
            'perplexity': float(best_run['val_ppl']),
            'loss': float(best_run['val_loss']),
            'improvement_pct': float(improvements[best_idx]),
            'hyperparameters': {
                'lr': float(best_run['lr']),
                'batch_size': int(best_run['batch_size']),
                'epochs': int(best_run['epochs']),
                'label_smoothing': float(best_run['label_smoothing']),
                'warmup_ratio': float(best_run['warmup_ratio']),
                'dropout': float(best_run['dropout']),
                'weight_decay': float(best_run['weight_decay']),
            },
            'notes': best_run['notes'],
        },
        'summary': {
            'total_runs': len(finetuned),
            'best_improvement': float(improvements.max()),
            'mean_improvement': float(improvements.mean()),
            'worst_improvement': float(improvements.min()),
            'target_met': improvements.max() >= 10.0,
        }
    }
    
    # Save report
    report_path = Path("experiments/best_run.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print report
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*80)
    
    print(f"\nBaseline Performance:")
    print(f"  Perplexity: {baseline_ppl:.4f}")
    
    print(f"\nBest Model (Run #{int(best_run['run_id'])}):")
    print(f"  Perplexity:  {best_run['val_ppl']:.4f}")
    print(f"  Improvement: {improvements[best_idx]:+.2f}%")
    print(f"  Configuration: {best_run['notes']}")
    
    print(f"\nHyperparameters:")
    for key, value in report['best_model']['hyperparameters'].items():
        print(f"  {key:20s}: {value}")
    
    print(f"\nOverall Summary:")
    print(f"  Total Runs:       {len(finetuned)}")
    print(f"  Best Improvement: {improvements.max():+.2f}%")
    print(f"  Mean Improvement: {improvements.mean():+.2f}%")
    print(f"  Target (≥10%):    {'✓ MET' if improvements.max() >= 10.0 else '✗ NOT MET'}")
    
    print("\n" + "="*80)
    print(f"Report saved to: {report_path}")
    print("="*80 + "\n")
    
    # Create comparison table
    comparison_path = Path("experiments/comparison_table.csv")
    comparison_df = finetuned[['run_id', 'lr', 'batch_size', 'epochs', 
                                'label_smoothing', 'warmup_ratio', 'dropout',
                                'val_ppl', 'improvement_pct', 'notes']].copy()
    comparison_df = comparison_df.sort_values('val_ppl')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to: {comparison_path}\n")

def main():
    print("\n" + "="*80)
    print("AUTOMATED HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Total configurations to test: {len(CONFIGURATIONS)}")
    print("="*80 + "\n")
    
    # Step 1: Evaluate baseline
    run_baseline()
    
    # Step 2: Run all configurations
    successful_runs = 0
    for i, config in enumerate(CONFIGURATIONS, 1):
        success = run_training(config, i, len(CONFIGURATIONS))
        if success:
            successful_runs += 1
        
        # Brief pause between runs
        if i < len(CONFIGURATIONS):
            time.sleep(2)
    
    # Step 3: Generate report
    print("\n" + "="*80)
    print(f"COMPLETED {successful_runs}/{len(CONFIGURATIONS)} RUNS")
    print("="*80 + "\n")
    
    generate_report()
    
    # Print final summary
    subprocess.run(['python', 'scripts/02_train_t5_tf.py', '--summary'])

if __name__ == "__main__":
    main()
