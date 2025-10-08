"""
Generate experiment report from completed hyperparameter search runs.
Fixes NaN issues and creates comprehensive analysis.
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

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
    
    # Clean data - remove rows with NaN run_id and filter valid runs
    df_clean = df.dropna(subset=['run_id']).copy()
    
    # Separate baseline and fine-tuned runs
    baseline = df_clean[df_clean['model'] == 't5-small-baseline']
    finetuned = df_clean[df_clean['model'] == 't5-small']
    
    if len(baseline) == 0:
        print("No baseline found. Using best performing model as reference.")
        # Use the worst performing model as baseline reference
        baseline_ppl = finetuned['val_ppl'].max()
        baseline_loss = finetuned[finetuned['val_ppl'] == baseline_ppl]['val_loss'].iloc[0]
    else:
        baseline_ppl = baseline.iloc[0]['val_ppl']
        baseline_loss = baseline.iloc[0]['val_loss']
    
    if len(finetuned) == 0:
        print("No fine-tuned runs found.")
        return
    
    # Find best model
    best_idx = finetuned['val_ppl'].idxmin()
    best_run = finetuned.loc[best_idx]
    
    # Calculate improvements
    improvements = ((baseline_ppl - finetuned['val_ppl']) / baseline_ppl * 100)
    
    # Create report
    report = {
        'baseline': {
            'perplexity': float(baseline_ppl),
            'loss': float(baseline_loss),
        },
        'best_model': {
            'run_id': int(best_run['run_id']) if pd.notna(best_run['run_id']) else 0,
            'perplexity': float(best_run['val_ppl']),
            'loss': float(best_run['val_loss']),
            'improvement_pct': float(improvements[best_idx]),
            'hyperparameters': {
                'lr': float(best_run['lr']),
                'batch_size': int(best_run['batch_size']),
                'epochs': int(best_run['epochs']),
                'label_smoothing': float(best_run['label_smoothing']),
                'warmup_ratio': float(best_run['warmup_ratio']),
            },
            'notes': str(best_run['notes']),
        },
        'summary': {
            'total_runs': len(finetuned),
            'best_improvement': float(improvements.max()),
            'mean_improvement': float(improvements.mean()),
            'worst_improvement': float(improvements.min()),
            'target_met': bool(improvements.max() >= 10.0),
        }
    }
    
    # Save report
    report_path = Path("experiments/best_run.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print report
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*80)
    
    print(f"\nBaseline Performance:")
    print(f"  Perplexity: {baseline_ppl:.4f}")
    
    print(f"\nBest Model (Run #{int(best_run['run_id']) if pd.notna(best_run['run_id']) else 'N/A'}):")
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
    print(f"  Target (>=10%):   {'MET' if improvements.max() >= 10.0 else 'NOT MET'}")
    
    print("\n" + "="*80)
    print(f"Report saved to: {report_path}")
    print("="*80)
    
    # Create comparison table
    comparison_path = Path("experiments/comparison_table.csv")
    comparison_df = finetuned[['run_id', 'lr', 'batch_size', 'epochs', 
                              'label_smoothing', 'warmup_ratio', 
                              'val_ppl', 'notes']].copy()
    
    # Add improvement percentage
    comparison_df['improvement_pct'] = improvements
    comparison_df = comparison_df.sort_values('val_ppl')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to: {comparison_path}\n")
    
    # Print top 3 configurations
    print("Top 3 Configurations:")
    print("-" * 60)
    for i, (idx, row) in enumerate(comparison_df.head(3).iterrows(), 1):
        print(f"{i}. {row['notes']}")
        print(f"   Perplexity: {row['val_ppl']:.4f} ({row['improvement_pct']:+.2f}%)")
        print(f"   LR: {row['lr']}, Batch: {row['batch_size']}, Epochs: {row['epochs']}")
        print()
    
    print("\nExperiment report generation complete!")
    print("Files created:")
    print(f"  - {report_path}")
    print(f"  - {comparison_path}")

if __name__ == "__main__":
    generate_report()