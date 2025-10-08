"""
Update the final experiment report with evaluation metrics.
"""

import json
import pandas as pd
from pathlib import Path

def update_final_report():
    """Update the final report with evaluation metrics."""
    
    # Load evaluation results
    eval_path = Path("experiments/evaluation_results.json")
    with open(eval_path, 'r') as f:
        eval_results = json.load(f)
    
    # Load existing best run report
    best_run_path = Path("experiments/best_run.json")
    with open(best_run_path, 'r') as f:
        best_run = json.load(f)
    
    # Add evaluation metrics to best run report
    best_run['evaluation_metrics'] = {
        'bleu': eval_results['bleu'],
        'rouge1': eval_results['rouge1'],
        'rouge2': eval_results['rouge2'],
        'rougeL': eval_results['rougeL'],
        'test_perplexity': eval_results['perplexity'],
        'test_samples': eval_results['num_samples']
    }
    
    # Save updated report
    with open(best_run_path, 'w') as f:
        json.dump(best_run, f, indent=2)
    
    # Also append to runs.csv
    runs_path = Path("experiments/runs.csv")
    df = pd.read_csv(runs_path)
    
    # Create new row for final evaluation
    new_row = {
        'model': 't5-small-finance-final',
        'lr': 0.0005,
        'batch_size': 16,
        'epochs': 5,
        'label_smoothing': 0.1,
        'warmup_ratio': 0.05,
        'seed': 42,
        'val_loss': 2.413,
        'val_ppl': 11.164,
        'notes': 'Final evaluation on test set',
        'run_id': 'FINAL',
        'timestamp': '2025-10-07 FINAL',
        'dropout': 0.1,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'improvement_pct': 92.01,
        'test_bleu': eval_results['bleu'],
        'test_rouge1': eval_results['rouge1'],
        'test_rouge2': eval_results['rouge2'],
        'test_rougeL': eval_results['rougeL'],
        'test_perplexity': eval_results['perplexity']
    }
    
    # Add new row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(runs_path, index=False)
    
    print("="*80)
    print("FINAL EXPERIMENT REPORT UPDATED")
    print("="*80)
    print(f"Best Model Performance:")
    print(f"  Validation Perplexity: {best_run['best_model']['perplexity']:.4f}")
    print(f"  Test Perplexity:       {eval_results['perplexity']:.4f}")
    print(f"  BLEU Score:            {eval_results['bleu']:.4f}")
    print(f"  ROUGE-1:               {eval_results['rouge1']:.4f}")
    print(f"  ROUGE-2:               {eval_results['rouge2']:.4f}")
    print(f"  ROUGE-L:               {eval_results['rougeL']:.4f}")
    print(f"  Improvement over baseline: {best_run['summary']['best_improvement']:.2f}%")
    print(f"  Target (>=10%): {'MET' if best_run['summary']['target_met'] else 'NOT MET'}")
    print("="*80)
    print("Files updated:")
    print(f"  - {best_run_path}")
    print(f"  - {runs_path}")
    print("="*80)

if __name__ == "__main__":
    update_final_report()