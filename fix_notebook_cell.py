"""
Quick fix for the notebook cell - change test_samples to num_samples
"""

# Replace this line in your notebook:
# print(f"Test Samples: {eval_results['test_samples']}")

# With this line:
print(f"Test Samples: {eval_results['num_samples']}")

# The correct key in evaluation_results.json is 'num_samples', not 'test_samples'