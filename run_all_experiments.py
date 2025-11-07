"""
Automated script to run experiments for all schedules and target MSE values.

Usage:
    python run_all_experiments.py

This script runs experiments for:
- Schedules: a1, a2, a3
- Target MSEs: [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
"""

import subprocess
import sys

# Configuration
schedules = ['a1', 'a2', 'a3'] 
target_mses = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

# Common parameters
num_samples = 2000
num_times = 101
num_seeds = 5
batch_size = 2
num_batches_per_epoch = 2

def run_experiment(schedule, target_mse):
    """Run a single experiment."""
    cmd = [
        'python', 'experiment.py',
        '--schedule', schedule,
        '--num_samples', str(num_samples),
        '--num_times', str(num_times),
        '--num_seeds', str(num_seeds),
        '--batch_size', str(batch_size),
        '--target_mse', str(target_mse),
        '--num_batches_per_epoch', str(num_batches_per_epoch)
    ]
    
    print("\n" + "="*80)
    print(f"Running experiment: schedule={schedule}, target_mse={target_mse}")
    print("Command:", ' '.join(cmd))
    print("="*80 + "\n")
    
    subprocess.run(cmd, check=False)

def main():
    """Run all experiments."""
    print("\n" + "="*80)
    print("AUTOMATED EXPERIMENT RUNNER")
    print("="*80)
    print(f"Total experiments: {len(schedules) * len(target_mses)}")
    print(f"Schedules: {schedules}")
    print(f"Target MSEs: {target_mses}")
    print("="*80 + "\n")
    
    for schedule in schedules:
        for target_mse in target_mses:
            run_experiment(schedule, target_mse)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()

