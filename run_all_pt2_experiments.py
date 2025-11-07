"""
Automated script to run all Part-2 bound verification experiments.

Runs experiments for all schedules (a1, a2, a3) with all delta configurations:
- Constant deltas: [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2] (9 values)
- Sine deltas: [0.025, 0.05, 0.075, 0.1] (4 values)
"""

import subprocess
import sys

# Configuration
schedules = ['a1', 'a2', 'a3']
# Constant deltas: doubled from 4 to 8+ values with finer increments
constant_betas = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
# Sine deltas: doubled from 2 to 4 values with finer increments
sine_betas = [0.025, 0.05, 0.075, 0.1]

# Default experiment parameters (as per instructions)
default_params = {
    'K_eps': 101,
    'N_eps': 4096,
    'N_kl': 20000,
    'K_S': 101,
    'N_S': 2048,
    'rtol': 1e-6,
    'atol': 1e-8,
    'seed': 42,
    'num_seeds': 1,
}


def run_experiment(schedule, delta_type, betas):
    """Run a single experiment."""
    cmd = [
        'python', 'experiment_pt2.py',
        '--mode', 'synthetic',
        '--schedule', schedule,
        '--delta_type', delta_type,
        '--seed', str(default_params['seed']),
        '--K_eps', str(default_params['K_eps']),
        '--N_eps', str(default_params['N_eps']),
        '--N_kl', str(default_params['N_kl']),
        '--K_S', str(default_params['K_S']),
        '--N_S', str(default_params['N_S']),
        '--rtol', str(default_params['rtol']),
        '--atol', str(default_params['atol']),
        '--num_seeds', str(default_params['num_seeds']),
    ]
    
    # Add delta_beta arguments (all values after single flag, as argparse expects with nargs='+')
    cmd.extend(['--delta_beta'])
    for beta in betas:
        cmd.append(str(beta))
    
    print("\n" + "="*80)
    print(f"Running: schedule={schedule}, delta_type={delta_type}, betas={betas}")
    print("Command:", ' '.join(cmd))
    print("="*80 + "\n")
    
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def main():
    """Run all experiments."""
    print("\n" + "="*80)
    print("PART-2 AUTOMATED EXPERIMENT RUNNER")
    print("="*80)
    print(f"Total experiments:")
    print(f"  Constant deltas: {len(schedules)} schedules × {len(constant_betas)} betas = {len(schedules) * len(constant_betas)}")
    print(f"  Sine deltas: {len(schedules)} schedules × {len(sine_betas)} betas = {len(schedules) * len(sine_betas)}")
    print(f"  Total: {len(schedules) * (len(constant_betas) + len(sine_betas))} experiments")
    print(f"\nSchedules: {schedules}")
    print(f"Constant betas: {constant_betas}")
    print(f"Sine betas: {sine_betas}")
    print("="*80 + "\n")
    
    results = []
    
    # Run constant delta experiments
    print("\n" + "="*80)
    print("PHASE 1: Constant Delta Experiments")
    print("="*80)
    for schedule in schedules:
        success = run_experiment(schedule, 'constant', constant_betas)
        results.append(('constant', schedule, success))
        print(f"\n{'✓' if success else '✗'} Constant deltas for schedule {schedule} completed")
    
    # Run sine delta experiments
    print("\n" + "="*80)
    print("PHASE 2: Sine Delta Experiments")
    print("="*80)
    for schedule in schedules:
        success = run_experiment(schedule, 'sine', sine_betas)
        results.append(('sine', schedule, success))
        print(f"\n{'✓' if success else '✗'} Sine deltas for schedule {schedule} completed")
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    successful = sum(1 for _, _, success in results if success)
    total = len(results)
    
    print(f"Total experiments: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    
    if total - successful > 0:
        print("\nFailed experiments:")
        for delta_type, schedule, success in results:
            if not success:
                print(f"  {delta_type} - {schedule}")
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80 + "\n")
    
    return successful == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

