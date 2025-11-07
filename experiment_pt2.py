"""
Main experiment script for Part-2 bound verification.

Validates the bound: KL(p_1|q_1) ≤ ε√S

Usage:
    python experiment_pt2.py --mode synthetic --schedule a1 --delta_type constant --delta_beta 0.0 0.05 0.1 0.2
    python experiment_pt2.py --mode synthetic --schedule a2 --delta_type sine --delta_beta 0.05 0.1
"""

import argparse
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from utils import set_seed, get_device, ensure_dirs
from true_path import Schedule, schedule_to_enum, get_schedule_functions
from synthetic_velocity import SyntheticVelocity, constant_delta, sine_delta
from eval_pt2 import (
    compute_epsilon_pt2, compute_kl_at_t1_pt2, 
    compute_score_gap_integral_pt2
)
from plot_eps_curves import plot_lhs_rhs_vs_eps


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Verify KL bound via synthetic velocity fields')
    
    # Mode
    parser.add_argument('--mode', type=str, default='synthetic', choices=['synthetic'],
                        help='Experiment mode (only synthetic supported for Part-2)')
    
    # Schedule
    parser.add_argument('--schedule', type=str, choices=['a1', 'a2', 'a3'], required=True,
                        help='Schedule to use')
    
    # Delta configuration
    parser.add_argument('--delta_type', type=str, choices=['constant', 'sine'], required=True,
                        help='Type of perturbation δ')
    parser.add_argument('--delta_beta', type=float, nargs='+', required=True,
                        help='Beta values for δ (e.g., 0.0 0.05 0.1 0.2)')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_seeds', type=int, default=1,
                        help='Number of random seeds to average over')

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    
    # Evaluation parameters
    parser.add_argument('--K_eps', type=int, default=101,
                        help='Number of time points for ε computation')
    parser.add_argument('--N_eps', type=int, default=4096,
                        help='Samples per time point for ε computation')
    parser.add_argument('--N_kl', type=int, default=20000,
                        help='Number of samples for KL computation')
    parser.add_argument('--K_S', type=int, default=101,
                        help='Number of time points for S computation')
    parser.add_argument('--N_S', type=int, default=2048,
                        help='Samples per time point for S computation')
    parser.add_argument('--rtol', type=float, default=1e-6,
                        help='Relative tolerance for ODE solver')
    parser.add_argument('--atol', type=float, default=1e-8,
                        help='Absolute tolerance for ODE solver')
    
    # Output
    parser.add_argument('--outdir', type=str, default='data/part-2',
                        help='Output directory')
    
    # Plotting
    parser.add_argument('--no_make_eps_curves', dest='make_eps_curves', action='store_false', default=True,
                        help='Disable ε-curves plot generation (default: enabled)')
    
    return parser.parse_args()


def ensure_part2_dirs(outdir='data/part-2'):
    """Ensure Part-2 output directories exist."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(f'{outdir}/plots').mkdir(parents=True, exist_ok=True)
    Path(f'{outdir}/results').mkdir(parents=True, exist_ok=True)


def plot_bound_verification(lhs_list, rhs_list, delta_labels, schedule, save_path,
                             lhs_std=None, rhs_std=None):
    """
    Plot LHS vs RHS scatter with y=x reference line.
    
    Args:
        lhs_list: List of LHS values (KL(p_1|q_1))
        rhs_list: List of RHS values (ε√S)
        delta_labels: List of labels for each δ
        schedule: Schedule string
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    lhs_array = np.array(lhs_list, dtype=np.float64)
    rhs_array = np.array(rhs_list, dtype=np.float64)
    plt.scatter(rhs_array, lhs_array, s=100, alpha=0.7, label='Experiments', zorder=3)

    # Add labels
    for i, label in enumerate(delta_labels):
        plt.annotate(label, (rhs_array[i], lhs_array[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Reference line y=x
    max_rhs = np.max(rhs_array) if len(rhs_array) else 0
    max_lhs = np.max(lhs_array) if len(lhs_array) else 0
    max_val = max(max_rhs, max_lhs, 1e-6) * 1.1  # Ensure at least 1e-6
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (bound)', zorder=1)
    
    plt.xlabel('RHS = ε√S', fontsize=12)
    plt.ylabel('LHS = KL(p₁|q₁)', fontsize=12)
    plt.title(f'Bound Verification: KL(p₁|q₁) ≤ ε√S\nSchedule {schedule.upper()}', 
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scatter plot to {save_path}")
    plt.close()


def plot_bar_chart(lhs_list, rhs_list, delta_labels, schedule, save_path,
                   lhs_std=None, rhs_std=None):
    """
    Plot grouped bar chart of LHS and RHS per δ.
    
    Args:
        lhs_list: List of LHS values
        rhs_list: List of RHS values
        delta_labels: List of labels for each δ
        schedule: Schedule string
        save_path: Path to save plot
    """
    x = np.arange(len(delta_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, lhs_list, width, label='LHS = KL(p₁|q₁)', alpha=0.8)
    bars2 = ax.bar(x + width/2, rhs_list, width, label='RHS = ε√S', alpha=0.8)
    
    ax.set_xlabel('Perturbation δ', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Bound Verification: LHS vs RHS per δ\nSchedule {schedule.upper()}', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(delta_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (lhs, rhs) in enumerate(zip(lhs_list, rhs_list)):
        ax.text(i - width/2, lhs, f'{lhs:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, rhs, f'{rhs:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved bar chart to {save_path}")
    plt.close()


def plot_fhat_curves(fhat_data, delta_labels, schedule, save_path):
    """
    Plot f̂(t) curves (score gap over time) for each δ.
    
    Args:
        fhat_data: List of (t_grid, f_vals) tuples
        delta_labels: List of labels for each δ
        schedule: Schedule string
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    for i, ((t_grid, f_mean, f_std), label) in enumerate(zip(fhat_data, delta_labels)):
        plt.plot(t_grid, f_mean, label=f'δ: {label}', linewidth=2, alpha=0.8)
        if f_std is not None:
            f_lower = np.clip(f_mean - f_std, a_min=0.0, a_max=None)
            f_upper = f_mean + f_std
            plt.fill_between(t_grid, f_lower, f_upper, alpha=0.2)
    
    plt.xlabel('Time t', fontsize=12)
    plt.ylabel('E[|s_p - s_q|²]', fontsize=12)
    plt.title(f'Score Gap f̂(t) Over Time\nSchedule {schedule.upper()}', 
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved f̂(t) curve to {save_path}")
    plt.close()


def main():
    """Main experiment loop."""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    ensure_part2_dirs(args.outdir)
    
    # Convert schedule string to enum
    schedule_enum = schedule_to_enum(args.schedule)
    
    # Get schedule function
    a_fn, A_fn = get_schedule_functions(schedule_enum)
    
    print("=" * 80)
    print("Part-2: KL Bound Verification")
    print("=" * 80)
    print(f"Schedule: {args.schedule}")
    print(f"Delta type: {args.delta_type}")
    print(f"Delta betas: {args.delta_beta}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Storage for results
    results = []
    lhs_list = []
    rhs_list = []
    delta_labels = []
    fhat_data = []
    lhs_std_list = []
    rhs_std_list = []

    # Loop over δ values
    for beta in tqdm(args.delta_beta, desc="Processing δ values"):
        print(f"\n{'='*60}")
        print(f"Processing δ: beta = {beta}")
        print(f"{'='*60}")
        
        # Build delta function
        if args.delta_type == 'constant':
            delta_fn = constant_delta(beta)
            delta_label = f"δ(t)={beta}"
        elif args.delta_type == 'sine':
            delta_fn = sine_delta(beta)
            delta_label = f"δ(t)={beta}·sin(2πt)"
        else:
            raise ValueError(f"Unknown delta_type: {args.delta_type}")
        
        epsilon_vals = []
        KL_vals = []
        S_vals = []
        RHS_vals = []
        ratio_vals = []
        bound_flags = []
        f_vals_seeds = []
        t_grid = None

        for seed_idx in range(args.num_seeds):
            seed_value = args.seed + seed_idx
            set_seed(seed_value)

            # Construct synthetic velocity
            velocity = SyntheticVelocity(a_fn, delta_fn, dim=2)

            # Compute ε (RMS flow-matching loss)
            epsilon_hat_seed = compute_epsilon_pt2(
                a_fn, delta_fn, schedule_enum,
                K_eps=args.K_eps, N_eps=args.N_eps, dim=2,
                device=device, dtype=torch.float64
            )

            # Compute KL at t=1
            KL_hat_seed = compute_kl_at_t1_pt2(
                velocity, schedule_enum,
                N_kl=args.N_kl,
                rtol=args.rtol, atol=args.atol,
                device=device, dtype=torch.float64
            )

            # Compute S (score-gap integral)
            S_hat_seed, (t_grid_seed, f_vals_seed) = compute_score_gap_integral_pt2(
                velocity, schedule_enum,
                K_S=args.K_S, N_S=args.N_S,
                rtol=args.rtol, atol=args.atol,
                device=device, dtype=torch.float64
            )

            t_grid = np.array(t_grid_seed) if t_grid is None else t_grid
            f_vals_seeds.append(np.array(f_vals_seed, dtype=np.float64))

            RHS_seed = epsilon_hat_seed * math.sqrt(S_hat_seed)

            if RHS_seed < 1e-10:
                bound_satisfied_seed = KL_hat_seed < 1e-6
                ratio_seed = float('inf') if not bound_satisfied_seed else 0.0
            else:
                ratio_seed = KL_hat_seed / RHS_seed
                bound_satisfied_seed = KL_hat_seed <= RHS_seed

            epsilon_vals.append(epsilon_hat_seed)
            KL_vals.append(KL_hat_seed)
            S_vals.append(S_hat_seed)
            RHS_vals.append(RHS_seed)
            ratio_vals.append(ratio_seed)
            bound_flags.append(bound_satisfied_seed)

        epsilon_vals = np.array(epsilon_vals, dtype=np.float64)
        KL_vals = np.array(KL_vals, dtype=np.float64)
        S_vals = np.array(S_vals, dtype=np.float64)
        RHS_vals = np.array(RHS_vals, dtype=np.float64)
        ratio_vals = np.array(ratio_vals, dtype=np.float64)
        f_vals_array = np.stack(f_vals_seeds)

        epsilon_mean = float(np.mean(epsilon_vals))
        KL_mean = float(np.mean(KL_vals))
        S_mean = float(np.mean(S_vals))
        RHS_mean = float(np.mean(RHS_vals))

        epsilon_std = float(np.std(epsilon_vals, ddof=0)) if args.num_seeds > 1 else 0.0
        KL_std = float(np.std(KL_vals, ddof=0)) if args.num_seeds > 1 else 0.0
        S_std = float(np.std(S_vals, ddof=0)) if args.num_seeds > 1 else 0.0
        RHS_std = float(np.std(RHS_vals, ddof=0)) if args.num_seeds > 1 else 0.0

        finite_ratio = ratio_vals[np.isfinite(ratio_vals)]
        if finite_ratio.size > 0:
            ratio_mean = float(np.mean(finite_ratio))
            ratio_std = float(np.std(finite_ratio, ddof=0)) if finite_ratio.size > 1 else 0.0
        else:
            ratio_mean = float('inf')
            ratio_std = float('nan')

        if ratio_mean == float('inf') and np.all(np.isinf(ratio_vals)):
            print(f"  Ratio (LHS/RHS) = inf (RHS≈0) for all seeds")
        else:
            print(f"  Ratio (LHS/RHS) = {ratio_mean:.6f}")

        bound_satisfied = bool(np.all(bound_flags))
        print(f"  Bound satisfied across seeds: {'YES ✓' if bound_satisfied else 'NO ✗'}")

        f_vals_mean = np.mean(f_vals_array, axis=0)
        f_vals_std = np.std(f_vals_array, axis=0, ddof=0) if args.num_seeds > 1 else None

        print("  ε̂  = " + (f"{epsilon_mean:.6e} ± {epsilon_std:.2e}" if args.num_seeds > 1 else f"{epsilon_mean:.6e}"))
        print("  KL = " + (f"{KL_mean:.6e} ± {KL_std:.2e}" if args.num_seeds > 1 else f"{KL_mean:.6e}"))
        print("  Ŝ  = " + (f"{S_mean:.6e} ± {S_std:.2e}" if args.num_seeds > 1 else f"{S_mean:.6e}"))
        print("  RHS = ε̂√Ŝ = " + (f"{RHS_mean:.6e} ± {RHS_std:.2e}" if args.num_seeds > 1 else f"{RHS_mean:.6e}"))

        result = {
            'schedule': args.schedule,
            'delta_type': args.delta_type,
            'beta': float(beta),
            'delta_label': delta_label,
            'epsilon_hat': float(epsilon_mean),
            'epsilon_hat_std': float(epsilon_std),
            'KL_hat': float(KL_mean),
            'KL_hat_std': float(KL_std),
            'S_hat': float(S_mean),
            'S_hat_std': float(S_std),
            'RHS': float(RHS_mean),
            'RHS_std': float(RHS_std),
            'ratio': float(ratio_mean) if np.isfinite(ratio_mean) else float('inf'),
            'ratio_std': float(ratio_std) if np.isfinite(ratio_mean) else 'nan',
            'bound_satisfied': bound_satisfied,
            'num_seeds': args.num_seeds
        }
        results.append(result)

        lhs_list.append(KL_mean)
        rhs_list.append(RHS_mean)
        lhs_std_list.append(KL_std)
        rhs_std_list.append(RHS_std)
        delta_labels.append(delta_label)
        fhat_data.append((t_grid, f_vals_mean, f_vals_std))
 
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"{'δ':<30} {'LHS':<22} {'RHS':<22} {'Ratio':<15} {'Bound'}")
    print("-" * 80)
    for result, lhs_std, rhs_std in zip(results, lhs_std_list, rhs_std_list):
        status = "✓" if result['bound_satisfied'] else "✗"
        lhs_display = f"{result['KL_hat']:.6e}"
        rhs_display = f"{result['RHS']:.6e}"
        if args.num_seeds > 1:
            lhs_display += f" ± {lhs_std:.2e}"
            rhs_display += f" ± {rhs_std:.2e}"
        ratio_str = "inf" if result['ratio'] == float('inf') else f"{result['ratio']:.3f}"
        if args.num_seeds > 1 and np.isfinite(result['ratio']) and isinstance(result['ratio_std'], float):
            ratio_str += f" ± {result['ratio_std']:.3f}"
        print(f"{result['delta_label']:<30} {lhs_display:<22} {rhs_display:<22} "
              f"{ratio_str:<15} {status}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = Path(args.outdir) / 'results' / f'bound_{args.schedule}_{args.delta_type}_{timestamp}.json'
    with open(json_path, 'w') as f:
        # Convert inf to string for JSON compatibility
        results_json = []
        for result in results:
            result_copy = result.copy()
            if result_copy['ratio'] == float('inf'):
                result_copy['ratio'] = 'inf'
            if isinstance(result_copy.get('ratio_std'), float) and not math.isfinite(result_copy['ratio_std']):
                result_copy['ratio_std'] = 'nan'
            results_json.append(result_copy)
        json.dump({
            'args': {**vars(args), 'num_seeds': args.num_seeds},
            'results': results_json
        }, f, indent=2)
    print(f"\nSaved results to {json_path}")
    
    # Save CSV (use UTF-8 encoding to handle Greek delta character)
    csv_path = Path(args.outdir) / 'results' / f'bound_{args.schedule}_{args.delta_type}_{timestamp}.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('schedule,delta_type,beta,delta_label,epsilon_hat,epsilon_hat_std,KL_hat,KL_hat_std,S_hat,S_hat_std,RHS,RHS_std,ratio,ratio_std,bound_satisfied,num_seeds\n')
        for result in results:
            ratio_val = result['ratio']
            ratio_str = 'inf' if ratio_val == float('inf') else str(ratio_val)
            ratio_std = result['ratio_std']
            ratio_std_str = ratio_std if isinstance(ratio_std, str) else str(ratio_std)
            f.write(
                f"{result['schedule']},{result['delta_type']},{result['beta']},"
                f'"{result["delta_label"]}",{result["epsilon_hat"]},{result["epsilon_hat_std"]},'
                f"{result['KL_hat']},{result['KL_hat_std']},{result['S_hat']},{result['S_hat_std']},"
                f"{result['RHS']},{result['RHS_std']},{ratio_str},{ratio_std_str},{result['bound_satisfied']},{result['num_seeds']}\n"
            )
    print(f"Saved CSV to {csv_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Scatter plot
    scatter_path = Path(args.outdir) / 'plots' / f'bound_scatter_{args.schedule}_{args.delta_type}_{timestamp}.png'
    plot_bound_verification(lhs_list, rhs_list, delta_labels, args.schedule, scatter_path,
                            lhs_std=lhs_std_list if args.num_seeds > 1 else None,
                            rhs_std=rhs_std_list if args.num_seeds > 1 else None)
    
    # Bar chart
    bar_path = Path(args.outdir) / 'plots' / f'bound_bars_{args.schedule}_{args.delta_type}_{timestamp}.png'
    plot_bar_chart(lhs_list, rhs_list, delta_labels, args.schedule, bar_path,
                   lhs_std=lhs_std_list if args.num_seeds > 1 else None,
                   rhs_std=rhs_std_list if args.num_seeds > 1 else None)
    
    # f̂(t) curves
    fhat_path = Path(args.outdir) / 'plots' / f'fhat_curves_{args.schedule}_{args.delta_type}_{timestamp}.png'
    plot_fhat_curves(fhat_data, delta_labels, args.schedule, fhat_path)
    
    # ε-curves plot
    if args.make_eps_curves:
        eps_curves_path = Path(args.outdir) / 'plots' / f'eps_curves_synthetic_{args.schedule}_{args.delta_type}_{timestamp}.png'
        try:
            plot_lhs_rhs_vs_eps(
                str(csv_path),
                str(eps_curves_path),
                schedule=args.schedule,
                ylog=True,
                annotate=True,
                title=f"Bound components vs ε — synthetic {args.delta_type} — schedule {args.schedule.upper()}"
            )
        except Exception as e:
            print(f"Warning: Failed to generate ε-curves plot: {e}")
    
    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

