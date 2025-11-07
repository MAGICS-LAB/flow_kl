"""
Main experiment script for Part-2 (Learning) bound verification.

Trains a velocity MLP and validates the bound: KL(p_1|q^θ_1) ≤ ε_θ√S_θ

Usage:
    python experiment_learn_pt2.py --schedule a1 --epochs 400 --device cpu --eval_checkpoints "all" --eval_only
    python experiment_learn_pt2.py --schedule a2 --epochs 400 --device cpu
    python experiment_learn_pt2.py --schedule a1 --eval_checkpoints "final,best,50,100" --eval_only
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
from true_path import Schedule, schedule_to_enum
from model_learn_pt2 import VelocityMLP
from train_learn_pt2 import train_velocity_model
from eval_learn_pt2 import (
    compute_epsilon_learn, compute_score_gap_integral_learn, 
    compute_kl_at_t1_learn
)
from plot_eps_curves import plot_lhs_rhs_vs_eps


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Verify KL bound via learned velocity model')
    
    # Schedule
    parser.add_argument('--schedule', type=str, choices=['a1', 'a2', 'a3'], required=True,
                        help='Schedule to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size (small default to slow training and capture diverse checkpoints)')
    parser.add_argument('--batches_per_epoch', type=int, default=2,
                        help='Number of batches per epoch (small default to slow training)')
    parser.add_argument('--val_times', type=int, default=64,
                        help='Number of time points for training validation')
    parser.add_argument('--val_samples_per_time', type=int, default=2048,
                        help='Samples per time point for training validation')
    parser.add_argument('--save_epochs', type=int, nargs='+', default=[10, 20, 40, 80, 160, 320],
                        help='Epoch numbers to save checkpoints regardless of improvement')
    
    # Evaluation parameters (separate from training)
    parser.add_argument('--eval_checkpoints', type=str, default='final,best',
                        help='Comma-separated list: final,best, or epoch numbers (e.g., "final,best,50,100")')
    parser.add_argument('--eval_val_times', type=int, default=101,
                        help='Number of time points for ε_θ computation')
    parser.add_argument('--eval_val_samples_per_time', type=int, default=1024,
                        help='Samples per time point for ε_θ computation')
    parser.add_argument('--eval_K_S', type=int, default=101,
                        help='Number of time points for S_θ computation')
    parser.add_argument('--eval_N_S', type=int, default=512,
                        help='Samples per time point for S_θ computation (reduced for speed)')
    parser.add_argument('--eval_N_kl', type=int, default=5000,
                        help='Number of samples for KL computation (reduced for speed)')
    parser.add_argument('--eval_rtol', type=float, default=1e-6,
                        help='Relative tolerance for ODE solver (evaluation)')
    parser.add_argument('--eval_atol', type=float, default=1e-8,
                        help='Absolute tolerance for ODE solver (evaluation)')
    parser.add_argument('--eval_chunk_size', type=int, default=1024,
                        help='Chunk size for batched evaluation')
    parser.add_argument('--tight_eval', action='store_true',
                        help='Use tighter tolerances for evaluation (rtol=3e-7, atol=3e-9)')
    parser.add_argument('--eval_num_seeds', type=int, default=1,
                        help='Number of random seeds to average over during evaluation')

    # Seeds
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for training')
    parser.add_argument('--eval_seed', type=int, default=12345,
                        help='Random seed for evaluation (independent from training)')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    
    # Output
    parser.add_argument('--outdir', type=str, default='data/part-2-learn',
                        help='Output directory root')
    
    # Mode
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training and evaluate existing checkpoints')
    
    # Plotting
    parser.add_argument('--no_make_eps_curves', dest='make_eps_curves', action='store_false', default=True,
                        help='Disable ε-curves plot generation (default: enabled)')
    
    return parser.parse_args()


def ensure_learn_dirs(outdir, schedule):
    """Ensure Part-2-learn output directories exist."""
    base_dir = Path(outdir) / schedule
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (base_dir / 'results').mkdir(parents=True, exist_ok=True)
    (base_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (base_dir / 'logs').mkdir(parents=True, exist_ok=True)
    return base_dir


def plot_bound_verification(results, schedule, save_path):
    """
    Plot LHS vs RHS scatter with y=x reference line.
    
    Args:
        results: List of dicts with 'kl_hat', 'rhs', 'epoch' keys
        schedule: Schedule string
        save_path: Path to save plot
    """
    if not results:
        print("No results to plot")
        return
    
    lhs_list = [r['kl_hat'] for r in results]
    rhs_list = [r['rhs'] for r in results]
    epochs = [r['epoch'] for r in results]
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with epoch labels
    scatter = plt.scatter(rhs_list, lhs_list, s=100, alpha=0.7, c=epochs, 
                         cmap='viridis', label='Checkpoints', zorder=3)
    plt.colorbar(scatter, label='Epoch')
    
    # Add epoch labels
    for r in results:
        plt.annotate(f"E{r['epoch']}", (r['rhs'], r['kl_hat']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Reference line y=x
    max_rhs = max(rhs_list) if rhs_list else 0
    max_lhs = max(lhs_list) if lhs_list else 0
    max_val = max(max_rhs, max_lhs, 1e-6) * 1.1
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (bound)', zorder=1)
    
    plt.xlabel('RHS = ε_θ√S_θ', fontsize=12)
    plt.ylabel('LHS = KL(p₁|q₁^θ)', fontsize=12)
    plt.title(f'Bound Verification: KL(p₁|q₁^θ) ≤ ε_θ√S_θ\nSchedule {schedule.upper()}', 
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved scatter plot to {save_path}")
    plt.close()


def plot_fhat_curves(fhat_data_list, labels, schedule, save_path):
    """
    Plot f̂(t) curves for multiple checkpoints.
    
    Args:
        fhat_data_list: List of (t_grid, f_vals) tuples
        labels: List of labels (e.g., epoch numbers)
        schedule: Schedule string
        save_path: Path to save plot
    """
    if not fhat_data_list:
        return
    
    plt.figure(figsize=(10, 6))
    
    for data, label in zip(fhat_data_list, labels):
        if len(data) == 3:
            t_grid, f_vals, f_std = data
        else:
            t_grid, f_vals = data
            f_std = None

        plt.plot(t_grid, f_vals, label=f"Epoch {label}", linewidth=1.5, alpha=0.7)
        if f_std is not None and np.any(f_std > 1e-12):
            f_lower = np.clip(f_vals - f_std, a_min=0.0, a_max=None)
            f_upper = f_vals + f_std
            plt.fill_between(t_grid, f_lower, f_upper, alpha=0.18)
    
    plt.xlabel('Time t', fontsize=12)
    plt.ylabel('f̂(t) = E|s_p - s_q^θ|²', fontsize=12)
    plt.title(f'Score Gap Integrand f̂(t)\nSchedule {schedule.upper()}', 
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved f̂(t) curve to {save_path}")
    plt.close()


def load_checkpoint_metadata(checkpoint_path):
    """Load checkpoint metadata JSON."""
    metadata_path = checkpoint_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def main():
    """Main experiment loop."""
    args = parse_args()
    
    # Setup
    schedule_enum = schedule_to_enum(args.schedule)
    base_dir = ensure_learn_dirs(args.outdir, args.schedule)
    checkpoint_dir = base_dir / 'checkpoints'
    
    # Apply tight eval if requested
    if args.tight_eval:
        args.eval_rtol = 3e-7
        args.eval_atol = 3e-9
    
    # Set default dtype
    torch.set_default_dtype(torch.float64)
    
    print("=" * 80)
    print("Part-2 (Learning): KL Bound Verification")
    print("=" * 80)
    print(f"Schedule: {args.schedule}")
    print(f"Device: {args.device}")
    print(f"Training seed: {args.seed}")
    print(f"Eval seed: {args.eval_seed}")
    print("=" * 80)
    
    model = None
    
    # Training phase
    if not args.eval_only:
        print("\n" + "=" * 80)
        print("Training Phase")
        print("=" * 80)
        
        # Set training seed
        set_seed(args.seed)
        device = get_device(args.device)
        
        # Initialize model
        model = VelocityMLP(hidden_dims=[64, 64], activation='silu')
        
        # Auto batches_per_epoch if not specified
        batches_per_epoch = args.batches_per_epoch
        if batches_per_epoch is None:
            # Default: 2 batches per epoch to slow training and capture diverse checkpoints
            batches_per_epoch = 2
        
        # Train
        best_mse, history = train_velocity_model(
            model=model,
            schedule=schedule_enum,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_batches_per_epoch=batches_per_epoch,
            val_times=args.val_times,
            val_samples_per_time=args.val_samples_per_time,
            device=device,
            dtype=torch.float64,
            checkpoint_dir=str(checkpoint_dir),
            save_epochs=args.save_epochs,
            seed=args.seed
        )
        
        print(f"\nTraining complete. Best val MSE: {best_mse:.6e}")
        print(f"Checkpoints saved: {len(history.get('checkpoints', []))}")
    else:
        print("\nSkipping training (--eval_only mode)")
        device = get_device(args.device)
    
    # Evaluation phase
    print("\n" + "=" * 80)
    print("Evaluation Phase")
    print("=" * 80)
    
    # Parse checkpoint list
    checkpoint_names_raw = [name.strip() for name in args.eval_checkpoints.split(',')]
    
    # Handle "all" to evaluate all saved checkpoints
    if 'all' in checkpoint_names_raw:
        # Find all checkpoint .pt files (excluding aliases)
        all_checkpoints = []
        for ckpt_file in checkpoint_dir.glob("ckpt__*.pt"):
            # Extract epoch from filename
            try:
                # Format: ckpt__sched=a1__epoch=134__valmse=...
                parts = ckpt_file.stem.split('__')
                for part in parts:
                    if part.startswith('epoch='):
                        epoch_num = int(part.split('=')[1])
                        all_checkpoints.append(epoch_num)
                        break
            except (ValueError, IndexError):
                continue
        # Sort by epoch and convert to strings
        all_checkpoints = sorted(set(all_checkpoints))
        checkpoint_names = ['best', 'final'] + [str(e) for e in all_checkpoints]
        print(f"\nFound {len(all_checkpoints)} saved checkpoints. Evaluating all of them plus 'best' and 'final'.")
    else:
        checkpoint_names = checkpoint_names_raw
    
    results = []
    fhat_data_list = []
    labels = []
    
    for checkpoint_name in checkpoint_names:
        print(f"\nProcessing checkpoint: {checkpoint_name}")
        
        # Determine checkpoint path
        if checkpoint_name == 'final':
            ckpt_path = checkpoint_dir / 'final.pt'
        elif checkpoint_name == 'best':
            ckpt_path = checkpoint_dir / 'best.pt'
        else:
            # Try to find epoch-specific checkpoint
            try:
                epoch_num = int(checkpoint_name)
                # Find checkpoint file matching this epoch
                pattern = f"*epoch={epoch_num}*.pt"
                matches = list(checkpoint_dir.glob(pattern))
                if matches:
                    ckpt_path = matches[0]
                else:
                    print(f"  Warning: No checkpoint found for epoch {epoch_num}, skipping")
                    continue
            except ValueError:
                print(f"  Warning: Invalid checkpoint name '{checkpoint_name}', skipping")
                continue
        
        if not ckpt_path.exists():
            print(f"  Warning: Checkpoint {ckpt_path} not found, skipping")
            continue
        
        # Load checkpoint metadata
        metadata = load_checkpoint_metadata(ckpt_path)
        if metadata is None:
            print(f"  Warning: No metadata found for {ckpt_path}")
            epoch = int(checkpoint_name) if checkpoint_name.isdigit() else -1
            val_mse_train = None
        else:
            epoch = metadata['epoch']
            val_mse_train = metadata['val_mse']
        
        val_mse_str = f"{val_mse_train:.6e}" if val_mse_train is not None else 'N/A'
        print(f"  Epoch: {epoch}, Val MSE (train): {val_mse_str}")
        
        # Load model if needed
        if model is None:
            model = VelocityMLP(hidden_dims=[64, 64], activation='silu')
        
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        model = model.to(device)
        
        # Set eval seed
        set_seed(args.eval_seed)
        
        epsilon_vals = []
        S_vals = []
        KL_vals = []
        RHS_vals = []
        ratio_vals = []
        bound_flags = []
        f_vals_seeds = []
        t_grid = None

        for seed_offset in range(args.eval_num_seeds):
            current_seed = args.eval_seed + seed_offset
            set_seed(current_seed)

            print(f"  Seed {seed_offset + 1}/{args.eval_num_seeds}: computing ε_θ...")
            epsilon_hat = compute_epsilon_learn(
                v_theta=model,
                schedule=schedule_enum,
                val_times=args.eval_val_times,
                val_samples_per_time=args.eval_val_samples_per_time,
                device=device,
                dtype=torch.float64
            )

            print("    Computing S_θ...")
            S_hat, (t_grid_seed, f_vals_seed) = compute_score_gap_integral_learn(
                v_theta=model,
                schedule=schedule_enum,
                K_S=args.eval_K_S,
                N_S=args.eval_N_S,
                rtol=args.eval_rtol,
                atol=args.eval_atol,
                eval_seed=current_seed,
                device=device,
                dtype=torch.float64,
                chunk_size=args.eval_chunk_size
            )

            print("    Computing KL(p₁|q₁^θ)...")
            KL_hat = compute_kl_at_t1_learn(
                v_theta=model,
                schedule=schedule_enum,
                N_kl=args.eval_N_kl,
                rtol=args.eval_rtol,
                atol=args.eval_atol,
                eval_seed=current_seed,
                device=device,
                dtype=torch.float64,
                chunk_size=args.eval_chunk_size
            )

            RHS = epsilon_hat * math.sqrt(S_hat)

            if RHS < 1e-10:
                bound_satisfied_seed = KL_hat < 1e-6
                ratio_seed = float('inf') if not bound_satisfied_seed else 0.0
            else:
                ratio_seed = KL_hat / RHS
                bound_satisfied_seed = KL_hat <= RHS

            epsilon_vals.append(epsilon_hat)
            S_vals.append(S_hat)
            KL_vals.append(KL_hat)
            RHS_vals.append(RHS)
            ratio_vals.append(ratio_seed)
            bound_flags.append(bound_satisfied_seed)

            if t_grid is None:
                t_grid = np.array(t_grid_seed, dtype=np.float64)
            f_vals_seeds.append(np.array(f_vals_seed, dtype=np.float64))

        epsilon_vals = np.array(epsilon_vals, dtype=np.float64)
        S_vals = np.array(S_vals, dtype=np.float64)
        KL_vals = np.array(KL_vals, dtype=np.float64)
        RHS_vals = np.array(RHS_vals, dtype=np.float64)
        ratio_vals = np.array(ratio_vals, dtype=np.float64)

        epsilon_mean = float(np.mean(epsilon_vals))
        S_mean = float(np.mean(S_vals))
        KL_mean = float(np.mean(KL_vals))
        RHS_mean = float(np.mean(RHS_vals))

        epsilon_std = float(np.std(epsilon_vals, ddof=0)) if args.eval_num_seeds > 1 else 0.0
        S_std = float(np.std(S_vals, ddof=0)) if args.eval_num_seeds > 1 else 0.0
        KL_std = float(np.std(KL_vals, ddof=0)) if args.eval_num_seeds > 1 else 0.0
        RHS_std = float(np.std(RHS_vals, ddof=0)) if args.eval_num_seeds > 1 else 0.0

        finite_ratio_mask = np.isfinite(ratio_vals)
        if finite_ratio_mask.any():
            ratio_mean = float(np.mean(ratio_vals[finite_ratio_mask]))
            ratio_std = float(np.std(ratio_vals[finite_ratio_mask], ddof=0)) if (
                args.eval_num_seeds > 1 and finite_ratio_mask.sum() > 1
            ) else 0.0
        else:
            ratio_mean = float('inf')
            ratio_std = 'nan'

        bound_satisfied = bool(np.all(bound_flags))

        f_vals_array = np.stack(f_vals_seeds)
        f_vals_mean = np.mean(f_vals_array, axis=0)
        f_vals_std = np.std(f_vals_array, axis=0, ddof=0) if args.eval_num_seeds > 1 else None

        def format_mean_std(mean_val, std_val):
            if args.eval_num_seeds > 1:
                return f"{mean_val:.6e} ± {std_val:.2e}"
            return f"{mean_val:.6e}"

        print(f"    ε_θ = {format_mean_std(epsilon_mean, epsilon_std)}")
        print(f"    S_θ = {format_mean_std(S_mean, S_std)}")
        print(f"    KL(p₁|q₁^θ) = {format_mean_std(KL_mean, KL_std)}")
        print(f"    RHS = ε_θ√S_θ = {format_mean_std(RHS_mean, RHS_std)}")

        if ratio_mean == float('inf'):
            print("    Ratio (LHS/RHS) = inf (RHS≈0)")
        else:
            ratio_disp = (
                f"{ratio_mean:.6f} ± {ratio_std:.3f}" if (
                    args.eval_num_seeds > 1 and isinstance(ratio_std, float) and np.isfinite(ratio_std)
                ) else f"{ratio_mean:.6f}"
            )
            print(f"    Ratio (LHS/RHS) = {ratio_disp}")
        print(f"    Bound satisfied across seeds: {'YES ✓' if bound_satisfied else 'NO ✗'}")

        # Store results
        result = {
            'epoch': epoch,
            'ckpt_path': str(ckpt_path),
            'val_mse_train': val_mse_train,
            'eps_eval': epsilon_mean,
            'eps_eval_std': epsilon_std,
            'S_eval': S_mean,
            'S_eval_std': S_std,
            'KL_eval': KL_mean,
            'KL_eval_std': KL_std,
            'kl_hat': KL_mean,
            'kl_hat_std': KL_std,
            'rhs': RHS_mean,
            'rhs_std': RHS_std,
            'RHS': RHS_mean,
            'RHS_std': RHS_std,
            'ratio': ratio_mean if np.isfinite(ratio_mean) else float('inf'),
            'ratio_std': ratio_std,
            'bound_satisfied': bool(bound_satisfied),
            'eval_params': {
                'eval_val_times': args.eval_val_times,
                'eval_val_samples_per_time': args.eval_val_samples_per_time,
                'eval_K_S': args.eval_K_S,
                'eval_N_S': args.eval_N_S,
                'eval_N_kl': args.eval_N_kl,
                'eval_rtol': args.eval_rtol,
                'eval_atol': args.eval_atol,
                'eval_chunk_size': args.eval_chunk_size
            },
            'train_params': metadata.get('train_params', {}) if metadata else {},
            'seeds': {
                'training': args.seed,
                'eval_start': args.eval_seed,
                'eval_num_seeds': args.eval_num_seeds,
            }
        }
        
        results.append(result)
        fhat_data_list.append((t_grid, f_vals_mean, f_vals_std))
        labels.append(epoch)
    
    # Save results
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = base_dir / 'results' / f'bound_{args.schedule}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json_results = []
            for r in results:
                r_copy = r.copy()
                if r_copy['ratio'] == float('inf'):
                    r_copy['ratio'] = 'inf'
                ratio_std_val = r_copy.get('ratio_std')
                if isinstance(ratio_std_val, float) and not math.isfinite(ratio_std_val):
                    r_copy['ratio_std'] = 'nan'
                json_results.append(r_copy)
            args_dict = vars(args).copy()
            json.dump({
                'args': args_dict,
                'results': json_results
            }, f, indent=2)
        print(f"\nSaved results to {json_path}")
        
        # Save CSV
        csv_path = base_dir / 'results' / f'bound_{args.schedule}_{timestamp}.csv'
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('epoch,ckpt_path,val_mse_train,eps_eval,eps_eval_std,S_eval,S_eval_std,'
                    'KL_eval,KL_eval_std,RHS,RHS_std,ratio,ratio_std,bound_satisfied,num_eval_seeds\n')
            for r in results:
                ratio_val = r['ratio']
                ratio_str = 'inf' if ratio_val == float('inf') else f"{ratio_val}"
                ratio_std = r.get('ratio_std', '')
                if isinstance(ratio_std, float) and not math.isfinite(ratio_std):
                    ratio_std = 'nan'
                f.write(
                    f"{r['epoch']},{r['ckpt_path']},{r['val_mse_train'] or 'N/A'},"
                    f"{r['eps_eval']},{r.get('eps_eval_std', 0.0)},{r['S_eval']},{r.get('S_eval_std', 0.0)},"
                    f"{r['KL_eval']},{r.get('KL_eval_std', 0.0)},{r['RHS']},{r.get('RHS_std', 0.0)},"
                    f"{ratio_str},{ratio_std},{r['bound_satisfied']},{args.eval_num_seeds}\n"
                )
        print(f"Saved CSV to {csv_path}")
        
        # Generate plots
        print("\nGenerating plots...")
        
        # Scatter plot
        scatter_path = base_dir / 'plots' / f'bound_scatter_{args.schedule}_{timestamp}.png'
        plot_bound_verification(results, args.schedule, scatter_path)
        
        # f̂(t) curves
        if fhat_data_list:
            fhat_path = base_dir / 'plots' / f'fhat_curves_{args.schedule}_{timestamp}.png'
            plot_fhat_curves(fhat_data_list, labels, args.schedule, fhat_path)
        
        # ε-curves plot
        if args.make_eps_curves:
            eps_curves_path = base_dir / 'plots' / f'eps_curves_{args.schedule}_{timestamp}.png'
            try:
                plot_lhs_rhs_vs_eps(
                    str(csv_path),
                    str(eps_curves_path),
                    schedule=args.schedule,
                    ylog=True,
                    # annotate_epochs=True,
                    # title=f"KL Error Bound Verification (Learned) - Schedule {args.schedule.upper()}"
                )
            except Exception as e:
                print(f"Warning: Failed to generate ε-curves plot: {e}")
        
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"{'Epoch':<8} {'LHS':<25} {'RHS':<25} {'Ratio':<18} {'Bound'}")
        print("-" * 80)
        for r in results:
            status = "✓" if r['bound_satisfied'] else "✗"
            lhs_display = f"{r['KL_eval']:.6e}"
            if args.eval_num_seeds > 1 and r.get('KL_eval_std', 0.0):
                lhs_display += f" ± {r['KL_eval_std']:.2e}"
            rhs_display = f"{r['RHS']:.6e}"
            if args.eval_num_seeds > 1 and r.get('RHS_std', 0.0):
                rhs_display += f" ± {r['RHS_std']:.2e}"
            ratio_val = r['ratio']
            if ratio_val == float('inf'):
                ratio_display = 'inf'
            else:
                ratio_display = f"{ratio_val:.3f}"
                if args.eval_num_seeds > 1 and isinstance(r.get('ratio_std'), float) and math.isfinite(r['ratio_std']):
                    ratio_display += f" ± {r['ratio_std']:.3f}"
            print(f"{r['epoch']:<8} {lhs_display:<25} {rhs_display:<25} "
                  f"{ratio_display:<18} {status}")
    
    print("\n" + "=" * 80)
    print("Experiment complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

