"""
Utility functions for setting up experiments: seed, device, plotting, etc.
"""

import torch
import numpy as np
import random
import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_str='cpu'):
    """Get torch device from string."""
    if device_str == 'cpu':
        return torch.device('cpu')
    elif device_str.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            print(f"Warning: CUDA not available, using CPU instead")
            return torch.device('cpu')
    else:
        raise ValueError(f"Unknown device: {device_str}")


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    dirs = ['data', 'data/models', 'data/results', 'data/plots', 'data/plot-data']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, path, metadata=None):
    """Save model checkpoint."""
    ensure_dirs()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {'model_state_dict': model.state_dict()}
    if metadata:
        save_dict.update(metadata)
    torch.save(save_dict, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {path}")
    return checkpoint.get('metadata', {})


def plot_comparison(
    t_grid,
    kl_curve,
    rhs_curve,
    schedule,
    save_path=None,
    kl_std=None,
    rhs_std=None,
):
    """Plot LHS (KL) vs RHS (integrated) comparison with optional variance shading."""
    # Compute smoothed curves once
    kl_smooth = smooth_curve(kl_curve, window_size=5)
    rhs_smooth = smooth_curve(rhs_curve, window_size=5)
    
    # Combined plot (two subplots)
    plt.figure(figsize=(12, 5))
    
    # Left subplot: raw curves
    plt.subplot(1, 2, 1)
    if kl_std is not None and np.any(kl_std > 1e-12):
        kl_lower = np.clip(kl_curve - kl_std, a_min=0.0, a_max=None)
        kl_upper = kl_curve + kl_std
        plt.fill_between(
            t_grid,
            kl_lower,
            kl_upper,
            color='darkgrey',
            alpha=0.2,
            label='KL ±1σ',
        )
    if rhs_std is not None and np.any(rhs_std > 1e-12):
        pass  # No shaded variance for RHS per requirement
    plt.plot(t_grid, kl_curve, label='KL Divergence', linewidth=4, alpha=0.9, color='darkgrey')
    plt.plot(t_grid, rhs_curve, label='Lemma 3.1 Identity', linewidth=4, alpha=0.9, linestyle='--', color='darkred')
    plt.xlabel('Time t', fontsize=24)
    plt.ylabel('KL Divergence', fontsize=24)
    plt.title('Raw Curves', fontsize=20, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Right subplot: smoothed curves
    plt.subplot(1, 2, 2)
    # plt.plot(t_grid, kl_smooth, label='LHS (smoothed)', linewidth=4, alpha=0.9, color='darkgrey')
    plt.plot(t_grid, kl_smooth, label='KL Divergence (smoothed)', linewidth=4, alpha=0.9, color='darkgrey')
    # plt.plot(t_grid, rhs_smooth, label='RHS (smoothed)', linewidth=4, alpha=0.9, linestyle='--', color='darkred')
    plt.plot(t_grid, rhs_smooth, label='Lemma 3.1 Identity (smoothed)', linewidth=4, alpha=0.9, linestyle='--', color='darkred')
    plt.xlabel('Time t', fontsize=24)
    plt.ylabel('KL Divergence', fontsize=24)
    plt.title('Smoothed Curves', fontsize=20, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'KL Identity Verification - Schedule {schedule.upper()}', fontsize=21, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        ensure_dirs()
        plt.savefig(save_path, dpi=225, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()
    
    # Individual plots
    if save_path:
        save_path_obj = Path(save_path)
        # Generate filenames for individual plots
        raw_path = save_path_obj.parent / f"{save_path_obj.stem}_raw{save_path_obj.suffix}"
        smoothed_path = save_path_obj.parent / f"{save_path_obj.stem}_smoothed{save_path_obj.suffix}"
        
        # Raw plot
        plt.figure(figsize=(8, 6))
        if kl_std is not None and np.any(kl_std > 1e-12):
            kl_lower = np.clip(kl_curve - kl_std, a_min=0.0, a_max=None)
            kl_upper = kl_curve + kl_std
            plt.fill_between(
                t_grid,
                kl_lower,
                kl_upper,
                color='darkgrey',
                alpha=0.2,
                label='KL ±1σ',
            )
        if rhs_std is not None and np.any(rhs_std > 1e-12):
            pass  # No shaded variance for RHS per requirement
        plt.plot(t_grid, kl_curve, label='KL Divergence', linewidth=4, alpha=0.9, color='darkgrey')
        plt.plot(t_grid, rhs_curve, label='Lemma 3.1 Identity', linewidth=4, alpha=0.9, linestyle='--', color='darkred')
        
        plt.xlabel('Time t', fontsize=24)
        plt.ylabel('KL Divergence', fontsize=24)
        # plt.title(f'KL Identity Verification (Learned) - Schedule {schedule.upper()}', fontsize=21, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(raw_path, dpi=225, bbox_inches='tight')
        print(f"Saved raw plot to {raw_path}")
        plt.close()
        
        # Smoothed plot
        plt.figure(figsize=(8, 6))
        # plt.plot(t_grid, kl_smooth, label='LHS (smoothed)', linewidth=4, alpha=0.9, color='darkgrey')
        plt.plot(t_grid, kl_smooth, label='KL Divergence (smoothed)', linewidth=4, alpha=0.9, color='darkgrey')
        # plt.plot(t_grid, rhs_smooth, label='RHS (smoothed)', linewidth=4, alpha=0.9, linestyle='--', color='darkred')
        plt.plot(t_grid, rhs_smooth, label='Lemma 3.1 Identity (smoothed)', linewidth=4, alpha=0.9, linestyle='--', color='darkred')
        plt.xlabel('Time t', fontsize=24)
        plt.ylabel('KL Divergence', fontsize=24)
        plt.title(f'Smoothed KL Identity Verification - Schedule {schedule.upper()}', fontsize=21, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(smoothed_path, dpi=225, bbox_inches='tight')
        print(f"Saved smoothed plot to {smoothed_path}")
        plt.close()


def smooth_curve(y, window_size=5):
    """
    Apply moving average smoothing to reduce noise in curves.
    
    Args:
        y: Array of values to smooth
        window_size: Size of the moving average window (must be odd)
    
    Returns:
        Smoothed array
    """
    if window_size < 3:
        return y
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    half = window_size // 2
    smoothed = np.zeros_like(y)
    
    # Pad boundaries with edge values
    y_padded = np.pad(y, (half, half), mode='edge')
    
    # Apply moving average
    for i in range(len(y)):
        smoothed[i] = np.mean(y_padded[i:i+window_size])
    
    return smoothed


def compute_relative_error(rhs, kl):
    """Compute relative error between RHS and LHS."""
    # Avoid division by zero
    epsilon = 1e-8
    abs_error = np.abs(rhs - kl)
    rel_error = abs_error / np.maximum(epsilon, np.abs(kl))
    
    return {
        'median': float(np.median(rel_error) * 100),
        'max': float(np.max(rel_error) * 100),
        'mean': float(np.mean(rel_error) * 100),
        'abs_error': abs_error.tolist(),
        'rel_error': rel_error.tolist()
    }


def save_results(
    t_grid,
    kl_curve,
    rhs_integrand,
    rhs_cumulative,
    schedule,
    metadata,
    target_mse=None,
    kl_curve_std=None,
    rhs_integrand_std=None,
    rhs_cumulative_std=None,
):
    """Save experimental results to files."""
    ensure_dirs()
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename suffix with target_mse if provided
    if target_mse is not None:
        mse_str = str(target_mse).replace('.', '-')
        suffix = f"_mse_{mse_str}_{timestamp}"
    else:
        suffix = f"_{timestamp}"
    
    # Save numpy arrays
    results_dir = Path('data/results')
    np.save(results_dir / f't_grid_{schedule}{suffix}.npy', t_grid)
    np.save(results_dir / f'kl_curve_{schedule}{suffix}.npy', kl_curve)
    np.save(results_dir / f'rhs_integrand_{schedule}{suffix}.npy', rhs_integrand)
    np.save(results_dir / f'rhs_cumulative_{schedule}{suffix}.npy', rhs_cumulative)
    if kl_curve_std is not None:
        np.save(results_dir / f'kl_curve_std_{schedule}{suffix}.npy', kl_curve_std)
    if rhs_integrand_std is not None:
        np.save(results_dir / f'rhs_integrand_std_{schedule}{suffix}.npy', rhs_integrand_std)
    if rhs_cumulative_std is not None:
        np.save(results_dir / f'rhs_cumulative_std_{schedule}{suffix}.npy', rhs_cumulative_std)
    
    # Save metadata
    metadata_path = results_dir / f'metadata_{schedule}{suffix}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved results to {results_dir}")
    
    # Plot comparison
    plot_path = Path('data/plots') / f'kl_comparison_{schedule}{suffix}.png'
    plot_comparison(
        t_grid,
        kl_curve,
        rhs_cumulative,
        schedule,
        plot_path,
        kl_std=kl_curve_std,
        rhs_std=rhs_cumulative_std,
    )
    
    # Save plot data (for regeneration)
    plot_data_dir = Path('data/plot-data')
    plot_data = {
        't_grid': t_grid.tolist(),
        'kl_curve': kl_curve.tolist(),
        'rhs_cumulative': rhs_cumulative.tolist(),
        'schedule': schedule,
        'metadata': metadata
    }
    if kl_curve_std is not None:
        plot_data['kl_curve_std'] = kl_curve_std.tolist()
    if rhs_cumulative_std is not None:
        plot_data['rhs_cumulative_std'] = rhs_cumulative_std.tolist()
    plot_data_path = plot_data_dir / f'kl_comparison_{schedule}{suffix}.json'
    with open(plot_data_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"Saved plot data to {plot_data_path}")


if __name__ == '__main__':
    # Test utilities
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print("Utilities module loaded successfully")

