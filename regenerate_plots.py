"""
Regenerate plots from saved plot data.

Usage:
    python regenerate_plots.py
    
This will regenerate all plots from the data/plot-data directory.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from utils import smooth_curve, plot_comparison, ensure_dirs


def regenerate_from_data(plot_data_path, plot_data_dir, output_dir):
    """Regenerate a plot from saved data."""
    with open(plot_data_path, 'r') as f:
        data = json.load(f)
    
    t_grid = np.array(data['t_grid'])
    kl_curve = np.array(data['kl_curve'])
    rhs_cumulative = np.array(data['rhs_cumulative'])
    kl_std = np.array(data['kl_curve_std']) if 'kl_curve_std' in data else None
    rhs_std = np.array(data['rhs_cumulative_std']) if 'rhs_cumulative_std' in data else None
    schedule = data['schedule']
    
    # Generate plot filename from data filename, preserving directory structure
    plot_data_path = Path(plot_data_path)
    
    # Get relative path from plot_data_dir to preserve structure
    relative_path = plot_data_path.relative_to(plot_data_dir)
    
    # Change .json to .png and maintain directory structure
    plot_path = output_dir / relative_path.with_suffix('.png')
    
    # Create output directory if it doesn't exist
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Regenerate plot
    plot_comparison(
        t_grid,
        kl_curve,
        rhs_cumulative,
        schedule,
        plot_path,
        kl_std=kl_std,
        rhs_std=rhs_std,
    )
    
    return plot_path


def main():
    """Regenerate all plots from plot-data directory."""
    ensure_dirs()
    
    plot_data_dir = Path('data/plot-data')
    
    if not plot_data_dir.exists():
        print("No plot-data directory found!")
        return
    
    # Use rglob to recursively search for JSON files
    plot_data_files = list(plot_data_dir.rglob('kl_comparison_*.json'))
    
    if not plot_data_files:
        print("No plot data files found!")
        return
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'data/regen_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Found {len(plot_data_files)} plot data files to regenerate\n")
    
    for plot_data_path in plot_data_files:
        print(f"Regenerating: {plot_data_path.relative_to(plot_data_dir)}")
        plot_path = regenerate_from_data(plot_data_path, plot_data_dir, output_dir)
        print(f"  Saved to: {plot_path}\n")
    
    print(f"Regenerated {len(plot_data_files)} plots")


if __name__ == '__main__':
    main()

