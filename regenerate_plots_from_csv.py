"""
Regenerate plots from a cleaned CSV file.

Generates both the scatter plot and ε-curves plot from Part 2 Learning or Synthetic CSV data.
"""

import sys
import pandas as pd
from pathlib import Path
from plot_eps_curves import plot_lhs_rhs_vs_eps
import matplotlib.pyplot as plt


def plot_bound_verification_from_csv(df, schedule, save_path):
    """
    Generate scatter plot from CSV data (works with both learned and synthetic formats).
    
    Args:
        df: DataFrame with columns (learned: epoch, eps_eval, KL_eval, RHS; synthetic: delta_label, epsilon_hat, KL_hat, RHS)
        schedule: Schedule name
        save_path: Path to save plot
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Auto-detect format (learned vs synthetic)
    lhs_std = None
    rhs_std = None

    if 'KL_eval' in df.columns and 'eps_eval' in df.columns:
        # Learned format
        lhs_col = 'KL_eval'
        eps_col = 'eps_eval'
        label_col = 'epoch'
        if 'KL_eval_std' in df.columns:
            lhs_std = df['KL_eval_std'].values
        if 'RHS_std' in df.columns:
            rhs_std = df['RHS_std'].values
    elif 'KL_hat' in df.columns and 'epsilon_hat' in df.columns:
        # Synthetic format
        lhs_col = 'KL_hat'
        eps_col = 'epsilon_hat'
        label_col = 'delta_label'
        if 'KL_hat_std' in df.columns:
            lhs_std = df['KL_hat_std'].values
        if 'RHS_std' in df.columns:
            rhs_std = df['RHS_std'].values
    else:
        raise ValueError("Could not identify CSV format. Expected either learned (eps_eval, KL_eval) or synthetic (epsilon_hat, KL_hat) columns.")
    
    # Extract data
    lhs_list = df[lhs_col].values
    if 'RHS' in df.columns:
        rhs_list = df['RHS'].values
    else:
        # Compute RHS if not present
        if label_col == 'epoch':
            s_col = 'S_eval'
        else:
            s_col = 'S_hat'
        if s_col in df.columns:
            rhs_list = df[eps_col].values * (df[s_col].values ** 0.5)
        else:
            raise ValueError(f"Could not compute RHS: missing 'RHS' or '{s_col}' column")
    
    # Get labels for annotations
    if label_col in df.columns:
        labels = df[label_col].values
    else:
        labels = None
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    if labels is not None and label_col == 'epoch':
        # Use color coding for epochs (learned)
        scatter = plt.scatter(rhs_list, lhs_list, s=100, alpha=0.7, c=labels, 
                             cmap='viridis', label='Checkpoints', zorder=3)
        plt.colorbar(scatter, label='Epoch')
        # Add epoch annotations
        for i, (rhs, lhs, label) in enumerate(zip(rhs_list, lhs_list, labels)):
            plt.annotate(f"E{int(label)}", (rhs, lhs), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    else:
        # Simple scatter for synthetic (no color coding)
        plt.scatter(rhs_list, lhs_list, s=100, alpha=0.7, label='Experiments', zorder=3)
        # Add delta label annotations if available
        if labels is not None:
            for rhs, lhs, label in zip(rhs_list, lhs_list, labels):
                plt.annotate(str(label), (rhs, lhs), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

    if lhs_std is not None or rhs_std is not None:
        plt.errorbar(
            rhs_list,
            lhs_list,
            xerr=rhs_std if rhs_std is not None else None,
            yerr=lhs_std if lhs_std is not None else None,
            fmt='none',
            ecolor='black',
            alpha=0.4,
            capsize=5,
            zorder=2,
        )
    
    # Reference line y=x
    max_rhs = rhs_list.max() if len(rhs_list) > 0 else 0
    max_lhs = lhs_list.max() if len(lhs_list) > 0 else 0
    max_val = max(max_rhs, max_lhs, 1e-6) * 1.1
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (bound)', zorder=1)
    
    plt.xlabel('RHS = ε_θ√S_θ', fontsize=12)
    plt.ylabel('LHS = KL(p₁|q₁^θ)', fontsize=12)
    plt.title(f'Bound Verification: KL(p₁|q₁^θ) ≤ ε_θ√S_θ\nSchedule {schedule.upper()}', 
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved scatter plot to {save_path}")
    plt.close()


def regenerate_plots(csv_path, schedule=None, output_dir=None):
    """
    Regenerate both scatter and ε-curves plots from a CSV file.
    
    Args:
        csv_path: Path to CSV file
        schedule: Schedule name (if None, try to infer from filename or CSV)
        output_dir: Output directory for plots (if None, use CSV directory)
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return False
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {csv_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Infer schedule from filename or CSV if not provided
    if schedule is None:
        # Try to extract from filename (e.g., bound_a1_*.csv)
        filename_parts = csv_path.stem.split('_')
        if 'a1' in filename_parts or 'a2' in filename_parts or 'a3' in filename_parts:
            for part in filename_parts:
                if part in ['a1', 'a2', 'a3']:
                    schedule = part
                    break
        
        # Fallback: check CSV for schedule column
        if schedule is None and 'schedule' in df.columns:
            schedule = df['schedule'].iloc[0] if len(df) > 0 else None
        
        if schedule is None:
            print("Error: Could not determine schedule. Please specify --schedule")
            return False
    
    print(f"  Schedule: {schedule}")
    
    # Determine output directory
    if output_dir is None:
        # Try to use plots directory near the CSV
        csv_dir = csv_path.parent
        if csv_dir.name == 'results':
            output_dir = csv_dir.parent / 'plots'
        else:
            output_dir = csv_dir / 'plots'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp suffix from filename (extract YYYYMMDD_HHMMSS pattern)
    import re
    timestamp_match = re.search(r'(\d{8}_\d{6})', csv_path.stem)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
    else:
        # Fallback: try to extract from last part of filename
        parts = csv_path.stem.split('_')
        timestamp = None
        # Look for timestamp pattern in filename parts
        for part in reversed(parts):
            if len(part) == 15 and part.replace('_', '').isdigit():
                timestamp = part
                break
        if timestamp is None:
            # Last resort: use current time
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate scatter plot
    scatter_path = output_dir / f'bound_scatter_{schedule}_{timestamp}.png'
    print(f"\nGenerating scatter plot...")
    plot_bound_verification_from_csv(df, schedule, scatter_path)
    
    # Generate ε-curves plot
    eps_curves_path = output_dir / f'eps_curves_{schedule}_{timestamp}.png'
    print(f"\nGenerating ε-curves plot...")
    
    # Determine if this is a learned or closed-form (synthetic) plot
    # Check the CSV path or column names to determine the type
    csv_path_str = str(csv_path)
    is_learned = 'part-2-learn' in csv_path_str or ('eps_eval' in df.columns and 'KL_eval' in df.columns)
    
    if is_learned:
        title = f"KL Error Bound Verification (Learned) - Schedule {schedule.upper()}"
    else:
        title = f"KL Error Bound Verification (Closed-Form) - Schedule {schedule.upper()}"
    
    try:
        plot_lhs_rhs_vs_eps(
            str(csv_path),
            str(eps_curves_path),
            schedule=schedule,
            ylog=True,
            annotate=False,
            title=title
        )
        print(f"✓ Successfully generated both plots!")
        print(f"  Scatter plot: {scatter_path}")
        print(f"  ε-curves plot: {eps_curves_path}")
        return True
    except Exception as e:
        print(f"✗ Error generating ε-curves plot: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Regenerate plots from Part 2 Learning CSV')
    parser.add_argument('csv_path', type=str, help='Path to CSV file')
    parser.add_argument('--schedule', type=str, choices=['a1', 'a2', 'a3'], default=None,
                        help='Schedule name (auto-detected from filename if not provided)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: plots/ directory near CSV)')
    
    args = parser.parse_args()
    
    success = regenerate_plots(args.csv_path, args.schedule, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

