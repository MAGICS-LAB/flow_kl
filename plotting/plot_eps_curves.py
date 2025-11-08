"""
Plotting utility for Part 2 (Learning and Synthetic) ε-curves.

Generates curves showing LHS (KL) and RHS (ε√S) vs ε (RMS flow-matching loss).
Works with both learned (Part 2 Learning) and synthetic (Part 2) CSVs.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def plot_lhs_rhs_vs_eps(
    results_csv: str,
    out_png: str,
    schedule: str,
    ylog: bool = True,
    annotate_epochs: bool = True,
    annotate: bool = None,
    title: str = None,
):
    """
    Make the 'curves vs epsilon' plot:
      - x: eps (eval-time RMS flow-matching loss)
      - y1: KL(p1 || q1)  (LHS)
      - y2: eps * sqrt(S) (RHS)
    One line per quantity; points are checkpoints sorted by eps.
    
    Supports both learned CSVs (with eps_eval, KL_eval, S_eval) and 
    synthetic CSVs (with epsilon_hat, KL_hat, S_hat).
    
    Args:
        results_csv: Path to CSV file with results
        out_png: Path to save output PNG
        schedule: Schedule name (for filtering and title)
        ylog: Whether to use log scale for y-axis
        annotate_epochs: Whether to annotate points with epoch numbers (learned) - deprecated, use annotate
        annotate: Whether to annotate points (auto-detects epoch for learned, delta_label for synthetic)
        title: Optional custom title (default: auto-generated)
    """
    # Backward compatibility: if annotate not provided, use annotate_epochs
    if annotate is None:
        annotate = annotate_epochs
    
    # Read CSV
    df = pd.read_csv(results_csv)
    
    # Filter by schedule if 'schedule' column exists
    if 'schedule' in df.columns:
        df = df[df['schedule'] == schedule].copy()
    
    if df.empty:
        raise ValueError(f"No rows for schedule={schedule} in {results_csv}")

    # Auto-detect column names (learned vs synthetic)
    # Learned: eps_eval, KL_eval, S_eval
    # Synthetic: epsilon_hat, KL_hat, S_hat
    if 'eps_eval' in df.columns and 'KL_eval' in df.columns and 'S_eval' in df.columns:
        # Learned format
        eps_col = 'eps_eval'
        kl_col = 'KL_eval'
        s_col = 'S_eval'
        label_col = 'epoch' if 'epoch' in df.columns else None
    elif 'epsilon_hat' in df.columns and 'KL_hat' in df.columns and 'S_hat' in df.columns:
        # Synthetic format
        eps_col = 'epsilon_hat'
        kl_col = 'KL_hat'
        s_col = 'S_hat'
        label_col = 'delta_label' if 'delta_label' in df.columns else None
    else:
        # Try to find any combination
        eps_cols = [c for c in df.columns if 'eps' in c.lower()]
        kl_cols = [c for c in df.columns if 'kl' in c.lower()]
        s_cols = [c for c in df.columns if c.lower() in ['s', 's_eval', 's_hat']]
        
        if not eps_cols or not kl_cols or not s_cols:
            raise ValueError(f"Could not identify required columns in {results_csv}. "
                           f"Expected either (eps_eval, KL_eval, S_eval) for learned or "
                           f"(epsilon_hat, KL_hat, S_hat) for synthetic.")
        
        eps_col = eps_cols[0]
        kl_col = kl_cols[0]
        s_col = s_cols[0]
        label_col = None

    # Compute RHS if not present
    if 'RHS' not in df.columns:
        df['RHS'] = df[eps_col] * (df[s_col] ** 0.5)

    # Sort by epsilon (stable sort)
    df.sort_values(eps_col, inplace=True, kind='mergesort')

    # Extract data
    x = df[eps_col].values
    y_lhs = df[kl_col].values
    y_rhs = df['RHS'].values

    eps_std = None
    lhs_std = None
    rhs_std = None

    std_column_candidates = [
        ('epsilon_hat_std', 'KL_hat_std', 'RHS_std'),
        ('eps_eval_std', 'KL_eval_std', 'RHS_std'),
        ('epsilon_std', 'KL_std', 'RHS_std'),
    ]

    for eps_std_col, lhs_std_col, rhs_std_col in std_column_candidates:
        if eps_std is None and eps_std_col in df.columns:
            eps_std = df[eps_std_col].values
        if lhs_std is None and lhs_std_col in df.columns:
            lhs_std = df[lhs_std_col].values
        if rhs_std is None and rhs_std_col in df.columns:
            rhs_std = df[rhs_std_col].values

    # Handle zero or very small epsilon values for log scale
    # Replace zero with a small value relative to the minimum non-zero epsilon
    min_nonzero_eps = x[x > 0].min() if (x > 0).any() else 1e-6
    x_plot = x.copy()
    x_plot[x_plot == 0] = min_nonzero_eps * 0.1  # Place zero points at 10% of min non-zero

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_lhs, marker='o', linestyle='-', label='KL Divergence', 
            linewidth=4, alpha=0.9, color='darkgrey')
    plt.plot(x_plot, y_rhs, marker='s', linestyle='-', label='Theorem 3.1 Bound', 
            linewidth=4, alpha=0.9, color='darkred')

    if lhs_std is not None and np.any(lhs_std > 1e-12):
        lhs_lower = np.clip(y_lhs - lhs_std, a_min=0.0, a_max=None)
        lhs_upper = y_lhs + lhs_std
        plt.fill_between(
            x_plot,
            lhs_lower,
            lhs_upper,
            color='darkgrey',
            alpha=0.18,
            label='KL ±1σ',
        )
    # Do not plot variance shading for RHS (per request)

    # Optionally annotate points (use x_plot for positioning)
    if annotate and label_col is not None and label_col in df.columns:
        labels = df[label_col].astype(str).values
        for xi, xi_plot, yi, lab in zip(x, x_plot, y_lhs, labels):
            # For epochs, convert to int string; for delta_label, use as-is
            if label_col == 'epoch':
                lab_str = str(int(float(lab))) if lab.replace('.', '').replace('-', '').isdigit() else str(lab)
            else:
                lab_str = str(lab)
            plt.annotate(lab_str, (xi_plot, yi), xytext=(3, 3), 
                       textcoords='offset points', fontsize=8)

    # Set log scales first (before setting tick params)
    plt.xscale('log')  # x-axis on log scale
    if ylog:
        plt.yscale('log')  # y-axis on log scale (if enabled)
    
    plt.xlabel('ε (RMS flow-matching loss)', fontsize=24)
    plt.ylabel('Value', fontsize=24)
    ttl = title or f"KL Error Bound Verification (Closed-Form) - Schedule {schedule.upper()}"
    # plt.title(ttl, fontsize=21, fontweight='bold')
    
    # Set tick parameters - ensure consistent font size for all tick labels
    ax = plt.gca()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    
    # Get tick positions first by forcing a draw
    plt.draw()
    
    # Get major and minor tick positions (actual positions matplotlib has placed)
    x_major_ticks = set(ax.get_xticks())
    y_major_ticks = set(ax.get_yticks()) if ylog else set()
    
    # Also get the actual minor tick locations that matplotlib has placed
    x_minor_locs = ax.xaxis.get_minorticklocs()
    y_minor_locs = ax.yaxis.get_minorticklocs() if ylog else []
    
    # Get all tick locations (major + minor) from the locator
    x_locator = ax.xaxis.get_major_locator()
    y_locator = ax.yaxis.get_major_locator() if ylog else None
    
    # For log scales, we need to get minor ticks differently
    # Get all tick positions from the view limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim() if ylog else (0, 1)
    
    # Generate minor tick positions for log scale
    # Minor ticks are typically at 2, 3, 4, 5, 6, 7, 8, 9 times each power of 10
    def get_log_minor_ticks(vmin, vmax):
        """Get minor tick positions for log scale."""
        if vmin <= 0 or vmax <= 0:
            return []
        min_exp = int(np.floor(np.log10(vmin)))
        max_exp = int(np.ceil(np.log10(vmax)))
        minor_ticks = []
        for exp in range(min_exp, max_exp + 1):
            for mult in [2, 3, 4, 5, 6, 7, 8, 9]:
                tick_val = mult * (10 ** exp)
                if vmin <= tick_val <= vmax:
                    minor_ticks.append(tick_val)
        return minor_ticks
    
    # Use matplotlib's actual minor tick locations if available, otherwise generate them
    if len(x_minor_locs) > 0:
        x_minor_ticks = [t for t in x_minor_locs if t not in x_major_ticks]
    else:
        x_minor_ticks = [t for t in get_log_minor_ticks(x_min, x_max) if t not in x_major_ticks]
    
    if ylog:
        if len(y_minor_locs) > 0:
            y_minor_ticks = [t for t in y_minor_locs if t not in y_major_ticks]
        else:
            y_minor_ticks = [t for t in get_log_minor_ticks(y_min, y_max) if t not in y_major_ticks]
    else:
        y_minor_ticks = []
    
    # Select every third minor tick for labeling
    x_minor_ticks_to_label = set(x_minor_ticks[::3]) if len(x_minor_ticks) > 0 else set()  # Every third minor tick
    y_minor_ticks_to_label = set(y_minor_ticks[::3]) if ylog and len(y_minor_ticks) > 0 else set()
    
    # Always include the rightmost (largest) x-axis tick
    # Get all possible ticks (major and minor) that matplotlib has actually placed
    all_x_ticks = list(x_major_ticks) + x_minor_ticks
    # Find the rightmost tick (closest to x_max but not exceeding it)
    if len(all_x_ticks) > 0:
        # Filter to ticks within the visible range
        visible_ticks = [t for t in all_x_ticks if x_min <= t <= x_max]
        if len(visible_ticks) > 0:
            x_max_tick = max(visible_ticks)
        else:
            # If no ticks in range, use the maximum tick
            x_max_tick = max(all_x_ticks)
    else:
        x_max_tick = x_max
    
    # Always add the rightmost tick to the set of ticks to label
    x_minor_ticks_to_label.add(x_max_tick)
    
    # Use E notation for scientific notation (e.g., "6e-2" instead of "6×10⁻²")
    def make_e_notation_formatter(ticks_to_label):
        """Create a formatter that shows labels only for specified ticks."""
        def formatter(x, pos):
            # Check if this tick is in our list (with tolerance for floating point)
            # Find the closest tick in our list
            if len(ticks_to_label) == 0:
                return ''
            
            closest_tick = min(ticks_to_label, key=lambda t: abs(t - x))
            # Use relative tolerance for comparison
            if abs(x) > 0:
                rel_tol = abs((closest_tick - x) / x)
            else:
                rel_tol = abs(closest_tick - x)
            
            if rel_tol > 1e-6:  # Not close enough to any tick we want to label
                return ''
            
            # Format the label
            if abs(x) < 1e-10:  # Handle very small numbers
                return '0'
            # Get the exponent
            exp = int(np.floor(np.log10(abs(x))))
            # Get the mantissa
            mantissa = x / (10 ** exp)
            # Format: mantissa with appropriate precision, then 'e' and exponent
            if abs(mantissa) >= 1:
                return f'{mantissa:.0f}e{exp}'
            else:
                return f'{mantissa:.1f}e{exp}'
        return formatter
    
    # Create formatters
    x_minor_formatter = FuncFormatter(make_e_notation_formatter(x_minor_ticks_to_label))
    y_minor_formatter = FuncFormatter(make_e_notation_formatter(y_minor_ticks_to_label)) if ylog else None
    
    # Hide major tick labels, show labels on every third minor tick
    # But always show the rightmost x-axis tick if it's a major tick
    def make_major_formatter_with_max_tick(max_tick, ticks_to_label):
        """Create a major formatter that shows label only for the rightmost tick."""
        # Reuse the same formatter logic
        e_formatter = make_e_notation_formatter(ticks_to_label)
        def formatter(x, pos):
            # Check if this is the rightmost tick (with tolerance)
            # Use a more lenient tolerance
            if abs(x) > 0:
                rel_tol = abs((max_tick - x) / x)
            else:
                rel_tol = abs(max_tick - x)
            
            # Use a more lenient tolerance (1% relative or absolute)
            if rel_tol < 0.01 or abs(max_tick - x) < 1e-10:  # This is the rightmost tick
                # Use the same formatting as minor ticks
                return e_formatter(x, pos)
            return ''  # Hide all other major ticks
        return formatter
    
    x_major_formatter = make_major_formatter_with_max_tick(x_max_tick, x_minor_ticks_to_label)
    ax.xaxis.set_major_formatter(x_major_formatter)
    ax.xaxis.set_minor_formatter(x_minor_formatter)  # Labels on every third minor tick
    if ylog:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: ''))  # No labels for major ticks
        ax.yaxis.set_minor_formatter(y_minor_formatter)  # Labels on every third minor tick
    
    # Disable offset text (the "×10⁻²" that appears separately)
    ax.xaxis.offsetText.set_visible(False)
    if ylog:
        ax.yaxis.offsetText.set_visible(False)
    
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=16)

    # Save
    out_dir = os.path.dirname(out_png)
    if out_dir:  # Only create directory if path contains a directory
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=225, bbox_inches='tight')
    plt.close()
    print(f"Saved ε-curves plot to {out_png}")

