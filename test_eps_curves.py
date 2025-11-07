"""
Test script for ε-curves plotting functionality.

Tests the plot_eps_curves module using existing CSV files or generates a test CSV.
"""

import os
import sys
from pathlib import Path
from plot_eps_curves import plot_lhs_rhs_vs_eps


def test_with_existing_csv():
    """Test plotting with an existing CSV file."""
    # Use the most recent CSV for schedule a1
    csv_path = Path('data/part-2-learn/a1/results/bound_a1_20251031_031129.csv')
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print("Please run a Part 2 Learning experiment first or use test_with_dummy_csv()")
        return False
    
    print(f"Testing with existing CSV: {csv_path}")
    print(f"CSV exists: {csv_path.exists()}")
    
    # Read and check CSV contents
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"\nCSV columns: {list(df.columns)}")
    print(f"Number of rows: {len(df)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Test plotting
    output_path = Path('data/part-2-learn/a1/plots/test_eps_curves_a1.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        plot_lhs_rhs_vs_eps(
            str(csv_path),
            str(output_path),
            schedule='a1',
            ylog=True,
            annotate_epochs=True
        )
        
        if output_path.exists():
            print(f"\n✓ Success! Plot generated: {output_path}")
            print(f"  File size: {output_path.stat().st_size} bytes")
            return True
        else:
            print(f"\n✗ Error: Plot file was not created")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_dummy_csv():
    """Test plotting with a dummy CSV file (for testing without existing data)."""
    import pandas as pd
    import numpy as np
    
    # Create dummy data
    np.random.seed(42)
    n_checkpoints = 10
    
    # Generate realistic-looking data
    epochs = np.arange(0, n_checkpoints * 10, 10)
    eps_vals = np.linspace(0.5, 0.05, n_checkpoints)  # Decreasing epsilon
    S_vals = np.linspace(2.0, 0.1, n_checkpoints)  # Decreasing S
    KL_vals = eps_vals * np.sqrt(S_vals) * (1 + np.random.normal(0, 0.1, n_checkpoints))  # LHS ~ RHS with noise
    RHS_vals = eps_vals * np.sqrt(S_vals)
    
    # Create DataFrame
    df = pd.DataFrame({
        'epoch': epochs,
        'eps_eval': eps_vals,
        'S_eval': S_vals,
        'KL_eval': np.maximum(KL_vals, 0.001),  # Ensure positive
        'RHS': RHS_vals,
        'ratio': KL_vals / RHS_vals,
        'bound_satisfied': KL_vals <= RHS_vals
    })
    
    # Save dummy CSV
    dummy_csv = Path('test_dummy_results.csv')
    df.to_csv(dummy_csv, index=False)
    print(f"Created dummy CSV: {dummy_csv}")
    print(f"\nDummy data preview:")
    print(df.head())
    
    # Test plotting
    output_path = Path('test_eps_curves_dummy.png')
    
    try:
        plot_lhs_rhs_vs_eps(
            str(dummy_csv),
            str(output_path),
            schedule='a1',
            ylog=True,
            annotate_epochs=True
        )
        
        if output_path.exists():
            print(f"\n✓ Success! Plot generated: {output_path}")
            print(f"  File size: {output_path.stat().st_size} bytes")
            
            # Cleanup
            dummy_csv.unlink()
            print(f"  Cleaned up: {dummy_csv}")
            
            return True
        else:
            print(f"\n✗ Error: Plot file was not created")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup on error
        if dummy_csv.exists():
            dummy_csv.unlink()
        return False


def main():
    """Run tests."""
    print("=" * 80)
    print("Testing ε-curves plotting functionality")
    print("=" * 80)
    
    # Try with existing CSV first
    if Path('data/part-2-learn/a1/results/bound_a1_20251031_031129.csv').exists():
        print("\n[Test 1] Testing with existing CSV file...")
        success1 = test_with_existing_csv()
    else:
        print("\n[Test 1] Skipping (no existing CSV found)")
        success1 = None
    
    # Always test with dummy CSV
    print("\n[Test 2] Testing with dummy CSV file...")
    success2 = test_with_dummy_csv()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    if success1 is not None:
        print(f"Test 1 (existing CSV): {'✓ PASSED' if success1 else '✗ FAILED'}")
    else:
        print("Test 1 (existing CSV): SKIPPED")
    print(f"Test 2 (dummy CSV): {'✓ PASSED' if success2 else '✗ FAILED'}")
    
    if (success1 is None or success1) and success2:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

