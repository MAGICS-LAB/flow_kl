"""
No-Learning Test: Closed-form KL identity verification.

Uses two different linear velocity fields (a_u and a_v) to generate
closed-form p_t and q_t distributions. Both LHS and RHS can be computed
analytically, allowing verification of the identity without any learning
or Monte Carlo sampling.

Based on nolearning_test.md.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from true_path import Schedule, get_schedule_functions, schedule_to_enum


def a_u(t):
    """Velocity schedule for p_t."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return torch.sin(np.pi * t)


def A_u(t):
    """Integral of a_u."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return (1 - torch.cos(np.pi * t)) / np.pi


def a_v(t):
    """Velocity schedule for q_t."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return 0.3 * torch.sin(np.pi * t) + 0.2


def A_v(t):
    """Integral of a_v."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return 0.3 * (1 - torch.cos(np.pi * t)) / np.pi + 0.2 * t


def sigma_p_sq(t):
    """σ_p²(t) = e^(2A_u(t))."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return torch.exp(2 * A_u(t))


def sigma_q_sq(t):
    """σ_q²(t) = e^(2A_v(t))."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return torch.exp(2 * A_v(t))


def r_t(t):
    """r(t) = σ_p²(t) / σ_q²(t)."""
    return sigma_p_sq(t) / sigma_q_sq(t)


def kl_lhs_analytic(t):
    """
    LHS: KL(p_t|q_t) = r - 1 - log(r)
    where r = σ_p²/σ_q².
    """
    if isinstance(t, (int, float)):
        t = torch.tensor(t, dtype=torch.float64)
    elif not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    
    r = r_t(t)
    kl = r - 1 - torch.log(r)
    return kl


def g_rhs_analytic(t):
    """
    RHS integrand: g(t) = 2(a_u - a_v)(r - 1)
    where r = σ_p²/σ_q².
    """
    if isinstance(t, (int, float)):
        t = torch.tensor(t, dtype=torch.float64)
    elif not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    
    a_u_val = a_u(t)
    a_v_val = a_v(t)
    r = r_t(t)
    
    g = 2 * (a_u_val - a_v_val) * (r - 1)
    return g


def integrate_trapezoidal(g_values, t_grid):
    """
    Integrate g(t) over time using trapezoidal rule.
    """
    rhs_cumulative = np.zeros_like(g_values)
    
    for m in range(1, len(t_grid)):
        dt = t_grid[m] - t_grid[m-1]
        rhs_cumulative[m] = rhs_cumulative[m-1] + (dt / 2) * (g_values[m] + g_values[m-1])
    
    return rhs_cumulative


def plot_comparison(t_grid, kl_values, rhs_values, save_path=None):
    """Plot LHS vs RHS comparison."""
    plt.figure(figsize=(8, 6))
    # plt.plot(t_grid, kl_values, label='LHS: KL(p_t||q_t)', linewidth=12, color='darkgrey')
    plt.plot(t_grid, kl_values, label='KL Divergence', linewidth=12, color='darkgrey')
    
    # plt.plot(t_grid, rhs_values, label='RHS: ∫₀ᵗ g(s) ds', linewidth=4, linestyle='--', color='darkred')
    plt.plot(t_grid, rhs_values, label='Lemma 3.1 Identity', linewidth=4, linestyle='--', color='darkred')
    
    plt.xlabel('Time t', fontsize=24)
    plt.ylabel('KL Divergence', fontsize=24)
    # plt.title('KL Identity Verification (Closed-Form)', fontsize=21, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=225, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    # plt.show()  # Commented out for automated runs
    plt.close()


def test_nolearning(schedule_p='a1', schedule_q='a2'):
    """
    Run the no-learning test.
    
    Args:
        schedule_p: Schedule for distribution p_t (options: 'a1', 'a2', 'a3')
        schedule_q: Schedule for distribution q_t (options: 'a1', 'a2', 'a3')
    """
    # Convert string to enum and get schedule functions
    schedule_p_enum = schedule_to_enum(schedule_p)
    schedule_q_enum = schedule_to_enum(schedule_q)
    a_p_func, A_p_func = get_schedule_functions(schedule_p_enum)
    a_q_func, A_q_func = get_schedule_functions(schedule_q_enum)
    
    print("=" * 60)
    print("NO-LEARNING TEST: Closed-Form KL Identity Verification")
    print(f"Schedule p (p_t): {schedule_p}")
    print(f"Schedule q (q_t): {schedule_q}")
    print("=" * 60)
    
    # Helper functions using the schedules
    def sigma_p_sq_dyn(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return torch.exp(2 * A_p_func(t))
    
    def sigma_q_sq_dyn(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return torch.exp(2 * A_q_func(t))
    
    # Time grid
    t_grid = np.linspace(0, 1, 201)
    
    # Compute LHS: KL(p_t|q_t) analytically
    print("\n1. Computing LHS (KL divergence) analytically...")
    kl_values = []
    for t_val in t_grid:
        # Compute r(t) = σ_p²/σ_q²
        r_val = (sigma_p_sq_dyn(t_val) / sigma_q_sq_dyn(t_val)).item()
        kl = r_val - 1 - np.log(r_val)
        kl_values.append(kl)
    
    # Compute RHS integrand g(t) analytically
    print("2. Computing RHS integrand g(t) analytically...")
    g_values = []
    for t_val in t_grid:
        # Get velocity values
        a_p_val = a_p_func(t_val).item()
        a_q_val = a_q_func(t_val).item()
        r_val = (sigma_p_sq_dyn(t_val) / sigma_q_sq_dyn(t_val)).item()
        
        # g(t) = 2(a_p - a_q)(r - 1)
        g = 2 * (a_p_val - a_q_val) * (r_val - 1)
        g_values.append(g)
    
    # Integrate RHS over time
    print("3. Integrating RHS over time...")
    rhs_integrated = integrate_trapezoidal(np.array(g_values), t_grid)
    
    # Compute errors
    print("4. Computing errors...")
    errors = np.abs(kl_values - rhs_integrated)
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    # Relative errors (skip t=0 where KL=0)
    rel_errors = errors[1:] / (np.abs(kl_values[1:]) + 1e-10)
    max_rel_error = np.max(rel_errors)
    mean_rel_error = np.mean(rel_errors)
    
    # Find where max relative error occurs
    max_idx = np.argmax(rel_errors)
    
    # Only compute relative error where KL is significant (>1e-6)
    # to avoid near-zero issues
    kl_threshold = 1e-6
    significant_indices = np.where(np.abs(kl_values[1:]) > kl_threshold)[0]
    if len(significant_indices) > 0:
        max_rel_error_significant = np.max(rel_errors[significant_indices])
        mean_rel_error_significant = np.mean(rel_errors[significant_indices])
    else:
        max_rel_error_significant = max_rel_error
        mean_rel_error_significant = mean_rel_error
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Max absolute error: {max_error:.2e}")
    print(f"Mean absolute error: {mean_error:.2e}")
    print(f"Max relative error (all): {max_rel_error*100:.3f}%")
    print(f"Mean relative error (all): {mean_rel_error*100:.3f}%")
    print(f"Max relative error (KL>1e-6): {max_rel_error_significant*100:.3f}%")
    print(f"Mean relative error (KL>1e-6): {mean_rel_error_significant*100:.3f}%")
    print(f"\nMax relative error at t={t_grid[max_idx+1]:.4f}")
    print(f"  KL(t) = {kl_values[max_idx+1]:.6e}")
    print(f"  RHS(t) = {rhs_integrated[max_idx+1]:.6e}")
    print(f"  g(t) = {g_values[max_idx+1]:.6e}")
    
    # Acceptance criteria (focus on significant KL values)
    accepted = max_error < 1e-3 and max_rel_error_significant < 0.05  # 5% threshold for significant KL
    print(f"\nAcceptance: {'PASS' if accepted else 'FAIL'}")
    if not accepted:
        print("Threshold: max relative error ≤ 10%")
    
    print(f"{'='*60}\n")
    
    # Plot comparison
    plot_filename = f'data/plots/nolearning_test_{schedule_p}_{schedule_q}.png'
    plot_comparison(t_grid, kl_values, rhs_integrated, save_path=plot_filename)
    
    return {
        'max_error': max_error,
        'mean_error': mean_error,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'accepted': accepted
    }


def test_derivative_check():
    """Check that KL'(t) matches g(t) using finite differences."""
    print("\n" + "=" * 60)
    print("DERIVATIVE CHECK: KL'(t) vs g(t)")
    print("=" * 60)
    
    # Time grid (avoid boundaries for finite difference)
    t_grid = np.linspace(0.05, 0.95, 19)
    dt = t_grid[1] - t_grid[0]
    
    # Compute KL at each time
    kl_values = [kl_lhs_analytic(t).item() for t in t_grid]
    
    # Finite difference derivative
    kl_derivative = []
    for i in range(1, len(t_grid) - 1):
        kl_dot = (kl_values[i+1] - kl_values[i-1]) / (2 * dt)
        kl_derivative.append(kl_dot)
    
    # Compare with g(t)
    g_values = [g_rhs_analytic(t).item() for t in t_grid[1:-1]]
    
    errors = np.abs(np.array(kl_derivative) - np.array(g_values))
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    print(f"\nMax absolute error: {max_error:.2e}")
    print(f"Mean absolute error: {mean_error:.2e}")
    print(f"Note: Finite difference has error O(dt²), expected ~1e-3 for dt=0.05")
    print(f"Threshold: < 1e-2 (reasonable for finite difference)")
    
    passed = max_error < 1e-2
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    
    return passed


def test_ode_pipeline():
    """
    Optional ODE pipeline validation.
    
    Set v_θ ≡ v and:
    - Compute q_t via backward ODE + divergence
    - Check that log q_t and ∇log q_t match Gaussian formulas
    - Estimate g(t) by Monte Carlo using your usual code
    """
    print("\n" + "=" * 60)
    print("TEST: ODE PIPELINE VALIDATION")
    print("=" * 60)
    
    from eval import log_q_t, score_q_t
    from true_path import sample_p_t, Schedule
    
    # Create a model that returns a_v(t)·x (the velocity for q)
    class QVelocityModel:
        def __call__(self, x, t):
            if isinstance(t, (int, float)):
                t = torch.tensor(t, dtype=torch.float64)
            # v(x,t) = a_v(t)·x
            a_v_val = a_v(t)
            if a_v_val.dim() > 0 and x.dim() > 1:
                a_v_val = a_v_val.unsqueeze(-1)
            return a_v_val * x
    
    model = QVelocityModel()
    
    # Test at different times
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_samples = 1000
    
    print("\n1. Checking log q_t via ODE pipeline:")
    print("   Compare with analytical: log q_t(x) = -log(2π) - |x|²/(2σ_q²)")
    print("   Note: We sample x ~ q_t, compute log q_t(x), and compare with analytical")
    
    log_q_errors = []
    for t_val in t_values:
        # Sample from q_t (Gaussian with σ_q²)
        sigma_q_sq_val = sigma_q_sq(t_val).item()
        z = torch.randn(n_samples, 2, dtype=torch.float64)
        x_samples = z * np.sqrt(sigma_q_sq_val)
        
        # ODE-based log q_t
        log_q_ode = log_q_t(x_samples, t_val, model, Schedule.A1)
        
        # Analytical log q_t
        x_np = x_samples.detach().numpy()
        log_q_analytic = -np.log(2 * np.pi) - np.sum(x_np**2, axis=1) / (2 * sigma_q_sq_val)
        
        error = np.abs(log_q_ode.numpy() - log_q_analytic).mean()
        log_q_errors.append(error)
        print(f"  t={t_val:.1f}: mean error = {error:.6e}")
    
    max_log_error = max(log_q_errors)
    print(f"\n  Max error: {max_log_error:.6e}")
    
    print("\n2. Checking ∇log q_t via ODE pipeline:")
    print("   Compare with analytical: ∇log q_t(x) = -x/σ_q²")
    
    score_errors = []
    for t_val in t_values:
        # Sample from q_t
        sigma_q_sq_val = sigma_q_sq(t_val).item()
        z = torch.randn(n_samples, 2, dtype=torch.float64)
        x_samples = z * np.sqrt(sigma_q_sq_val)
        
        # ODE-based score
        x_grad = x_samples.requires_grad_(True)
        score_q_ode = score_q_t(x_grad, t_val, model, Schedule.A1)
        
        # Analytical score
        score_q_analytic = -x_samples / sigma_q_sq_val
        
        error = torch.norm(score_q_ode - score_q_analytic, dim=-1).mean().item()
        score_errors.append(error)
        print(f"  t={t_val:.1f}: mean error = {error:.6e}")
    
    max_score_error = max(score_errors)
    print(f"\n  Max error: {max_score_error:.6e}")
    
    passed = max_log_error < 1e-3 and max_score_error < 1e-3
    
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"Threshold: < 1e-3 (ODE solver tolerance)")
    print(f"{'='*60}\n")
    
    return passed


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='No-learning KL identity test')
    parser.add_argument('--schedule_p', type=str, default='a1', choices=['a1', 'a2', 'a3'],
                        help='Schedule for distribution p_t (default: a1)')
    parser.add_argument('--schedule_q', type=str, default='a2', choices=['a1', 'a2', 'a3'],
                        help='Schedule for distribution q_t (default: a2)')
    parser.add_argument('--skip_ode', action='store_true',
                        help='Skip ODE pipeline test (faster)')
    args = parser.parse_args()
    
    # Run main test with specified schedules
    results = test_nolearning(schedule_p=args.schedule_p, schedule_q=args.schedule_q)
    
    # Run derivative check
    derivative_passed = test_derivative_check()
    
    # Run ODE pipeline test (optional)
    if not args.skip_ode:
        ode_passed = test_ode_pipeline()
    else:
        ode_passed = None
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"KL identity test: {'PASS' if results['accepted'] else 'FAIL'}")
    print(f"Derivative check: {'PASS' if derivative_passed else 'FAIL'}")
    if ode_passed is not None:
        print(f"ODE pipeline: {'PASS' if ode_passed else 'FAIL'}")
    else:
        print("ODE pipeline: SKIPPED")
    print("=" * 60)

