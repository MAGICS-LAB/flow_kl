import torch
import numpy as np
from true_path import Schedule, sample_p_t, velocity_u, log_p_t, sigma_p, score_p_t
from eval import log_q_t, score_q_t, compute_kl_lhs, compute_rhs_integrand
from model import VelocityMLP
import os


def test_R0_trivial_identity():
    """
    R0: Trivial identity check (no training): v = u
    
    Expectation: g(t) ≡ 0 for all t.
    If not 0 within MC error, score or divergence pipeline is still off.
    """
    print("=" * 60)
    print("TEST R0: Trivial Identity Check (v = u)")
    print("=" * 60)
    
    # Create a model that returns u(x, t) = a(t)·x
    class TrueVelocityModel:
        def __call__(self, x, t):
            if isinstance(t, (int, float)):
                t = torch.tensor(t, dtype=torch.float64)
            return velocity_u(x, t, Schedule.A1)
    
    model = TrueVelocityModel()
    schedule = Schedule.A1
    
    # Test at different times
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_samples = 2000
    
    results = []
    for t_val in t_values:
        # Sample from p_t
        x_samples = sample_p_t(t_val, n_samples, schedule)
        
        # Compute RHS integrand
        g_t = compute_rhs_integrand(x_samples, t_val, schedule, model)
        
        results.append((t_val, g_t))
        print(f"t = {t_val:.1f}: g(t) = {g_t:.6e}")
    
    # Check if all g(t) are close to 0
    g_values = [g for _, g in results]
    max_g = max(abs(g) for g in g_values)
    
    # MC error threshold (should be within ~2σ)
    # With N=2000, relative MC error ~ 1/sqrt(2000) ≈ 2.2%
    passed = max_g < 0.1  # Reasonable threshold for MC noise
    
    print(f"\n{'='*60}")
    print(f"Max |g(t)|: {max_g:.6e}")
    print(f"Threshold: < 0.1")
    print(f"{'='*60}")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}\n")
    
    return passed


def test_R1a_zero_velocity():
    """
    R1(a): Analytic check with v ≡ 0 (so q_t = p_0)
    
    Analytic formula: g(t) = d·a(t)·(σ_p²(t) - 1)
    where d=2, σ_p(t) = e^A(t), A(t) = ∫₀ᵗ a(s) ds
    """
    print("=" * 60)
    print("TEST R1(a): Analytic Check (v ≡ 0)")
    print("=" * 60)
    
    # Helper functions for schedule a1
    def a_func(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return torch.sin(np.pi * t)
    
    def A_func(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return (1 - torch.cos(np.pi * t)) / np.pi
    
    # Create a model that returns zero velocity
    class ZeroVelocityModel:
        def __call__(self, x, t):
            # Need to create a tensor that requires grad for divergence computation
            # Use x to maintain proper gradient flow
            return torch.zeros_like(x) * x.sum() * 0.0
    
    model = ZeroVelocityModel()
    schedule = Schedule.A1
    
    # Test at different times
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_samples = 2000
    
    print("Comparing numerical g(t) vs analytic formula:")
    print("g_analytic(t) = d·a(t)·(σ_p²(t) - 1)")
    print()
    
    rel_errors = []
    for t_val in t_values:
        # Sample from p_t
        x_samples = sample_p_t(t_val, n_samples, schedule)
        
        # Numerical g(t)
        g_num = compute_rhs_integrand(x_samples, t_val, schedule, model)
        
        # Analytic g(t)
        t_tensor = torch.tensor(t_val, dtype=torch.float64)
        a_val = a_func(t_val).item()
        A_val = A_func(t_val).item()
        sigma_p_sq = np.exp(2 * A_val)  # σ_p²(t) = e^(2A(t))
        d = 2
        g_analytic = d * a_val * (sigma_p_sq - 1)
        
        rel_error = abs(g_num - g_analytic) / max(abs(g_analytic), 1e-10)
        rel_errors.append(rel_error)
        
        print(f"t = {t_val:.1f}:")
        print(f"  Numerical: {g_num:.6e}")
        print(f"  Analytic:  {g_analytic:.6e}")
        print(f"  Rel error: {rel_error:.4f} ({rel_error*100:.2f}%)")
        print()
    
    # Check if relative errors are within threshold
    max_rel_error = max(rel_errors)
    passed = max_rel_error < 0.05  # 5% threshold
    
    print(f"{'='*60}")
    print(f"Max relative error: {max_rel_error:.4f} ({max_rel_error*100:.2f}%)")
    print(f"Threshold: < 5%")
    print(f"{'='*60}")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}\n")
    
    return passed


def test_R1b_scaled_velocity():
    """
    R1(b): Scaled field (v = c·u) where c = 0.8
    
    With v = c·u, we have q_t = N(0, σ_q²I) where σ_q²(t) = e^(2cA(t))
    and ∇log q_t(x) = -x/σ_q²(t)
    
    Analytic formula: g(t) = (1-c)·a(t)·d·(e^(2(1-c)A(t)) - 1)
    """
    print("=" * 60)
    print("TEST R1(b): Scaled Field Check (v = 0.8·u)")
    print("=" * 60)
    
    c = 0.8  # Scaling factor
    
    # Helper functions for schedule a1
    def a_func(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return torch.sin(np.pi * t)
    
    def A_func(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return (1 - torch.cos(np.pi * t)) / np.pi
    
    # Create a model that returns c·u(x, t)
    class ScaledVelocityModel:
        def __init__(self, c):
            self.c = c
        
        def __call__(self, x, t):
            if isinstance(t, (int, float)):
                t = torch.tensor(t, dtype=torch.float64)
            return self.c * velocity_u(x, t, Schedule.A1)
    
    model = ScaledVelocityModel(c)
    schedule = Schedule.A1
    
    # Test at different times
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_samples = 2000
    
    print(f"Comparing numerical g(t) vs analytic formula (c={c}):")
    print("g_analytic(t) = (1-c)·a(t)·d·(e^(2(1-c)A(t)) - 1)")
    print()
    
    rel_errors = []
    for t_val in t_values:
        # Sample from p_t
        x_samples = sample_p_t(t_val, n_samples, schedule)
        
        # Numerical g(t)
        g_num = compute_rhs_integrand(x_samples, t_val, schedule, model)
        
        # Analytic g(t)
        a_val = a_func(t_val).item()
        A_val = A_func(t_val).item()
        d = 2
        g_analytic = (1 - c) * a_val * d * (np.exp(2 * (1 - c) * A_val) - 1)
        
        rel_error = abs(g_num - g_analytic) / max(abs(g_analytic), 1e-10)
        rel_errors.append(rel_error)
        
        print(f"t = {t_val:.1f}:")
        print(f"  Numerical: {g_num:.6e}")
        print(f"  Analytic:  {g_analytic:.6e}")
        print(f"  Rel error: {rel_error:.4f} ({rel_error*100:.2f}%)")
        print()
    
    # Check if relative errors are within threshold
    max_rel_error = max(rel_errors)
    passed = max_rel_error < 0.05  # 5% threshold
    
    print(f"{'='*60}")
    print(f"Max relative error: {max_rel_error:.4f} ({max_rel_error*100:.2f}%)")
    print(f"Threshold: < 5%")
    print(f"{'='*60}")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}\n")
    
    return passed


def test_R2_derivative_consistency():
    """
    R2: Derivative consistency with LHS (for trained v_θ)
    
    Test: g(t_k) ≈ KL'(t_k) using finite difference
    KL'(t_k) ≈ (KL(t_{k+1}) - KL(t_{k-1})) / (t_{k+1} - t_{k-1})
    """
    print("=" * 60)
    print("TEST R2: Derivative Consistency with LHS")
    print("=" * 60)
    
    # Load trained model
    model_path = "data/models/vtheta_schedule_a1.pth"
    if not os.path.exists(model_path):
        print("WARNING: No trained model found. Skipping R2.")
        print("Run experiment.py first to train a model.")
        print(f"{'='*60}\n")
        return None
    
    model = VelocityMLP(hidden_dims=[128, 128], activation='silu')
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    schedule = Schedule.A1
    
    # Time grid
    t_grid = np.linspace(0.1, 0.9, 17)  # Avoid t=0,1 for finite difference
    n_samples = 2000
    dt = t_grid[1] - t_grid[0]
    
    # Compute KL at each time point
    kl_values = []
    g_values = []
    
    for t_val in t_grid:
        x_samples = sample_p_t(t_val, n_samples, schedule)
        
        # KL at this time
        kl_t = compute_kl_lhs(x_samples, t_val, schedule, model)
        kl_values.append(kl_t)
        
        # RHS integrand at this time
        g_t = compute_rhs_integrand(x_samples, t_val, schedule, model)
        g_values.append(g_t)
    
    kl_values = np.array(kl_values)
    g_values = np.array(g_values)
    
    # Compute finite difference derivative of KL
    # KL'(t_k) ≈ (KL(t_{k+1}) - KL(t_{k-1})) / (2*dt)
    kl_derivative = np.zeros_like(kl_values)
    for i in range(1, len(t_grid) - 1):
        kl_derivative[i] = (kl_values[i+1] - kl_values[i-1]) / (2 * dt)
    
    # Compare g(t) with KL'(t) (skip boundaries)
    inner_idx = np.arange(1, len(t_grid) - 1)
    errors = []
    
    print("Comparing g(t) vs finite-difference KL'(t):")
    print()
    
    for idx in inner_idx:
        g_t = g_values[idx]
        kl_dot = kl_derivative[idx]
        diff = abs(g_t - kl_dot)
        # Relative error
        rel_error = diff / max(abs(kl_dot), 1e-10)
        errors.append(rel_error)
        
        print(f"t = {t_grid[idx]:.2f}:")
        print(f"  g(t) = {g_t:.6f}")
        print(f"  KL'(t) = {kl_dot:.6f}")
        print(f"  Diff = {diff:.6f} ({rel_error*100:.2f}%)")
        print()
    
    max_rel_error = max(errors) if errors else 1.0
    passed = max_rel_error < 0.15  # 15% threshold (relaxed for finite difference)
    
    print(f"{'='*60}")
    print(f"Max relative error: {max_rel_error:.4f} ({max_rel_error*100:.2f}%)")
    print(f"Threshold: < 15%")
    print(f"{'='*60}")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}\n")
    
    return passed


def test_R3_decomposition_sanity():
    """
    R3: Internal decomposition sanity
    
    Check: g(t) = G_p(t) - G_q(t) where
    G_p(t) = E[(u-v)ᵀ∇log p_t]
    G_q(t) = E[(u-v)ᵀ∇log q_t]
    """
    print("=" * 60)
    print("TEST R3: Internal Decomposition Sanity")
    print("=" * 60)
    
    # Load trained model
    model_path = "data/models/vtheta_schedule_a1.pth"
    if not os.path.exists(model_path):
        print("WARNING: No trained model found. Skipping R3.")
        print("Run experiment.py first to train a model.")
        print(f"{'='*60}\n")
        return None
    
    model = VelocityMLP(hidden_dims=[128, 128], activation='silu')
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    schedule = Schedule.A1
    
    # Test at different times
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_samples = 2000
    
    print("Checking g(t) = G_p(t) - G_q(t):")
    print()
    
    errors = []
    for t_val in t_values:
        # Sample from p_t
        x_samples = sample_p_t(t_val, n_samples, schedule)
        
        # Compute components manually
        u = velocity_u(x_samples, t_val, schedule)
        v = model(x_samples, t_val)
        s_p = score_p_t(x_samples, t_val, schedule)
        s_q = score_q_t(x_samples, t_val, model, schedule)
        
        # G_p(t) = E[(u-v)ᵀ∇log p_t]
        inner_p = torch.sum((u - v) * s_p, dim=-1)
        G_p = torch.mean(inner_p).item()
        
        # G_q(t) = E[(u-v)ᵀ∇log q_t]
        inner_q = torch.sum((u - v) * s_q, dim=-1)
        G_q = torch.mean(inner_q).item()
        
        # g(t) from decomposition
        g_decomp = G_p - G_q
        
        # g(t) from direct computation
        inner_prod = torch.sum((u - v) * (s_p - s_q), dim=-1)
        g_direct = torch.mean(inner_prod).item()
        
        # Check difference (should be machine precision)
        diff = abs(g_decomp - g_direct)
        errors.append(diff)
        
        print(f"t = {t_val:.1f}:")
        print(f"  G_p(t) = {G_p:.6f}")
        print(f"  G_q(t) = {G_q:.6f}")
        print(f"  g(t) = G_p - G_q = {g_decomp:.6f}")
        print(f"  g(t) (direct) = {g_direct:.6f}")
        print(f"  |Diff| = {diff:.2e}")
        print()
    
    max_error = max(errors)
    passed = max_error < 1e-10  # Machine precision
    
    print(f"{'='*60}")
    print(f"Max absolute error: {max_error:.2e}")
    print(f"Threshold: < 1e-10 (machine precision)")
    print(f"{'='*60}")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}\n")
    
    return passed


if __name__ == '__main__':
    print("\n" + "="*60)
    print("RHS EVALUATION TESTS")
    print("="*60)
    
    # Run tests
    results = {}
    results['R0'] = test_R0_trivial_identity()
    results['R1a'] = test_R1a_zero_velocity()
    results['R1b'] = test_R1b_scaled_velocity()
    results['R2'] = test_R2_derivative_consistency()
    results['R3'] = test_R3_decomposition_sanity()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        if passed is not None:
            status = "PASS" if passed else "FAIL"
            print(f"{test_name}: {status}")
    
    all_passed = all(p for p in results.values() if p is not None)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60)

