"""
Golden-path tests for LHS evaluation pipeline (A1-A3).

These tests use v(x,t) = u(x,t) = a(t)x (the true velocity) to isolate
the backward ODE, divergence accumulation, and density computation
from the learned model.

Based on troubleshoot_lhs.md tests A1-A3.

Test Summary:
- A1: Verifies backward ODE correctly computes preimage x_0 = exp(-A(t))x
- A2: Verifies divergence integral ℓ(t) = 2·A(t) with correct sign
- A3: Verifies log q_t(x) = log p_t(x) when v = u
- B1: Verifies normalization M(t) = E[exp(log q_t - log p_t)] ≈ 1
"""

import torch
import numpy as np
from true_path import Schedule, sample_p_t, velocity_u, log_p_t, sigma_p
from eval import backward_ode_and_divergence, log_q_t
from torchdiffeq import odeint

def test_A1_preimage():
    """
    A1) Preimage (backward ODE) is correct.
    
    Test: Solve backward ODE with v = u (true velocity)
    Expected: x_0 = exp(-A(t)) x (for linear velocity field)
    """
    print("=" * 60)
    print("TEST A1: Preimage (Backward ODE)")
    print("=" * 60)
    
    # Use schedule a1: a(t) = sin(πt), A(t) = (1-cos(πt))/π
    def a_func(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return torch.sin(np.pi * t)
    
    def A_func(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return (1 - torch.cos(np.pi * t)) / np.pi
    
    # True velocity function (identity)
    def u(x, t):
        a_val = a_func(t)
        # Handle broadcasting
        if a_val.dim() > 0 and x.dim() > 1:
            a_val = a_val.unsqueeze(-1)
        return a_val * x
    
    # Create a dummy model that returns u(x,t)
    class TrueVelocityModel:
        def __call__(self, x, t):
            # Convert t to tensor if it's a scalar
            if isinstance(t, (int, float)):
                t = torch.tensor(t, dtype=torch.float64)
            return u(x, t)
    
    model = TrueVelocityModel()
    schedule = Schedule.A1
    
    # Test on random (x, t)
    errors = []
    for i in range(5):
        t = torch.tensor(np.random.rand(), dtype=torch.float64)
        x = torch.randn(10, 2, dtype=torch.float64)
        
        # Solve backward ODE
        x_0_computed, _ = backward_ode_and_divergence(model, x, t.item())
        
        # Analytic solution: x_0^* = exp(-A(t)) x
        A_val = A_func(t.item())
        x_0_expected = torch.exp(-A_val) * x
        
        # Compute error
        error = torch.norm(x_0_computed - x_0_expected, dim=1) / torch.norm(x, dim=1)
        errors.append(error.max().item())
        
        print(f"\nTest {i+1}: t = {t.item():.3f}")
        print(f"  Expected x_0 norm: {torch.norm(x_0_expected, dim=1).mean():.4f}")
        print(f"  Computed x_0 norm: {torch.norm(x_0_computed, dim=1).mean():.4f}")
        print(f"  Relative error: {error.max().item():.2e}")
    
    max_error = max(errors)
    passed = max_error < 1e-5
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"Max relative error: {max_error:.2e}")
    print(f"Threshold: < 1e-5")
    print(f"{'='*60}\n")
    
    return passed


def test_A2_divergence_sign():
    """
    A2) Divergence integral has the right sign.
    
    Test: ∇·u = d·a(t) for d=2, so ℓ*(t) = 2·A(t)
    Expected: ℓ(t) ≈ 2·A(t)
    """
    print("=" * 60)
    print("TEST A2: Divergence Integral Sign")
    print("=" * 60)
    
    # Use schedule a1: a(t) = sin(πt), A(t) = (1-cos(πt))/π
    def A_func(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return (1 - torch.cos(np.pi * t)) / np.pi
    
    # True velocity
    def u(x, t):
        a_val = torch.sin(np.pi * t)
        if a_val.dim() > 0 and x.dim() > 1:
            a_val = a_val.unsqueeze(-1)
        return a_val * x
    
    class TrueVelocityModel:
        def __call__(self, x, t):
            # Convert t to tensor if it's a scalar
            if isinstance(t, (int, float)):
                t = torch.tensor(t, dtype=torch.float64)
            return u(x, t)
    
    # Divergence of u = ∇·(a(t)x) = 2·a(t) for d=2
    # So ℓ* = ∫₀ᵗ ∇·u ds = 2·A(t)
    
    model = TrueVelocityModel()
    schedule = Schedule.A1
    
    errors = []
    for i in range(5):
        t = torch.tensor(np.random.rand(), dtype=torch.float64)
        x = torch.randn(5, 2, dtype=torch.float64)
        
        # Solve backward ODE and accumulate divergence
        _, ell_computed = backward_ode_and_divergence(model, x, t.item())
        
        # Expected divergence integral
        A_val = A_func(t.item())
        ell_expected = 2.0 * A_val  # d=2
        
        # Compute error
        ell_computed_mean = ell_computed.mean().item()
        error = abs(ell_computed_mean - ell_expected)
        errors.append(error)
        
        print(f"\nTest {i+1}: t = {t.item():.3f}")
        print(f"  Expected ell: {ell_expected:.6f}")
        print(f"  Computed ell (mean): {ell_computed_mean:.6f}")
        print(f"  Error: {error:.6e}")
    
    max_error = max(errors)
    passed = max_error < 1e-5
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"Max error: {max_error:.2e}")
    print(f"Threshold: < 1e-5")
    print(f"{'='*60}\n")
    
    return passed


def test_A3_log_density():
    """
    A3) Log-density equality when v=u.
    
    When v(x,t) = u(x,t), we should have q_t(x) = p_t(x).
    Therefore: log q_t(x) ≈ log p_t(x)
    """
    print("=" * 60)
    print("TEST A3: Log-Density Equality (v=u)")
    print("=" * 60)
    
    # Use schedule a1
    def a_func(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return torch.sin(np.pi * t)
    
    def A_func(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return (1 - torch.cos(np.pi * t)) / np.pi
    
    def u(x, t):
        a_val = a_func(t)
        if a_val.dim() > 0 and x.dim() > 1:
            a_val = a_val.unsqueeze(-1)
        return a_val * x
    
    class TrueVelocityModel:
        def __call__(self, x, t):
            # Convert t to tensor if it's a scalar
            if isinstance(t, (int, float)):
                t = torch.tensor(t, dtype=torch.float64)
            return u(x, t)
    
    model = TrueVelocityModel()
    schedule = Schedule.A1
    
    errors = []
    for i in range(5):
        t = torch.tensor(np.random.rand(), dtype=torch.float64)
        # Sample from p_t
        x = sample_p_t(t.item(), 10, schedule)
        
        # DEBUG: Inspect backward ODE and divergence
        print(f"\n{'='*60}")
        print(f"Test {i+1}: t = {t.item():.3f}")
        print(f"{'='*60}")
        
        # Get backward ODE result
        x_0, ell = backward_ode_and_divergence(model, x, t.item())
        
        # Expected divergence from A2
        A_val = A_func(t).item()
        ell_expected = 2.0 * A_val
        
        print(f"Backward ODE results:")
        print(f"  x shape: {x.shape}")
        print(f"  x_0 shape: {x_0.shape}")
        print(f"  Divergence ell: {ell.mean().item():.6f}")
        print(f"  Expected ell (2A(t)): {ell_expected:.6f}")
        print(f"  Divergence error: {abs(ell.mean().item() - ell_expected):.6e}")
        
        # Compute log q_t(x)
        log_q = log_q_t(x, t.item(), model, schedule)
        
        # Compute log p_t(x)
        log_p = log_p_t(x, t.item(), schedule)
        
        # Compute error
        error = torch.abs(log_q - log_p)
        errors.append(error.max().item())
        
        print(f"\nLog densities:")
        print(f"  log p_t (mean): {log_p.mean():.4f}")
        print(f"  log q_t (mean): {log_q.mean():.4f}")
        print(f"  log q - log p (mean): {(log_q - log_p).mean():.4f}")
        print(f"  Max absolute error: {error.max().item():.4e}")
        
        # DEBUG: Check analytical formula
        # For linear velocity u(x,t) = a(t)x, we have:
        # - Forward: x(t) = x(0) * exp(A(t))
        # - Backward: x(0) = x(t) * exp(-A(t))
        # - p_t = N(0, σ²(t) I) where σ(t) = exp(A(t))
        # - log p_t(x) = -d/2 log(2π) - d log(σ(t)) - |x|²/(2σ²(t))
        sigma_p = torch.exp(torch.tensor(A_val, dtype=torch.float64))
        log_p_analytical = -np.log(2 * np.pi) - 2 * torch.log(sigma_p) - torch.sum(x ** 2, dim=-1) / (2 * sigma_p ** 2)
        # Check if the error is systematic (always positive or always negative)
        if torch.all(log_q > log_p):
            print(f"\nSIGN: log_q consistently > log_p")
        elif torch.all(log_q < log_p):
            print(f"\nSIGN: log_q consistently < log_p")
        else:
            print(f"\nSIGN: Mixed (some >, some <)")
    
    max_error = max(errors)
    passed = max_error < 1e-4
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"Max absolute error: {max_error:.2e}")
    print(f"Threshold: < 1e-4")
    print(f"{'='*60}\n")
    
    return passed


def test_B1_normalization():
    """
    B1) Normalization check.
    
    Test: M(t) = E_x~p_t[exp(log q_t - log p_t)] should be ≈ 1
    This checks that q_t is a proper probability density.
    """
    print("=" * 60)
    print("TEST B1: Normalization Check")
    print("=" * 60)
    
    # Use learned model (from training)
    from model import VelocityMLP
    import os
    
    model_path = "data/models/vtheta_schedule_a1.pth"
    if not os.path.exists(model_path):
        print("WARNING: No trained model found. Skipping B1.")
        print("Run experiment.py first to train a model.")
        print(f"{'='*60}\n")
        return None  # Skip
    
    model = VelocityMLP(hidden_dims=[128, 128], activation='silu')
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    schedule = Schedule.A1
    
    # Test at different times
    m_values = []
    for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        t = torch.tensor(t_val, dtype=torch.float64)
        
        # Sample from p_t
        n_samples = 1000
        x_samples = sample_p_t(t_val, n_samples, schedule)
        
        # Note: log_q_t internally handles requires_grad for divergence computation
        # We just need to ensure x_samples requires grad
        x_samples_grad = x_samples.requires_grad_(True)
        
        # Compute log q_t and log p_t
        log_q = log_q_t(x_samples_grad, t_val, model, schedule)
        log_p = log_p_t(x_samples, t_val, schedule)
        
        # Compute M(t) = E[exp(log q - log p)]
        log_diff = log_q - log_p
        M_t = torch.mean(torch.exp(log_diff))
        m_values.append(M_t.item())
        
        print(f"t = {t_val:.1f}: M(t) = {M_t.item():.4f}")
    
    # Check if all M(t) are close to 1.0
    m_array = np.array(m_values)
    mean_m = np.mean(m_array)
    std_m = np.std(m_array)
    max_deviation = np.max(np.abs(m_array - 1.0))
    
    passed = max_deviation < 0.02  # ±0.02 threshold
    
    print(f"\n{'='*60}")
    print(f"Mean M(t): {mean_m:.4f}")
    print(f"Std M(t): {std_m:.4f}")
    print(f"Max deviation from 1.0: {max_deviation:.4f}")
    print(f"Threshold: ±0.02")
    print(f"{'='*60}")
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}\n")
    
    return passed


if __name__ == '__main__':
    print("\n" + "="*60)
    print("GOLDEN-PATH TESTS FOR LHS EVALUATION PIPELINE")
    print("="*60)
    
    # Run tests A1-A3
    results = {}
    results['A1'] = test_A1_preimage()
    results['A2'] = test_A2_divergence_sign()
    results['A3'] = test_A3_log_density()
    
    # Run B1 if model exists
    print("\n" + "="*60)
    print("TEST B1: Normalization Check")
    print("="*60)
    
    # Run B1
    result_B1 = test_B1_normalization()
    if result_B1 is not None:  # Only add if B1 was run
        results['B1'] = result_B1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        if passed is not None:  # Skip if not run
            status = "PASS" if passed else "FAIL"
            print(f"{test_name}: {status}")
    
    all_passed = all(p for p in results.values() if p is not None)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60)

