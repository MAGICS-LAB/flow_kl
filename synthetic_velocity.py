"""
Synthetic velocity backend for Part-2 bound verification.

Provides a velocity field v(x,t) = (a(t) + δ(t))·x with analytic divergence,
matching the API expected by the evaluation pipeline.
"""

import math
import torch


class SyntheticVelocity:
    """
    Synthetic velocity field v(x,t) = (a(t) + δ(t))·x.
    
    This implements the linear isotropic velocity field with perturbation δ(t).
    The divergence is computed analytically as ∇·v = d·(a(t) + δ(t)).
    
    Args:
        a_fn: Schedule function a(t) -> float
        delta_fn: Perturbation function δ(t) -> float
        dim: Spatial dimension (default 2)
    """
    
    def __init__(self, a_fn, delta_fn, dim=2):
        self.a_fn = a_fn          # callable: float -> float
        self.delta_fn = delta_fn  # callable: float -> float
        self.dim = dim
        self.name = "synthetic"
    
    def __call__(self, x, t):
        """Alias for forward to support callable syntax."""
        return self.forward(x, t)
    
    def forward(self, x, t):
        """
        Compute velocity v(x,t) = (a(t) + δ(t))·x.
        
        Args:
            x: Spatial coordinates of shape [..., dim]
            t: Time point(s) - scalar or tensor broadcastable to x's shape
        
        Returns:
            Velocity vectors of shape [..., dim]
        """
        coef = self._coef(x, t)   # (..., 1)
        return coef * x
    
    def divergence(self, x, t):
        """
        Compute divergence ∇·v(x,t) = dim·(a(t) + δ(t)).
        
        Note: This is independent of x for linear isotropic fields.
        
        Args:
            x: Spatial coordinates of shape [..., dim] (used only for dtype/device/shape)
            t: Time point(s) - scalar or tensor broadcastable
        
        Returns:
            Divergence scalars of shape [..., 1]
        """
        coef = self._coef(x, t)   # (..., 1)
        return self.dim * coef    # (..., 1)
    
    def _coef(self, x, t):
        """
        Compute coefficient (a(t) + δ(t)) preserving dtype/device/batch.
        
        Args:
            x: Tensor used to infer dtype and device
            t: Time point(s) - scalar or tensor
        
        Returns:
            Coefficient tensor of shape [..., 1] that broadcasts to x
        """
        # Extract scalar t value
        if torch.is_tensor(t):
            t_val = t.item() if t.dim() == 0 else float(t)
        else:
            t_val = float(t)
        
        # Compute a(t) and δ(t) as scalars
        a_val = float(self.a_fn(t_val))
        d_val = float(self.delta_fn(t_val))
        coef_val = a_val + d_val
        
        # Create tensor with shape that broadcasts to x
        # x has shape [..., dim], we want coef with shape [..., 1]
        # For simplicity, create shape [1, 1] which broadcasts to any [..., dim]
        # This works because the coefficient is constant (independent of x)
        coef = torch.full((1, 1), coef_val, dtype=x.dtype, device=x.device)
        
        return coef


def constant_delta(beta):
    """
    Factory for constant perturbation δ(t) = β.
    
    Args:
        beta: Constant value
    
    Returns:
        Callable δ(t) -> float
    """
    return lambda t: float(beta)


def sine_delta(beta):
    """
    Factory for oscillatory perturbation δ(t) = β·sin(2πt).
    
    Args:
        beta: Amplitude coefficient
    
    Returns:
        Callable δ(t) -> float
    """
    return lambda t: float(beta) * math.sin(2 * math.pi * float(t))


if __name__ == '__main__':
    """Basic sanity tests for SyntheticVelocity."""
    print("=" * 60)
    print("Testing SyntheticVelocity")
    print("=" * 60)
    
    import numpy as np
    from true_path import get_schedule_functions, Schedule
    
    # Get a schedule function
    a_fn, A_fn = get_schedule_functions(Schedule.A1)
    
    # Test constant delta
    delta_fn = constant_delta(0.1)
    velocity = SyntheticVelocity(a_fn, delta_fn, dim=2)
    
    # Test forward
    print("\n1. Testing forward pass")
    t_val = 0.5
    x = torch.randn(3, 2, dtype=torch.float64)
    
    # Compute manually
    a_val = a_fn(t_val)
    delta_val = delta_fn(t_val)
    expected = (a_val + delta_val) * x
    
    # Compute via velocity
    result = velocity.forward(x, t_val)
    
    diff = torch.abs(result - expected).max().item()
    print(f"   Max difference: {diff:.2e}")
    assert diff < 1e-12, f"Forward pass failed: max diff = {diff:.2e}"
    print("   ✓ Forward pass correct")
    
    # Test divergence
    print("\n2. Testing divergence")
    x1 = torch.randn(3, 2, dtype=torch.float64)
    x2 = torch.randn(5, 2, dtype=torch.float64)
    
    div1 = velocity.divergence(x1, t_val)
    div2 = velocity.divergence(x2, t_val)
    
    # All should be the same (independent of x)
    all_same = (div1.unsqueeze(0) == div2.unsqueeze(0)).all().item()
    expected_div = 2.0 * (a_val + delta_val)  # dim=2
    
    diff1 = torch.abs(div1 - expected_div).max().item()
    print(f"   Max difference from expected: {diff1:.2e}")
    print(f"   Independent of x: {all_same}")
    
    assert diff1 < 1e-12, f"Divergence failed: max diff = {diff1:.2e}"
    assert all_same, "Divergence depends on x (should not)"
    print("   ✓ Divergence correct and independent of x")
    
    # Test sine delta
    print("\n3. Testing sine delta")
    delta_fn_sine = sine_delta(0.2)
    velocity_sine = SyntheticVelocity(a_fn, delta_fn_sine, dim=2)
    
    t_test = 0.25
    expected_sine = 0.2 * math.sin(2 * math.pi * 0.25)
    result_sine = delta_fn_sine(t_test)
    
    assert abs(result_sine - expected_sine) < 1e-12, "Sine delta incorrect"
    print(f"   δ(0.25) = {result_sine:.6f}")
    print("   ✓ Sine delta correct")
    
    print("\n" + "=" * 60)
    print("All sanity tests passed!")
    print("=" * 60)

