"""
True path p_t implementations: schedules, sampling, log density, and score.

All formulas are for d=2 (2D Gaussian).
"""

import torch
import numpy as np
from enum import Enum


class Schedule(Enum):
    """Schedule functions for velocity field u(x,t) = a(t) x."""
    A1 = "a1"  # a(t) = sin(πt)
    A2 = "a2"  # a(t) = 0.3 sin(2πt) + 0.2
    A3 = "a3"  # a(t) = t - 1/2


def a1(t):
    """Schedule a₁(t) = sin(πt)."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return torch.sin(np.pi * t)


def a2(t):
    """Schedule a₂(t) = 0.3 sin(2πt) + 0.2."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return 0.3 * torch.sin(2 * np.pi * t) + 0.2


def a3(t):
    """Schedule a₃(t) = t - 1/2."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return t - 0.5


def A1(t):
    """Closed-form integral ∫₀ᵗ a₁(s) ds = (1-cos(πt))/π."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return (1 - torch.cos(np.pi * t)) / np.pi


def A2(t):
    """Closed-form integral ∫₀ᵗ a₂(s) ds = (0.3/(2π))(1-cos(2πt)) + 0.2t."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return (0.3 / (2 * np.pi)) * (1 - torch.cos(2 * np.pi * t)) + 0.2 * t


def A3(t):
    """Closed-form integral ∫₀ᵗ a₃(s) ds = (1/2) t² - (1/2) t."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    return 0.5 * t ** 2 - 0.5 * t


def get_schedule_functions(schedule):
    """Get the appropriate schedule functions for the given schedule."""
    if schedule == Schedule.A1:
        return a1, A1
    elif schedule == Schedule.A2:
        return a2, A2
    elif schedule == Schedule.A3:
        return a3, A3
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def schedule_to_enum(schedule_str):
    """Convert string to Schedule enum."""
    mapping = {'a1': Schedule.A1, 'a2': Schedule.A2, 'a3': Schedule.A3}
    return mapping.get(schedule_str, None)


def sigma_p(t, schedule, a_func=None, A_func=None):
    """
    Compute σ_p(t) = exp(A(t)) where A(t) = ∫₀ᵗ a(s) ds.
    
    Args:
        t: Time point(s) - scalar or tensor
        schedule: Schedule enum
        a_func, A_func: Optional pre-computed functions
        
    Returns:
        σ_p(t) as tensor
    """
    if A_func is None:
        _, A_func = get_schedule_functions(schedule)
    
    return torch.exp(A_func(t))


def log_p_t(x, t, schedule, sigma_p_val=None):
    """
    Log density of p_t = N(0, σ_p(t)² I₂) at point x.
    
    Formula: log p_t(x) = -(d/2) log(2π) - d log σ_p(t) - |x|²/(2σ_p(t)²)
    
    Args:
        x: Point(s) of shape [..., 2]
        t: Time point(s)
        schedule: Schedule enum
        sigma_p_val: Pre-computed σ_p(t) (optional)
    
    Returns:
        log p_t(x) as scalar or tensor
    """
    d = 2  # dimension
    
    if sigma_p_val is None:
        sigma_p_val = sigma_p(t, schedule)
    
    # Ensure x is a tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    
    # Compute log p_t(x)
    constant = -(d / 2) * np.log(2 * np.pi)
    log_sigma = -d * torch.log(sigma_p_val)
    quadratic = -torch.sum(x ** 2, dim=-1) / (2 * sigma_p_val ** 2)
    
    return constant + log_sigma + quadratic


def score_p_t(x, t, schedule, sigma_p_val=None):
    """
    Score ∇log p_t(x) = -x / σ_p(t)².
    
    Args:
        x: Point(s) of shape [..., 2]
        t: Time point(s)
        schedule: Schedule enum
        sigma_p_val: Pre-computed σ_p(t) (optional)
    
    Returns:
        Score vector(s) of shape [..., 2]
    """
    if sigma_p_val is None:
        sigma_p_val = sigma_p(t, schedule)
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    
    return -x / (sigma_p_val ** 2)


def velocity_u(x, t, schedule):
    """
    True velocity field u(x,t) = a(t) x.
    
    Args:
        x: Point(s) of shape [..., 2]
        t: Time point(s)
        schedule: Schedule enum
    
    Returns:
        Velocity vector(s) of shape [..., 2]
    """
    a_func, _ = get_schedule_functions(schedule)
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    
    a_val = a_func(t)
    
    # Ensure proper broadcasting: a_val can be scalar or [batch_size]
    # Output should be [batch_size, 2]
    if x.dim() > 1:
        # a_val has shape [batch_size] or scalar, x has shape [batch_size, 2]
        if a_val.dim() == 1:
            # Broadcast [batch_size] with [batch_size, 2]
            a_val = a_val.unsqueeze(-1)  # [batch_size, 1]
    
    return a_val * x


def sample_p_t(t, batch_size, schedule, device='cpu', dtype=torch.float64):
    """
    Sample x ~ p_t = N(0, σ_p(t)² I₂).
    
    Implementation: sample z ~ N(0, I₂), set x = σ_p(t) z.
    
    Args:
        t: Time point(s) - scalar or 1D tensor
        batch_size: Number of samples
        schedule: Schedule enum
        device: torch device
        dtype: torch dtype (default float64)
    
    Returns:
        Samples of shape [batch_size, 2]
    """
    sigma_p_val = sigma_p(t, schedule)
    
    # Sample z ~ N(0, I₂)
    z = torch.randn(batch_size, 2, dtype=dtype, device=device)
    
    # Scale by σ_p(t)
    x = sigma_p_val * z
    
    return x


if __name__ == '__main__':
    # Test schedule functions
    t = torch.linspace(0, 1, 101, dtype=torch.float64)
    
    print("Testing schedule functions...")
    
    # Test a1
    a1_vals = a1(t)
    A1_vals = A1(t)
    print(f"a1(0.5) = {a1(0.5):.4f}, A1(0.5) = {A1(0.5):.4f}")
    
    # Test a2
    a2_vals = a2(t)
    A2_vals = A2(t)
    print(f"a2(0.5) = {a2(0.5):.4f}, A2(0.5) = {A2(0.5):.4f}")
    
    # Test a3
    a3_vals = a3(t)
    A3_vals = A3(t)
    print(f"a3(0.5) = {a3(0.5):.4f}, A3(0.5) = {A3(0.5):.4f}")
    
    # Test sampling
    print("\nTesting sampling from p_t...")
    for sched in [Schedule.A1, Schedule.A2, Schedule.A3]:
        samples = sample_p_t(0.5, 1000, sched)
        sigma = sigma_p(0.5, sched)
        # Check variance
        var_estimate = torch.mean(samples ** 2, dim=0)
        print(f"Schedule {sched.value}: σ² = {sigma**2:.4f}, estimated = {var_estimate.mean():.4f}")
    
    print("\ntrue_path.py loaded successfully")

