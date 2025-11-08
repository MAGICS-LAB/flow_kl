"""
Evaluation routines for learned density q_t (Part-2 version).

This module supports both neural network velocities (Part 1) and synthetic velocities (Part 2).
Core functions:
- divergence_of: Generic divergence computation (analytical or autograd)
- backward_ode_and_divergence: Solve backward ODE and accumulate divergence
- log_q_t: Compute log density of q_t
- score_q_t: Compute score ∇log q_t
- compute_kl_lhs: Estimate KL(p_t|q_t)
- compute_rhs_integrand: Compute (u-v)ᵀ(s_p-s_q)
- compute_epsilon_pt2: Compute ε (RMS flow-matching loss)
- compute_score_gap_integral_pt2: Compute S (score-gap integral)
- compute_kl_at_t1_pt2: Compute KL at t=1
"""

import torch
import numpy as np
import math
from torchdiffeq import odeint

from core.true_path import (
    schedule_to_enum, velocity_u, score_p_t, log_p_t,
    sigma_p, Schedule, sample_p_t
)


def divergence_of(velocity, x, t):
    """
    Compute divergence ∇·v(x,t) using analytic divergence if available, otherwise autograd.

    Args:
        velocity: Velocity object with optional `divergence` method
        x: Spatial coordinates of shape [..., dim]
        t: Time point(s)

    Returns:
        Divergence tensor of shape [..., 1] matching x's batch shape
    """
    # Prefer analytic divergence if provided
    if hasattr(velocity, "divergence"):
        div = velocity.divergence(x, t)  # shape (..., 1)
        # Expand to match batch size if needed
        # div might be [1, 1] but x is [N, dim], so expand div to [N, 1]
        if x.dim() > 1 and div.shape[0] == 1:
            # x is batched, expand div to match batch size
            batch_size = x.shape[0]
            div = div.expand(batch_size, -1)
        return div

    # Fallback: autograd divergence wrt x of velocity.forward
    x_req = x.clone().requires_grad_(True)
    v = velocity.forward(x_req, t)  # (..., dim)

    # Sum_i ∂v_i/∂x_i
    grads = []
    for i in range(v.shape[-1]):
        gi = torch.autograd.grad(
            v[..., i].sum(), x_req, create_graph=False, retain_graph=True
        )[0][..., i:i+1]
        grads.append(gi)

    # Sum the diagonal terms
    div = sum(grads)
    return div  # (..., 1)


def backward_ode_and_divergence(v_theta, x, t, rtol=1e-6, atol=1e-8):
    """
    Solve backward ODE dx/ds = -v_θ(x(s),s) from s=t down to s=0.
    Simultaneously accumulate divergence ℓ = ∫₀ᵗ ∇·v_θ(x(s),s) ds.

    Args:
        v_theta: VelocityMLP model
        x: Terminal points of shape [B, 2]
        t: Terminal time (scalar or tensor of shape [B])
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver

    Returns:
        x_0: Source points of shape [B, 2] (x(0))
        ell: Divergence integral of shape [B, 1] (ℓ(t))
    """
    batch_size = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Ensure t is a tensor
    if not isinstance(t, torch.Tensor):
        t_val = t
    else:
        t_val = t.item() if t.dim() == 0 else t[0].item()

    # Edge case: t=0
    if t_val < 1e-10:
        return x, torch.zeros(batch_size, 1, dtype=dtype, device=device)

    # Initial state: [x, ℓ] where ℓ=0 initially
    z0 = torch.cat([x, torch.zeros(batch_size, 1, dtype=dtype, device=device)], dim=1)

    # Define ODE dynamics
    def ode_func(s, z):
        # Unpack state: [x₁, x₂, ℓ]
        x_s = z[:, :2]  # [B, 2]
        ell = z[:, 2:3]  # [B, 1]

        # Compute velocity (backward direction)
        v = v_theta.forward(x_s, s)

        # Compute divergence using the generic helper
        divergence = divergence_of(v_theta, x_s, s).squeeze(-1)

        # Backward flow: dx/ds = +v_θ
        dx_ds = v

        # Divergence accumulation: dℓ/ds = +∇·v_θ
        dell_ds = divergence.unsqueeze(-1)

        return torch.cat([dx_ds, dell_ds], dim=1)

    # Solve backward from t to 0
    t_grid = torch.tensor([t_val, 0.0], dtype=dtype, device=device)
    z_sol = odeint(ode_func, z0, t_grid, method='dopri5', rtol=rtol, atol=atol)

    # Extract solution at s=0
    z_final = z_sol[-1]
    x_0 = z_final[:, :2]
    ell_raw = z_final[:, 2:3]

    # Negate the divergence since we integrated backward in time
    ell = -ell_raw

    return x_0, ell


def log_q_t(x, t, v_theta, schedule, rtol=1e-6, atol=1e-8):
    """
    Compute log q_t(x) = log p_0(x_0) + ℓ(t).

    Args:
        x: Point(s) of shape [..., 2]
        t: Time point (scalar)
        v_theta: VelocityMLP model
        schedule: Schedule enum
        rtol, atol: ODE solver tolerances

    Returns:
        log q_t(x) as tensor
    """
    # Edge case: t=0
    if t < 1e-10:
        # log q_0(x) = log p_0(x)
        # p_0 = N(0, I₂) since σ_p(0) = exp(A(0)) = exp(0) = 1
        if x.dim() == 1:
            x_flat = x
        else:
            x_flat = x.reshape(-1, 2)

        # For d=2: log p_0(x) = -d/2 * log(2π) - |x|²/2 = -log(2π) - |x|²/2
        d = 2
        log_p0 = -(d / 2) * np.log(2 * np.pi) - 0.5 * torch.sum(x_flat ** 2, dim=-1)
        return log_p0.view(x.shape[:-1]) if x.dim() > 1 else log_p0

    # Reshape to batch
    original_shape = x.shape[:-1]
    x_batch = x.reshape(-1, 2)

    # Backward ODE
    x_0, ell = backward_ode_and_divergence(v_theta, x_batch, t, rtol, atol)

    # log p_0(x_0) for d=2
    d = 2
    # Note: p_0 = N(0, I_2) so log p_0(x) = -d/2 * log(2π) - |x|²/2
    log_p0_x0 = -(d / 2) * np.log(2 * np.pi) - 0.5 * torch.sum(x_0 ** 2, dim=-1)

    # log q_t(x) = log p_0(x_0) - ℓ(t)
    log_q = log_p0_x0 - ell.squeeze(-1)

    # Reshape back
    if len(original_shape) > 0:
        log_q = log_q.reshape(original_shape)

    return log_q


def score_q_t(x, t, v_theta, schedule, rtol=1e-6, atol=1e-8):
    """
    Compute score ∇_x log q_t(x) via single autograd call.

    Critical: treat log q_t(x) as a scalar and differentiate w.r.t. x.

    Args:
        x: Point(s) of shape [..., 2]
        t: Time point (scalar)
        v_theta: VelocityMLP model
        schedule: Schedule enum
        rtol, atol: ODE solver tolerances

    Returns:
        Score ∇_x log q_t(x) of shape [..., 2]
    """
    original_shape = x.shape[:-1]

    # Reshape to batch
    x_batch = x.reshape(-1, 2)

    # Enable gradients
    x_grad = x_batch.clone().detach().requires_grad_(True)

    # Compute log q_t(x) as scalar
    log_q = log_q_t(x_grad, t, v_theta, schedule, rtol, atol)

    # Compute gradient via autograd
    score = torch.autograd.grad(log_q.sum(), x_grad, create_graph=True)[0]

    # Reshape back
    if len(original_shape) > 0:
        score = score.reshape(original_shape + (2,))

    return score


def compute_kl_lhs(x_batch, t, schedule, v_theta, rtol=1e-6, atol=1e-8):
    """
    Estimate KL(p_t|q_t) = E_x~p_t [log p_t(x) - log q_t(x)].

    Args:
        x_batch: Samples from p_t of shape [N, 2]
        t: Time point
        schedule: Schedule enum
        v_theta: VelocityMLP model
        rtol, atol: ODE solver tolerances

    Returns:
        KL estimate (scalar)
    """
    # log p_t(x)
    log_p = log_p_t(x_batch, t, schedule)

    # log q_t(x)
    log_q = log_q_t(x_batch, t, v_theta, schedule, rtol, atol)

    # KL = mean(log p_t - log q_t)
    kl = torch.mean(log_p - log_q)

    return kl.item()


def compute_rhs_integrand(x_batch, t, schedule, v_theta, rtol=1e-6, atol=1e-8):
    """
    Compute RHS integrand ĝ(t) = (1/N) Σᵢ (u⁽ⁱ⁾ - v⁽ⁱ⁾)ᵀ (s_p⁽ⁱ⁾ - s_q⁽ⁱ⁾).

    Args:
        x_batch: Samples from p_t of shape [N, 2]
        t: Time point
        schedule: Schedule enum
        v_theta: VelocityMLP model
        rtol, atol: ODE solver tolerances

    Returns:
        RHS integrand value (scalar)
    """
    # True velocity u
    u = velocity_u(x_batch, t, schedule)

    # Learned velocity v_θ
    v = v_theta(x_batch, t)

    # True score s_p = ∇log p_t
    s_p = score_p_t(x_batch, t, schedule)

    # Learned score s_q = ∇log q_t
    s_q = score_q_t(x_batch, t, v_theta, schedule, rtol, atol)

    # Inner product (u-v)ᵀ(s_p-s_q)
    inner_prod = torch.sum((u - v) * (s_p - s_q), dim=-1)

    # Average
    g_hat = torch.mean(inner_prod)

    return g_hat.item()


def integrate_rhs(rhs_integrand, t_grid):
    """
    Integrate RHS integrand over time using trapezoidal rule.

    Args:
        rhs_integrand: Values of integrand at each time point
        t_grid: Time grid

    Returns:
        Cumulative integral R(t) = ∫₀ᵗ g(s) ds
    """
    rhs_cumulative = np.zeros_like(rhs_integrand)

    for m in range(1, len(t_grid)):
        dt = t_grid[m] - t_grid[m-1]
        rhs_cumulative[m] = rhs_cumulative[m-1] + (dt / 2) * (rhs_integrand[m] + rhs_integrand[m-1])

    return rhs_cumulative


# ======================== Part-2 specific functions ========================

def compute_epsilon_pt2(a_fn, delta_fn, schedule, K_eps=101, N_eps=4096, dim=2, device='cpu', dtype=torch.float64):
    """
    Compute ε (RMS flow-matching loss) = sqrt(E|v-u|^2).

    Args:
        a_fn: Schedule function a(t) -> float
        delta_fn: Perturbation function δ(t) -> float
        schedule: Schedule enum for p_t
        K_eps: Number of time points
        N_eps: Samples per time point
        dim: Spatial dimension (default 2)
        device, dtype: Tensor device and dtype

    Returns:
        epsilon (scalar)
    """
    # Time grid
    t_grid = torch.linspace(0., 1., K_eps, dtype=dtype, device=device)

    acc = 0.0
    count = 0

    for t_val in t_grid:
        # Sample x ~ p_t
        sigma_p_val = sigma_p(t_val, schedule)
        z = torch.randn(N_eps, dim, dtype=dtype, device=device)
        x = sigma_p_val * z

        # Compute error: |δ(t)·x|^2 = δ(t)^2 * |x|^2
        delta_val = delta_fn(t_val.item())
        err2 = (delta_val ** 2) * torch.sum(x ** 2, dim=-1, keepdim=True)  # (N_eps, 1)

        acc += err2.sum().item()
        count += N_eps

    eps2 = acc / count
    return math.sqrt(eps2)


def sample_p_t_with_crn(t, batch_size, schedule, Z, dtype=torch.float64):
    """
    Sample from p_t using common random numbers.

    Args:
        t: Time point(s) - scalar
        batch_size: Number of samples
        schedule: Schedule enum
        Z: Fixed random seed of shape [batch_size, 2]
        dtype: torch dtype

    Returns:
        Samples of shape [batch_size, 2]
    """
    sigma_p_val = sigma_p(t, schedule)
    x = sigma_p_val * Z
    return x


def compute_score_gap_integral_pt2(velocity, schedule, K_S=101, N_S=2048,
                                   rtol=1e-6, atol=1e-8, device='cpu', dtype=torch.float64):
    """
    Compute S (score-gap integral) = ∫₀¹ E|s_p - s_q|² dt using common random numbers.

    Args:
        velocity: Velocity object (neural or synthetic)
        schedule: Schedule enum
        K_S: Number of time points
        N_S: Samples per time point
        rtol, atol: ODE solver tolerances
        device, dtype: Tensor device and dtype

    Returns:
        S_hat (scalar), (t_grid, f_vals) tuple
    """
    # Time grid
    t_grid = torch.linspace(0., 1., K_S, dtype=dtype, device=device)

    # Common random numbers: fix Z once, reuse for all t
    Z = torch.randn(N_S, 2, dtype=dtype, device=device)

    f_vals = []

    for t_val in t_grid:
        # Sample x using common random numbers
        x = sample_p_t_with_crn(t_val.item(), N_S, schedule, Z, dtype)

        # Compute scores
        sp = score_p_t(x, t_val.item(), schedule)  # (-x/σ_p^2)
        sq = score_q_t(x, t_val.item(), velocity, schedule, rtol, atol)  # grad of scalar log_q_t

        # Compute squared norm difference
        diff2 = torch.sum((sp - sq) ** 2, dim=-1)  # (N_S,)
        f_vals.append(diff2.mean().item())

    # Trapezoidal integration
    S_hat = 0.0
    for k in range(1, len(t_grid)):
        dt = (t_grid[k] - t_grid[k-1]).item()
        S_hat += 0.5 * dt * (f_vals[k] + f_vals[k-1])

    return S_hat, (t_grid.numpy(), f_vals)


def compute_kl_at_t1_pt2(velocity, schedule, N_kl=20000, rtol=1e-6, atol=1e-8,
                         device='cpu', dtype=torch.float64):
    """
    Compute KL(p_1|q_1) at t=1.

    Args:
        velocity: Velocity object (neural or synthetic)
        schedule: Schedule enum
        N_kl: Number of samples
        rtol, atol: ODE solver tolerances
        device, dtype: Tensor device and dtype

    Returns:
        KL at t=1 (scalar)
    """
    # Sample x ~ p_1
    x = sample_p_t(1.0, N_kl, schedule, device=device, dtype=dtype)

    # Compute log densities
    lp = log_p_t(x, 1.0, schedule)
    lq = log_q_t(x, 1.0, velocity, schedule, rtol, atol)

    return (lp - lq).mean().item()

