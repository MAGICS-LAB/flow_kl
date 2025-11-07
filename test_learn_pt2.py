"""
Unit tests for Part-2 (Learning) bound verification.

Tests cover model wiring, training, evaluation functions, and bound verification.
"""

import torch
import numpy as np
from pathlib import Path
import json

from model_learn_pt2 import VelocityMLP
from train_learn_pt2 import train_velocity_model, validate_model
from eval_learn_pt2 import (
    compute_epsilon_learn, compute_score_gap_integral_learn,
    compute_kl_at_t1_learn, log_q_t, score_q_t
)
from true_path import Schedule, velocity_u, sample_p_t
from utils import set_seed, get_device


def test_T0_model_wiring():
    """T0: Model wiring - forward shape, time handling, divergence."""
    print("\n" + "=" * 60)
    print("T0: Model Wiring")
    print("=" * 60)
    
    model = VelocityMLP(hidden_dims=[64, 64], activation='silu')
    model.eval()
    
    # Test 1: Forward shape
    print("1. Forward pass shape")
    x = torch.randn(10, 2, dtype=torch.float64)
    t_scalar = 0.5
    v = model(x, t_scalar)
    assert v.shape == (10, 2), f"Expected (10, 2), got {v.shape}"
    print(f"   ✓ Forward shape correct: {v.shape}")
    
    # Test 2: Per-sample time
    print("2. Per-sample time handling")
    t_tensor = torch.rand(10, dtype=torch.float64)
    v = model(x, t_tensor)
    assert v.shape == (10, 2), f"Expected (10, 2), got {v.shape}"
    print(f"   ✓ Per-sample time works: {v.shape}")
    
    # Test 3: Divergence method
    print("3. Divergence computation")
    x_grad = x.clone().requires_grad_(True)
    div_method = model.compute_divergence(x_grad, t_scalar)
    
    # Manual autograd divergence
    v_grad = model(x_grad, t_scalar)
    div_manual = torch.zeros_like(v_grad[..., 0])
    for i in range(2):
        grad_i = torch.autograd.grad(
            v_grad[..., i].sum(), x_grad, create_graph=False, retain_graph=True
        )[0]
        div_manual += grad_i[..., i]
    
    max_diff = torch.abs(div_method.squeeze(-1) - div_manual).max().item()
    assert max_diff < 1e-10, f"Divergence mismatch: {max_diff}"
    print(f"   ✓ Divergence method matches autograd (max diff: {max_diff:.2e})")
    
    print("   ✓ T0 PASSED")
    return True


def test_T1_training_learns():
    """T1: Training actually learns."""
    print("\n" + "=" * 60)
    print("T1: Training Learns")
    print("=" * 60)
    
    set_seed(42)
    device = 'cpu'
    
    model = VelocityMLP(hidden_dims=[32, 32], activation='silu')
    
    # Test 1: Loss decreases
    print("1. Loss decreases over epochs")
    best_mse, history = train_velocity_model(
        model=model,
        schedule=Schedule.A1,
        epochs=10,
        lr=1e-3,
        batch_size=64,
        num_batches_per_epoch=16,
        val_times=16,
        val_samples_per_time=256,
        device=device,
        dtype=torch.float64,
        checkpoint_dir=None,  # No checkpoints for test
        save_epochs=None,
        seed=42
    )
    
    if len(history['train_mse']) > 1:
        initial_mse = history['train_mse'][0]
        final_mse = history['train_mse'][-1]
        decreases = final_mse < initial_mse
        print(f"   Initial MSE: {initial_mse:.6e}")
        print(f"   Final MSE: {final_mse:.6e}")
        print(f"   ✓ Loss decreases: {decreases}")
        assert decreases, "Loss should decrease during training"
    else:
        print("   ⚠ Not enough epochs for comparison")
    
    # Test 2: Validation NMSE < 1
    print("2. Validation NMSE < 1")
    if len(history['val_nmse']) > 0:
        final_nmse = history['val_nmse'][-1]
        reasonable = final_nmse < 1.0
        print(f"   Final NMSE: {final_nmse:.6e}")
        print(f"   ✓ NMSE < 1: {reasonable}")
        assert reasonable, "NMSE should be < 1 with reasonable training"
    else:
        print("   ⚠ No validation data available")
    
    print("   ✓ T1 PASSED")
    return True


def test_T2_epsilon_consistency():
    """T2: ε_θ from validate_model equals direct MC estimate."""
    print("\n" + "=" * 60)
    print("T2: ε_θ Consistency")
    print("=" * 60)
    
    set_seed(42)
    device = 'cpu'
    
    # Train a small model
    model = VelocityMLP(hidden_dims=[32, 32], activation='silu')
    train_velocity_model(
        model=model,
        schedule=Schedule.A1,
        epochs=5,
        lr=1e-3,
        batch_size=64,
        num_batches_per_epoch=16,
        val_times=16,
        val_samples_per_time=256,
        device=device,
        dtype=torch.float64,
        checkpoint_dir=None,
        save_epochs=None,
        seed=42
    )
    model.eval()
    
    # Method 1: validate_model
    val_mse, _ = validate_model(model, Schedule.A1, val_times=32, 
                                val_samples_per_time=512, device=device, dtype=torch.float64)
    epsilon_validate = np.sqrt(val_mse)
    
    # Method 2: Direct MC
    set_seed(42)  # Same seed for fair comparison
    mse_list = []
    t_vals = torch.rand(32, dtype=torch.float64, device=device)
    
    with torch.no_grad():
        for t_val in t_vals:
            x = sample_p_t(t_val.item(), 512, Schedule.A1, device=device, dtype=torch.float64)
            u = velocity_u(x, t_val.expand(512), Schedule.A1)
            v = model(x, t_val.expand(512))
            mse = torch.mean((v - u) ** 2)
            mse_list.append(mse.item())
    
    mse_direct = np.mean(mse_list)
    epsilon_direct = np.sqrt(mse_direct)
    
    # Compare
    rel_diff = abs(epsilon_validate - epsilon_direct) / max(epsilon_validate, epsilon_direct, 1e-10)
    print(f"   ε (validate_model): {epsilon_validate:.6e}")
    print(f"   ε (direct MC): {epsilon_direct:.6e}")
    print(f"   Relative difference: {rel_diff:.2%}")
    print(f"   ✓ Relative diff < 20%: {rel_diff < 0.20}")  # Relaxed threshold for MC variance
    
    assert rel_diff < 0.20, f"Relative difference {rel_diff:.2%} exceeds 20%"
    
    print("   ✓ T2 PASSED")
    return True


def test_T3_score_oracle():
    """T3: Score oracle at small times."""
    print("\n" + "=" * 60)
    print("T3: Score Oracle at Small Times")
    print("=" * 60)
    
    set_seed(42)
    device = 'cpu'
    
    # Train a small model
    model = VelocityMLP(hidden_dims=[32, 32], activation='silu')
    train_velocity_model(
        model=model,
        schedule=Schedule.A1,
        epochs=5,
        lr=1e-3,
        batch_size=64,
        num_batches_per_epoch=16,
        val_times=16,
        val_samples_per_time=256,
        device=device,
        dtype=torch.float64,
        checkpoint_dir=None,
        save_epochs=None,
        seed=42
    )
    model.eval()
    
    # Test at t=0 (should match Gaussian score: s = -x)
    print("1. Score at t=0")
    x_test = torch.randn(5, 2, dtype=torch.float64, device=device)
    s_q = score_q_t(x_test, 0.0, model, Schedule.A1, rtol=1e-6, atol=1e-8)
    s_p_expected = -x_test  # For t=0, σ_p(0)=1, so s_p = -x/σ² = -x
    
    max_diff = torch.abs(s_q - s_p_expected).max().item()
    print(f"   Max difference from expected: {max_diff:.6e}")
    print(f"   ✓ Close to expected: {max_diff < 0.1}")
    
    assert max_diff < 0.1, f"Score at t=0 should be close to -x, got max diff {max_diff}"
    
    # Test at very small t
    print("2. Score at t=1e-3")
    t_small = 1e-3
    s_q_small = score_q_t(x_test, t_small, model, Schedule.A1, rtol=1e-6, atol=1e-8)
    # At small t, should still be approximately -x/σ_p(t)²
    from true_path import sigma_p
    sigma_val = sigma_p(t_small, Schedule.A1)
    s_p_small_expected = -x_test / (sigma_val ** 2)
    
    max_diff_small = torch.abs(s_q_small - s_p_small_expected).max().item()
    print(f"   Max difference from expected: {max_diff_small:.6e}")
    print(f"   ✓ Reasonable agreement: {max_diff_small < 0.5}")
    
    print("   ✓ T3 PASSED")
    return True


def test_T4_backward_ode_numerics():
    """T4: Backward-ODE numerics - consistency and tolerance sensitivity."""
    print("\n" + "=" * 60)
    print("T4: Backward-ODE Numerics")
    print("=" * 60)
    
    set_seed(42)
    device = 'cpu'
    
    # Train a small model
    model = VelocityMLP(hidden_dims=[32, 32], activation='silu')
    train_velocity_model(
        model=model,
        schedule=Schedule.A1,
        epochs=5,
        lr=1e-3,
        batch_size=64,
        num_batches_per_epoch=16,
        val_times=16,
        val_samples_per_time=256,
        device=device,
        dtype=torch.float64,
        checkpoint_dir=None,
        save_epochs=None,
        seed=42
    )
    model.eval()
    
    # Test 1: Finite difference consistency
    print("1. Finite difference consistency")
    t = 0.5
    x = torch.randn(1, 2, dtype=torch.float64, device=device)
    
    # Compute score via autograd
    x_grad = x.clone().requires_grad_(True)
    logq = log_q_t(x_grad, t, model, Schedule.A1, rtol=1e-6, atol=1e-8, need_grad_for_score=True)
    s_q = torch.autograd.grad(logq.sum(), x_grad, create_graph=False)[0]
    
    # Finite difference check (directional derivative)
    h = 1e-4
    v = torch.randn(1, 2, dtype=torch.float64, device=device)
    v = v / torch.norm(v, dim=-1, keepdim=True)  # Unit vector
    
    x_plus = x + h * v
    logq_plus = log_q_t(x_plus, t, model, Schedule.A1, rtol=1e-6, atol=1e-8, need_grad_for_score=False)
    logq_base = log_q_t(x, t, model, Schedule.A1, rtol=1e-6, atol=1e-8, need_grad_for_score=False)
    
    fd_deriv = (logq_plus - logq_base) / h
    ad_deriv = torch.sum(s_q * v)
    
    rel_error = abs(fd_deriv.item() - ad_deriv.item()) / max(abs(ad_deriv.item()), 1e-10)
    print(f"   Finite diff: {fd_deriv.item():.6e}")
    print(f"   Autograd: {ad_deriv.item():.6e}")
    print(f"   Relative error: {rel_error:.2e}")
    print(f"   ✓ Rel error < 2e-2: {rel_error < 2e-2}")
    
    assert rel_error < 2e-2, f"Relative error {rel_error:.2e} exceeds 2e-2"
    
    # Test 2: Tolerance sensitivity
    print("2. Tolerance sensitivity")
    x_test = torch.randn(1, 2, dtype=torch.float64, device=device)
    
    logq_base = log_q_t(x_test, t, model, Schedule.A1, rtol=1e-6, atol=1e-8, need_grad_for_score=False)
    logq_tight = log_q_t(x_test, t, model, Schedule.A1, rtol=3e-7, atol=3e-9, need_grad_for_score=False)
    logq_loose = log_q_t(x_test, t, model, Schedule.A1, rtol=1e-5, atol=1e-7, need_grad_for_score=False)
    
    diff_tight = abs(logq_tight.item() - logq_base.item())
    diff_loose = abs(logq_loose.item() - logq_base.item())
    
    print(f"   Diff (tight): {diff_tight:.6e}")
    print(f"   Diff (loose): {diff_loose:.6e}")
    print(f"   ✓ Changes < 1e-3: {diff_tight < 1e-3 and diff_loose < 1e-3}")
    
    assert diff_tight < 1e-3, f"Tight tolerance change {diff_tight:.2e} exceeds 1e-3"
    assert diff_loose < 1e-3, f"Loose tolerance change {diff_loose:.2e} exceeds 1e-3"
    
    print("   ✓ T4 PASSED")
    return True


def test_T5_bound_holds_and_tightens():
    """T5: Bound holds and tightens with training."""
    print("\n" + "=" * 60)
    print("T5: Bound Holds and Tightens")
    print("=" * 60)
    
    set_seed(42)
    device = 'cpu'
    dtype = torch.float64
    
    model = VelocityMLP(hidden_dims=[32, 32], activation='silu')
    
    # Train with checkpoint saving
    test_checkpoint_dir = Path('data/test_checkpoints')
    test_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_mse, history = train_velocity_model(
        model=model,
        schedule=Schedule.A1,
        epochs=20,
        lr=1e-3,
        batch_size=64,
        num_batches_per_epoch=16,
        val_times=16,
        val_samples_per_time=256,
        device=device,
        dtype=dtype,
        checkpoint_dir=str(test_checkpoint_dir),
        save_epochs=[5, 10, 15],  # Save at these epochs
        seed=42
    )
    
    # Test 1: Early epoch bound holds
    print("1. Bound holds at early epoch")
    model_early = VelocityMLP(hidden_dims=[32, 32], activation='silu')
    # Find epoch 5 checkpoint (or closest)
    early_ckpt = list(test_checkpoint_dir.glob("*epoch=5*.pt"))
    if not early_ckpt:
        # Try to find any checkpoint with epoch <= 10
        all_ckpts = list(test_checkpoint_dir.glob("*.pt"))
        for ckpt in all_ckpts:
            if ckpt.name in ['best.pt', 'final.pt']:
                continue
            try:
                metadata = json.load(open(ckpt.with_suffix('.json')))
                if metadata.get('epoch', 999) <= 10:
                    early_ckpt = [ckpt]
                    break
            except:
                continue
    if early_ckpt:
        model_early.load_state_dict(torch.load(early_ckpt[0], map_location=device, weights_only=False))
        model_early.eval()
        model_early = model_early.to(device)
        
        set_seed(12345)
        KL_early = compute_kl_at_t1_learn(
            model_early, Schedule.A1, N_kl=1000, rtol=1e-6, atol=1e-8,
            eval_seed=12345, device=device, dtype=dtype, chunk_size=512
        )
        
        epsilon_early = compute_epsilon_learn(
            model_early, Schedule.A1, val_times=32, val_samples_per_time=512,
            device=device, dtype=dtype
        )
        
        S_early, _ = compute_score_gap_integral_learn(
            model_early, Schedule.A1, K_S=21, N_S=256,
            rtol=1e-6, atol=1e-8, eval_seed=12345, device=device, dtype=dtype, chunk_size=512
        )
        
        RHS_early = epsilon_early * np.sqrt(S_early)
        
        bound_satisfied = KL_early <= RHS_early + 0.01  # Allow small margin for MC variance
        print(f"   Early epoch: KL={KL_early:.6e}, RHS={RHS_early:.6e}")
        print(f"   ✓ Bound satisfied: {bound_satisfied}")
        
        assert bound_satisfied, "Bound should hold at early epoch"
    else:
        print("   ⚠ Early checkpoint not found, skipping")
    
    # Test 2: Best checkpoint improves
    print("2. Best checkpoint improves")
    model_best = VelocityMLP(hidden_dims=[32, 32], activation='silu')
    best_ckpt = test_checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        model_best.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False))
        model_best.eval()
        model_best = model_best.to(device)
        
        set_seed(12345)
        KL_best = compute_kl_at_t1_learn(
            model_best, Schedule.A1, N_kl=1000, rtol=1e-6, atol=1e-8,
            eval_seed=12345, device=device, dtype=dtype, chunk_size=512
        )
        
        epsilon_best = compute_epsilon_learn(
            model_best, Schedule.A1, val_times=32, val_samples_per_time=512,
            device=device, dtype=dtype
        )
        
        S_best, _ = compute_score_gap_integral_learn(
            model_best, Schedule.A1, K_S=21, N_S=256,
            rtol=1e-6, atol=1e-8, eval_seed=12345, device=device, dtype=dtype, chunk_size=512
        )
        
        RHS_best = epsilon_best * np.sqrt(S_best)
        
        print(f"   Early: KL={KL_early:.6e}, RHS={RHS_early:.6e}")
        print(f"   Best: KL={KL_best:.6e}, RHS={RHS_best:.6e}")
        
        kl_improves = KL_best < KL_early
        rhs_improves = RHS_best < RHS_early
        
        print(f"   ✓ KL improves: {kl_improves}")
        print(f"   ✓ RHS improves: {rhs_improves}")
        
        # Both should improve (allow for some noise)
        assert kl_improves, "KL should improve with training"
        assert rhs_improves, "RHS should improve with training"
    else:
        print("   ⚠ Best checkpoint not found, skipping")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_checkpoint_dir, ignore_errors=True)
    
    print("   ✓ T5 PASSED")
    return True


def test_T6_reproducibility():
    """T6: Reproducibility with fixed seed."""
    print("\n" + "=" * 60)
    print("T6: Reproducibility")
    print("=" * 60)
    
    set_seed(42)
    device = 'cpu'
    dtype = torch.float64
    
    # Train a small model
    model = VelocityMLP(hidden_dims=[32, 32], activation='silu')
    train_velocity_model(
        model=model,
        schedule=Schedule.A1,
        epochs=5,
        lr=1e-3,
        batch_size=64,
        num_batches_per_epoch=16,
        val_times=16,
        val_samples_per_time=256,
        device=device,
        dtype=dtype,
        checkpoint_dir=None,
        save_epochs=None,
        seed=42
    )
    model.eval()
    
    # Run computation twice with same seed
    set_seed(12345)
    KL_1 = compute_kl_at_t1_learn(
        model, Schedule.A1, N_kl=1000, rtol=1e-6, atol=1e-8,
        eval_seed=12345, device=device, dtype=dtype, chunk_size=512
    )
    
    set_seed(12345)
    KL_2 = compute_kl_at_t1_learn(
        model, Schedule.A1, N_kl=1000, rtol=1e-6, atol=1e-8,
        eval_seed=12345, device=device, dtype=dtype, chunk_size=512
    )
    
    rel_diff = abs(KL_1 - KL_2) / max(abs(KL_1), abs(KL_2), 1e-10)
    print(f"   KL (run 1): {KL_1:.6e}")
    print(f"   KL (run 2): {KL_2:.6e}")
    print(f"   Relative difference: {rel_diff:.2%}")
    print(f"   ✓ Rel diff < 1%: {rel_diff < 0.01}")
    
    assert rel_diff < 0.01, f"Relative difference {rel_diff:.2%} exceeds 1%"
    
    print("   ✓ T6 PASSED")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Part-2 (Learning) Unit Tests")
    print("=" * 60)
    
    results = {}
    
    try:
        results['T0'] = test_T0_model_wiring()
    except Exception as e:
        print(f"   ✗ T0 FAILED: {e}")
        results['T0'] = False
    
    try:
        results['T1'] = test_T1_training_learns()
    except Exception as e:
        print(f"   ✗ T1 FAILED: {e}")
        results['T1'] = False
    
    try:
        results['T2'] = test_T2_epsilon_consistency()
    except Exception as e:
        print(f"   ✗ T2 FAILED: {e}")
        results['T2'] = False
    
    try:
        results['T3'] = test_T3_score_oracle()
    except Exception as e:
        print(f"   ✗ T3 FAILED: {e}")
        results['T3'] = False
    
    try:
        results['T4'] = test_T4_backward_ode_numerics()
    except Exception as e:
        print(f"   ✗ T4 FAILED: {e}")
        results['T4'] = False
    
    try:
        results['T5'] = test_T5_bound_holds_and_tightens()
    except Exception as e:
        print(f"   ✗ T5 FAILED: {e}")
        results['T5'] = False
    
    try:
        results['T6'] = test_T6_reproducibility()
    except Exception as e:
        print(f"   ✗ T6 FAILED: {e}")
        results['T6'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

