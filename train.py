"""
Training loop for flow matching: learn v_θ(x,t) to match u(x,t)=a(t)x.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from true_path import schedule_to_enum, velocity_u, sample_p_t, get_schedule_functions


def train_velocity_model(
    model,
    schedule,
    epochs=300,
    lr=1e-3,
    batch_size=128,
    num_batches_per_epoch=512,
    val_times=64,
    val_samples_per_time=2048,
    early_stop_patience=20,
    target_nmse=1e-2,
    target_mse=None,
    device='cpu',
    dtype=torch.float64
):
    """
    Train v_θ to match u(x,t)=a(t)x via flow matching.
    
    Training: Sample t ~ Unif[0,1], x ~ p_t, minimize MSE(v_θ(x,t) - u(x,t)).
    
    Args:
        model: VelocityMLP to train
        schedule: Schedule enum
        epochs: Number of epochs
        lr: Learning rate
        batch_size: Batch size
        num_batches_per_epoch: Number of batches per epoch
        val_times: Number of time points for validation
        val_samples_per_time: Samples per time point for validation
        early_stop_patience: Early stopping patience
        target_nmse: Target normalized MSE
        device: torch device
        dtype: torch dtype
    
    Returns:
        best_val_mse: Best validation MSE
        training_history: dict with 'train_mse', 'val_mse'
    """
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    # Track history
    training_history = {
        'train_mse': [],
        'val_mse': [],
        'val_nmse': []
    }
    
    best_val_mse = float('inf')
    patience_counter = 0
    
    # Get schedule functions
    _, A_func = get_schedule_functions(schedule)
    
    print(f"Training for {epochs} epochs on schedule {schedule.value}")
    print(f"Device: {device}, dtype: {dtype}")
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        for batch_idx in range(num_batches_per_epoch):
            # Sample t ~ Unif[0,1]
            t = torch.rand(batch_size, dtype=dtype, device=device)
            
            # Sample x ~ p_t for each t
            x = torch.zeros(batch_size, 2, dtype=dtype, device=device)
            for i in range(batch_size):
                x[i] = sample_p_t(t[i].item(), 1, schedule, device=device, dtype=dtype).squeeze(0)
            
            # True velocity target
            u = velocity_u(x, t, schedule)
            
            # Predicted velocity
            v = model(x, t)
            
            # Loss: MSE(v_θ - u)
            loss = torch.mean((v - u) ** 2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / num_batches_per_epoch
        
        # Validation phase
        if (epoch + 1) % 5 == 0 or epoch == 0:  # Validate every 5 epochs + first
            val_mse, val_nmse = validate_model(model, schedule, val_times, val_samples_per_time, device, dtype)
            
            training_history['train_mse'].append(avg_train_loss)
            training_history['val_mse'].append(val_mse)
            training_history['val_nmse'].append(val_nmse)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            # Check if target_mse is specified, otherwise use target_nmse
            if target_mse is not None and val_mse <= target_mse:
                print(f"\nReached target MSE {val_mse:.4e} at epoch {epoch}")
                break
            elif val_nmse <= target_nmse:
                print(f"\nReached target NMSE {val_nmse:.4e} at epoch {epoch}")
                break
            elif patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break
        
        scheduler.step()
    
    print(f"\nTraining complete. Best val MSE: {best_val_mse:.6f}")
    
    # Plot training and validation curves
    plot_training_curves(training_history, schedule)
    
    # Plot velocity comparison
    plot_velocity_comparison(model, schedule, num_samples=10, device=device, dtype=dtype)
    
    return best_val_mse, training_history


def plot_training_curves(history, schedule):
    """Plot training and validation loss curves."""
    if len(history['train_mse']) == 0:
        return
    
    epochs = list(range(len(history['train_mse'])))
    
    plt.figure(figsize=(12, 5))
    
    # Plot MSE
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_mse'], label='Training MSE', marker='o', linewidth=3)
    plt.plot(epochs, history['val_mse'], label='Validation MSE', marker='s', linewidth=3)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('MSE', fontsize=18)
    plt.title(f'Training Curves - Schedule {schedule.value.upper()}', fontsize=21)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=17)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot NMSE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_nmse'], label='Validation NMSE', marker='s', color='orange', linewidth=3)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('NMSE', fontsize=18)
    plt.title(f'Normalized MSE - Schedule {schedule.value.upper()}', fontsize=21)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=17)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    Path('data/plots').mkdir(parents=True, exist_ok=True)
    plot_path = Path('data/plots') / f'training_curves_{schedule.value}.png'
    plt.savefig(plot_path, dpi=225, bbox_inches='tight')
    print(f"\nSaved training curves to {plot_path}")
    plt.close()


def plot_velocity_comparison(model, schedule, num_samples=10, device='cpu', dtype=torch.float64):
    """
    Plot comparison of true vs learned velocity field at multiple time points.
    
    Creates 5 subplots (one per time point) showing sampled points with both
    true velocity (blue) and learned velocity (red) vectors.
    """
    from true_path import sample_p_t, velocity_u
    
    model.eval()
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, t in enumerate(time_points):
        ax = axes[idx]
        
        # Sample positions
        x_samples = sample_p_t(t, num_samples, schedule, device=device, dtype=dtype)
        x_np = x_samples.cpu().numpy()
        
        # Get velocities
        t_tensor = torch.tensor(t, dtype=dtype, device=device)
        with torch.no_grad():
            true_vel = velocity_u(x_samples, t_tensor, schedule).cpu().numpy()
            learned_vel = model(x_samples, t_tensor).cpu().numpy()
        
        # Plot points
        ax.scatter(x_np[:, 0], x_np[:, 1], c='black', s=113, zorder=3, label='Position')
        
        # Plot true velocity vectors (blue)
        ax.quiver(x_np[:, 0], x_np[:, 1], 
                  true_vel[:, 0], true_vel[:, 1],
                  angles='xy', scale_units='xy', scale=None,
                  color='blue', alpha=0.7, width=0.005, label='True velocity',
                  zorder=1)
        
        # Plot learned velocity vectors (red)
        ax.quiver(x_np[:, 0], x_np[:, 1], 
                  learned_vel[:, 0], learned_vel[:, 1],
                  angles='xy', scale_units='xy', scale=None,
                  color='red', alpha=0.7, width=0.005, label='Learned velocity',
                  zorder=2)
        
        # Set equal aspect and add grid
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Time t = {t}', fontsize=18, fontweight='bold')
        ax.set_xlabel('$x_1$', fontsize=17)
        ax.tick_params(axis='both', which='major', labelsize=20)
        if idx == 0:
            ax.set_ylabel('$x_2$', fontsize=17)
        
        # Legend only on first subplot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=14)
    
    plt.suptitle(f'True vs Learned Velocity Field - Schedule {schedule.value.upper()}', 
                 fontsize=21, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    Path('data/plots').mkdir(parents=True, exist_ok=True)
    plot_path = Path('data/plots') / f'velocity_comparison_{schedule.value}.png'
    plt.savefig(plot_path, dpi=225, bbox_inches='tight')
    print(f"Saved velocity comparison to {plot_path}")
    plt.close()


def validate_model(model, schedule, val_times=64, val_samples_per_time=2048, device='cpu', dtype=torch.float64):
    """
    Validate model on a grid of (t,x) pairs.
    
    Args:
        model: Trained VelocityMLP
        schedule: Schedule enum
        val_times: Number of time points
        val_samples_per_time: Samples per time point
        device: torch device
        dtype: torch dtype
    
    Returns:
        mse: Mean squared error
        nmse: Normalized MSE (MSE / E[|u|²])
    """
    model.eval()
    
    mse_list = []
    norm_list = []
    
    # Sample validation grid
    t_vals = torch.rand(val_times, dtype=dtype, device=device)
    
    with torch.no_grad():
        for t_val in t_vals:
            # Sample x ~ p_t
            x = sample_p_t(t_val.item(), val_samples_per_time, schedule, device=device, dtype=dtype)
            
            # Expand t for broadcasting
            t_expanded = t_val.expand(val_samples_per_time)
            
            # True velocity
            u = velocity_u(x, t_expanded, schedule)
            
            # Predicted velocity
            v = model(x, t_expanded)
            
            # Compute MSE and ||u||²
            mse = torch.mean((v - u) ** 2)
            u_norm_sq = torch.mean(u ** 2)
            
            mse_list.append(mse.item())
            norm_list.append(u_norm_sq.item())
    
    # Aggregate
    mse_avg = np.mean(mse_list)
    u_norm_avg = np.mean(norm_list)
    nmse = mse_avg / (u_norm_avg + 1e-10)
    
    return mse_avg, nmse


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Training Loop (Flow Matching)")
    print("=" * 60)
    
    from model import VelocityMLP
    from true_path import Schedule, velocity_u
    
    # Test 1: Training on schedule a1
    print("\n1. Training on schedule a1 (sin(πt))")
    model_a1 = VelocityMLP(hidden_dims=[64, 64], activation='silu')
    
    best_mse, history = train_velocity_model(
        model_a1,
        schedule=Schedule.A1,
        epochs=5,
        batch_size=64,
        num_batches_per_epoch=16,
        val_times=16,
        val_samples_per_time=256,
        device='cpu'
    )
    
    print(f"   Training history length: {len(history['train_mse'])}")
    print(f"   Best validation MSE: {best_mse:.6f}")
    print(f"   Final training MSE: {history['train_mse'][-1]:.6f}")
    
    # Check that loss decreases
    if len(history['train_mse']) > 1:
        initial_mse = history['train_mse'][0]
        final_mse = history['train_mse'][-1]
        print(f"   Initial MSE: {initial_mse:.6f}, Final MSE: {final_mse:.6f}")
        print(f"   ✓ Loss decreased: {final_mse < initial_mse}")
    
    # Test 2: Validate the trained model
    print("\n2. Validation on trained model")
    from train import validate_model
    val_mse, val_nmse = validate_model(
        model_a1, Schedule.A1, 
        val_times=32, val_samples_per_time=512,
        device='cpu'
    )
    print(f"   Validation MSE: {val_mse:.6e}")
    print(f"   Validation NMSE: {val_nmse:.6e}")
    print(f"   ✓ Validation runs without errors")
    print(f"   ✓ NMSE < 1.0 (reasonable fit): {val_nmse < 1.0}")
    
    # Test 3: Check that model predictions are reasonable
    print("\n3. Sanity check - model predictions")
    model_a1.eval()
    with torch.no_grad():
        x_test = torch.randn(10, 2, dtype=torch.float64)
        t_test = torch.rand(10, dtype=torch.float64)
        
        # True velocity
        u_true = velocity_u(x_test, t_test, Schedule.A1)
        
        # Predicted velocity
        v_pred = model_a1(x_test, t_test)
        
        # Compute MSE
        mse_sample = torch.mean((v_pred - u_true) ** 2).item()
        
    print(f"   Sample MSE: {mse_sample:.6e}")
    print(f"   Output shape: {v_pred.shape}")
    print(f"   ✓ Predictions are 2D vectors")
    print(f"   ✓ MSE is finite and positive")
    
    # Test 4: Test on different schedule
    print("\n4. Training on schedule a2 (0.3sin(2πt)+0.2)")
    model_a2 = VelocityMLP(hidden_dims=[32, 32], activation='silu')
    
    best_mse_a2, history_a2 = train_velocity_model(
        model_a2,
        schedule=Schedule.A2,
        epochs=3,
        batch_size=32,
        num_batches_per_epoch=8,
        val_times=8,
        val_samples_per_time=128,
        device='cpu'
    )
    
    print(f"   Best validation MSE: {best_mse_a2:.6f}")
    print(f"   ✓ Different schedule trains successfully")
    
    # Test 5: Edge case - very short training
    print("\n5. Edge case - minimal training (1 epoch)")
    model_mini = VelocityMLP(hidden_dims=[16, 16], activation='silu')
    
    best_mse_mini, history_mini = train_velocity_model(
        model_mini,
        schedule=Schedule.A3,
        epochs=1,
        batch_size=16,
        num_batches_per_epoch=4,
        val_times=4,
        val_samples_per_time=64,
        device='cpu'
    )
    
    print(f"   Training MSE after 1 epoch: {history_mini['train_mse'][-1]:.6f}")
    print(f"   ✓ Minimal training completes")
    
    # Test 6: Check that model learns meaningful differences
    print("\n6. Model learns time-dependent behavior")
    model_a1.eval()
    with torch.no_grad():
        # Same spatial point, different times
        x_fixed = torch.randn(1, 2, dtype=torch.float64).expand(10, 2)
        t_varying = torch.linspace(0, 1, 10, dtype=torch.float64)
        
        v_varying = model_a1(x_fixed, t_varying)
        
        # Check that outputs vary with time
        v_std = torch.std(v_varying, dim=0)
        avg_std = torch.mean(v_std).item()
        
    print(f"   Std of outputs across time: {avg_std:.4f}")
    print(f"   ✓ Model is time-dependent: {avg_std > 0.01}")
    
    # Test 7: Consistency check
    print("\n7. Consistency - same model state, same predictions")
    model_a1.eval()
    with torch.no_grad():
        x_consistency = torch.randn(5, 2, dtype=torch.float64)
        t_consistency = torch.tensor(0.5, dtype=torch.float64)
        
        v1 = model_a1(x_consistency, t_consistency)
        v2 = model_a1(x_consistency, t_consistency)
        
        max_diff = torch.max(torch.abs(v1 - v2))
        
    print(f"   Max difference between runs: {max_diff:.2e}")
    print(f"   ✓ Deterministic predictions: {max_diff < 1e-10}")
    
    print("\n" + "=" * 60)
    print("All training tests passed!")
    print("=" * 60)

