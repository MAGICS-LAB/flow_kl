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

from core.true_path import schedule_to_enum, velocity_u, sample_p_t, get_schedule_functions


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
    from core.true_path import sample_p_t as sample_for_plot, velocity_u as velocity_for_plot

    model.eval()
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_samples = torch.zeros(len(time_points), num_samples, 2, dtype=dtype, device=device)
    u_true = torch.zeros_like(x_samples)
    v_pred = torch.zeros_like(x_samples)

    for idx, t in enumerate(time_points):
        x_samples[idx] = sample_for_plot(t, num_samples, schedule, device=device, dtype=dtype)
        u_true[idx] = velocity_for_plot(x_samples[idx], torch.tensor(t, dtype=dtype, device=device), schedule)
        v_pred[idx] = model(x_samples[idx], torch.tensor(t, dtype=dtype, device=device))

    # Plotting
    plt.figure(figsize=(18, 8))
    for idx, t in enumerate(time_points):
        plt.subplot(2, 3, idx + 1)
        x_np = x_samples[idx].cpu().numpy()
        u_np = u_true[idx].cpu().numpy()
        v_np = v_pred[idx].cpu().numpy()
        plt.quiver(x_np[:, 0], x_np[:, 1], u_np[:, 0], u_np[:, 1], color='blue', alpha=0.6, label='True velocity')
        plt.quiver(x_np[:, 0], x_np[:, 1], v_np[:, 0], v_np[:, 1], color='red', alpha=0.6, label='Learned velocity')
        plt.title(f'Time t = {t}', fontsize=16)
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='upper center', ncol=2, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    Path('data/plots').mkdir(parents=True, exist_ok=True)
    plot_path = Path('data/plots') / f'velocity_comparison_{schedule.value}.png'
    plt.savefig(plot_path, dpi=225, bbox_inches='tight')
    print(f"Saved velocity comparison plot to {plot_path}")
    plt.close()


def validate_model(model, schedule, val_times, val_samples_per_time, device='cpu', dtype=torch.float64):
    """
    Validate the model by computing MSE and normalized MSE on random samples.

    Args:
        model: VelocityMLP model
        schedule: Schedule enum
        val_times: Number of time points for validation
        val_samples_per_time: Samples per time point
        device: torch device
        dtype: torch dtype

    Returns:
        val_mse: Validation MSE
        val_nmse: Normalized MSE (relative to true velocity magnitude)
    """
    model.eval()

    mse_acc = []
    nmse_acc = []

    for _ in range(val_times):
        t = torch.rand(val_samples_per_time, dtype=dtype, device=device)
        x = torch.zeros(val_samples_per_time, 2, dtype=dtype, device=device)

        for i in range(val_samples_per_time):
            x[i] = sample_p_t(t[i].item(), 1, schedule, device=device, dtype=dtype).squeeze(0)

        u = velocity_u(x, t, schedule)
        v = model(x, t)

        mse = torch.mean((v - u) ** 2).item()
        nmse = torch.mean((v - u) ** 2).item() / torch.mean(u ** 2).item()

        mse_acc.append(mse)
        nmse_acc.append(nmse)

    val_mse = float(np.mean(mse_acc))
    val_nmse = float(np.mean(nmse_acc))

    return val_mse, val_nmse

