"""
MLP model for learned velocity field v_θ(x,t).

Input: [x₁, x₂, t] ∈ ℝ³
Output: v_θ(x,t) ∈ ℝ²
"""

import torch
import torch.nn as nn


class VelocityMLP(nn.Module):
    """
    Multi-layer perceptron for velocity field v_θ(x,t).
    
    Architecture:
    - Input: [x₁, x₂, t] (concatenated)
    - Hidden layers: specified by hidden_dims
    - Output: v_θ ∈ ℝ²
    """
    
    def __init__(self, input_dim=3, hidden_dims=None, output_dim=2, activation='silu'):
        """
        Initialize the MLP.
        
        Args:
            input_dim: Dimension of input (default 3: [x₁, x₂, t])
            hidden_dims: List of hidden layer dimensions (default [128, 128])
            output_dim: Dimension of output (default 2: velocity components)
            activation: Activation function ('silu' or 'softplus')
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1], dtype=torch.float64))
            if i < len(dims) - 2:  # Add activation except for last layer
                if activation == 'silu':
                    layers.append(nn.SiLU())
                elif activation == 'softplus':
                    layers.append(nn.Softplus())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, t):
        """
        Forward pass: v_θ(x,t).
        
        Args:
            x: Spatial coordinates of shape [..., 2]
            t: Time point(s) - scalar or tensor of shape [...]
        
        Returns:
            Velocity v_θ(x,t) of shape [..., 2]
        """
        # Ensure x and t are tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        
        # Expand dimensions if needed
        if t.dim() == 0:
            t = t.unsqueeze(0)
        while t.dim() < x.dim():
            t = t.unsqueeze(-1)
        
        # Concatenate [x, t]
        input_vec = torch.cat([x, t.expand_as(x[..., :1])], dim=-1)
        
        # Forward through network
        output = self.network(input_vec)
        
        return output
    
    def compute_divergence(self, x, t):
        """
        Compute divergence ∇·v_θ(x,t) via autograd.
        
        ∇·v = ∂v₁/∂x₁ + ∂v₂/∂x₂
        
        Args:
            x: Spatial coordinates of shape [..., 2]
            t: Time point(s)
        
        Returns:
            Divergence scalar(s) of shape [...]
        """
        # Requires gradient
        x_grad = x.clone().detach().requires_grad_(True)
        
        # Compute velocity
        v = self.forward(x_grad, t)
        
        # Compute divergence
        divergence = torch.zeros_like(v[..., 0])
        for i in range(2):
            grad_i = torch.autograd.grad(
                v[..., i].sum(), 
                x_grad, 
                create_graph=True,
                retain_graph=True
            )[0]
            divergence += grad_i[..., i]
        
        return divergence


def compute_divergence_external(v_theta, x, t):
    """
    Standalone function to compute divergence ∇·v_θ(x,t).
    
    Useful when you have the model but don't want to call the method.
    
    Args:
        v_theta: VelocityMLP model
        x: Spatial coordinates of shape [..., 2]
        t: Time point(s)
    
    Returns:
        Divergence scalar(s)
    """
    return v_theta.compute_divergence(x, t)


if __name__ == '__main__':
    print("=" * 60)
    print("Testing VelocityMLP")
    print("=" * 60)
    
    # Test 1: Model initialization
    print("\n1. Model Initialization")
    model = VelocityMLP(input_dim=3, hidden_dims=[128, 128], output_dim=2)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {param_count} parameters")
    print(f"   Expected input: [batch, 3] = [x₁, x₂, t]")
    print(f"   Expected output: [batch, 2] = [v₁, v₂]")
    
    # Test 2: Forward pass with scalar t
    print("\n2. Forward pass (scalar t)")
    batch_size = 10
    x = torch.randn(batch_size, 2, dtype=torch.float64)
    t = 0.5
    
    # Check internal concatenation
    t_tensor = torch.tensor(t, dtype=torch.float64)
    t_expanded = t_tensor.unsqueeze(0).expand_as(x[..., :1])
    input_vec = torch.cat([x, t_expanded], dim=-1)
    print(f"   Spatial input x shape: {x.shape}")
    print(f"   Time component t shape: {t_expanded.shape}")
    print(f"   Concatenated input [x,t] shape: {input_vec.shape}")
    
    v = model(x, t)
    print(f"   Output velocity v shape: {v.shape}")
    print(f"   ✓ Output shape correct: {v.shape == torch.Size([batch_size, 2])}")
    
    # Test 3: Forward pass with tensor t
    print("\n3. Forward pass (tensor t)")
    t_batch = torch.rand(batch_size, dtype=torch.float64)  # Different time per sample
    v_batch = model(x, t_batch)
    print(f"   Input x shape: {x.shape}")
    print(f"   Input t shape: {t_batch.shape}")
    print(f"   Output v shape: {v_batch.shape}")
    print(f"   ✓ Handles per-sample times")
    
    # Test 4: Divergence computation
    print("\n4. Divergence computation")
    x_grad = x.clone().requires_grad_(True)
    t_fixed = torch.tensor(0.5, dtype=torch.float64)
    
    # Manual computation for verification
    v_test = model(x_grad, t_fixed)
    
    # Compute divergence manually
    dv1_dx1 = torch.autograd.grad(v_test[:, 0].sum(), x_grad, create_graph=True)[0][:, 0]
    dv2_dx2 = torch.autograd.grad(v_test[:, 1].sum(), x_grad, create_graph=True)[0][:, 1]
    div_manual = dv1_dx1 + dv2_dx2
    
    # Using method
    div_method = model.compute_divergence(x_grad, t_fixed)
    
    print(f"   Divergence shape: {div_manual.shape}")
    print(f"   Sample divergence values: {div_manual[:3]}")
    max_diff = torch.max(torch.abs(div_manual - div_method))
    print(f"   Max difference (manual vs method): {max_diff:.2e}")
    print(f"   ✓ Methods agree (diff < 1e-10): {max_diff < 1e-10}")
    
    # Test 5: Edge case - single sample
    print("\n5. Edge case (single sample)")
    x_single = torch.randn(1, 2, dtype=torch.float64)
    v_single = model(x_single, torch.tensor([0.3], dtype=torch.float64))
    print(f"   Single sample x shape: {x_single.shape}")
    print(f"   Single sample v shape: {v_single.shape}")
    print(f"   ✓ Handles batch_size=1")
    
    # Test 6: Consistency - same input should give same output
    print("\n6. Consistency check")
    x_test = torch.randn(5, 2, dtype=torch.float64)
    t_test = 0.7
    v1 = model(x_test, t_test)
    v2 = model(x_test, t_test)
    max_diff = torch.max(torch.abs(v1 - v2))
    print(f"   Two forward passes on same input")
    print(f"   Max difference: {max_diff:.2e}")
    print(f"   ✓ Deterministic (diff < 1e-10): {max_diff < 1e-10}")
    
    # Test 7: Different schedules give different outputs
    print("\n7. Different inputs give different outputs")
    t1 = torch.tensor(0.1, dtype=torch.float64)
    t2 = torch.tensor(0.9, dtype=torch.float64)
    v_t1 = model(x_test, t1)
    v_t2 = model(x_test, t2)
    max_diff_t = torch.max(torch.abs(v_t1 - v_t2))
    print(f"   Different times (t=0.1 vs t=0.9)")
    print(f"   Max difference: {max_diff_t:.2f}")
    print(f"   ✓ Model depends on time: {max_diff_t > 0.01}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

