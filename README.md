# Flow Matching KL Divergence Bounds – Experiments

This repository hosts the implementation of the numerical studies reported in “On Flow Matching KL Divergence,” Su et al., 2025. The code reproduces the empirical evidence supporting the paper’s deterministic KL error bounds and statistical convergence guarantees for flow matching.

For reference, the paper builds on the KL evolution identity:
```
KL(p_t|q_t) = ∫₀ᵗ E_x~p_t[(u(x,s) - v_θ(x,s))ᵀ(∇log p_s(x) - ∇log q_s(x))] ds
```

where:
- `p_t` evolves under velocity field `u(x,t) = a(t) x`
- `q_t` evolves under learned velocity field `v_θ(x,t)`
- Both start as standard Gaussians: `p_0 = q_0 = N(0, I)`

## Project Structure

```
fm-kl-2/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Core Modules
│   ├── true_path.py            # True distribution p_t storming (schedules, sampling, densities)
│   ├── model.py                # Neural network v_θ(x,t)
│   ├── train.py                # Training loop for v_θ
│   ├── eval.py                 # Evaluation: backward ODE, KL LHS, RHS
│   ├── utils.py                # Utilities: plotting, saving, error computation
│   └── experiment.py           # Main orchestration script
│
├── Testing
│   ├── test_golden_path.py     # LHS pipeline tests (A1-A3, B1)
│   └── test_rhs.py             # RHS pipeline tests (R0-R3)
│
├── No-Learning Verification
│   ├── nolearning_test.py      # Closed-form KL identity tests
│   └── run_all_nolearning.sh   # Run all schedule permutations
│
├── Part 2: Synthetic Bound Verification
│   ├── synthetic_velocity.py        # Synthetic velocity fields v(x,t)
│   ├── eval_pt2.py                  # Part 2 evaluation functions
│   ├── experiment_pt2.py            # Part 2 orchestrator
│   ├── test_pt2.py                  # Part 2 unit tests
│   └── run_all_pt2_experiments.py   # Automated Part 2 experiments
│
├── Part 2 (Learning): Learned Bound Verification
│   ├── model_learn_pt2.py          # Velocity MLP (copy)
│   ├── train_learn_pt2.py          # Training with checkpointing
│   ├── eval_learn_pt2.py           # Learned evaluation functions
│   ├── experiment_learn_pt2.py     # Part 2 Learning orchestrator
│   └── test_learn_pt2.py           # Part 2 Learning unit tests
│
├── Automated Experiments
│   ├── run_all_experiments.py  # Train and evaluate all schedules
│   └── run_all_cross_eval.sh   # Cross-schedule evaluations
│
├── Plotting Utilities
│   ├── plot_eps_curves.py      # Generate ε-curves used in Section 5
│   ├── regenerate_plots.py     # Rebuild KL identity plots from saved JSON data
│   └── regenerate_plots_from_csv.py  # Rebuild Part 2 plots from cleaned CSV files
│
└── (output artifacts are written to a data directory)
```

## Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Create a conda environment:**
```bash
conda create -n flow-kl python=3.10
conda activate flow-kl
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run the closed-form no-learning test

This verifies the identity using analytic formulas (no neural networks):

```bash
conda activate flow-kl
python nolearning_test.py --schedule_p a1 --schedule_q aaffen
```

For all 6 schedule permutations:
```bash
bash run_all_nolearning.sh
```

### 2. Train and evaluate a model

Train a model to learn velocity field `v_θ`:

```bash
python experiment.py --schedule a1 --target_mse 0.05
```

This will reproduce the Section 5.2 checkpoints:
- Train a neural network to match the true velocity
- Write checkpoints, metrics, and plots into the configured output directory 
- Evaluate the KL identity
- Generate plots showing LHS vs RHS

### 3. Load a trained model and re-evaluate

```bash
python experiment.py --schedule a1 --load_model path/to/vtheta_schedule_a1_mse_0-05_TIMESTAMP.pth
```

### 4. Part 2: Synthetic Bound Verification

Validate the bound `KL(p₁|q₁) ≤ ε√S` using synthetic velocity fields:

```bash
python experiment_pt2.py --schedule a1 --delta_type constant --delta_beta 0.0 0.05 0.1 0.2
```

For oscillatory perturbations:
```bash
python experiment_pt2.py --schedule a1 --delta_type sine --delta_beta 0.025 0.05 0.075 0.1
```

Run all Part 2 experiments:
```bash
python run_all_pt2_experiments.py
```

### 5. Part 2 (Learning): Learned Bound Verification

Train a velocity MLP and verify the bound across training checkpoints:

```bash
python experiment_learn_pt2.py --schedule a1 --epochs 400 --eval_checkpoints "all"
```

This will:
- Train a neural network for up to 400 epochs
- Save multiple checkpoints (best, final, and on improvement)
- Evaluate the bound for all saved checkpoints
- Generate scatter plots showing bound tightening with training

## Core Concepts

### The Identity

For time-evolving distributions:
- **LHS**: KL divergence `KL(p_t|q_t)`
- **RHS**: Integral of local misalignment between velocity fields and score functions

The identity states these are equal for all `t ∈ [0,1]`.

### Three Velocity Schedules

The code supports three linear velocity schedules:

- **a1(t)**: `sin(πt)`
- **a2(t)**: `0.3 sin(2πt) + 0.2`
- **a3(t)**: `t - 1/2`

Each schedule generates a different time-evolving Gaussian distribution.

### Evaluation Pipeline

**LHS computation:**
1. Sample `x ~ p_t`
2. Solve backward ODE `ẋ(s) = v_θ(x(s), s)` from `s=t` to `s=0`
3. Accumulate divergence: `ℓ = ∫₀ᵗ ∇·v_θ ds`
4. Compute `log q_t(x) = log p_0(x₀) - ℓ`
5. Compute score `∇log q_t(x)` via autograd
6. Estimate `KL(p_t|q_t) = E[log p_t(x) - log q_t(x)]`

**RHS computation:**
1. Sample `x ~ p_t`
2. Compute true velocity `u(x,t)` and learned velocity `v_θ(x,t)`
3. Compute true score `∇log p_t(x)` and learned score `∇log q_t(x)`
4. Evaluate integrand `(u-v)ᵀ(s_p - s_q)`
5. Integrate over time using trapezoidal rule

## Command-Line Interface

### `experiment.py` - Main experiment script

**Basic usage:**
```bash
python experiment.py --schedule {a1,a2,a3} [OPTIONS]
```

**Key arguments:**
- `--schedule`: Which velocity schedule to use (required)
- `--seed`: Random seed (default: 42)
- `--target_mse`: Stop training when validation MSE reaches this value
- `--target_nmse`: Stop training when normalized MSE reaches this (default: 1e-2)
- `--load_model`: Path to saved model (skips training)
- `--epochs`: Max training epochs (default: 300)
- `--num_samples`: Samples per time point for evaluation (default: 2000)
- `--num_times`: Number of time points (default: 101)
- `--num_seeds`: Number of seeds to average over (default: 1)
- `--rtol`, `--atol`: ODE solver tolerances (default: 1e-6, 1e-8)

**Examples:**
```bash
# Train on schedule a1 with target MSE 0.05
python experiment.py --schedule a1 --target_mse 0.05

# High-resolution evaluation with tighter tolerances
python experiment.py --schedule a1 --num_samples 4000 --num_times 201 --rtol 1e-7 --atol 1e-9 \
    --load_model data/models/vtheta_schedule_a1_mse_0-05_TIMESTAMP.pth

# Reduce variance by averaging over 3 seeds
python experiment.py --schedule a1 --num_seeds 3 --load_model MODEL_PATH
```

### `nolearning_test.py` - Closed-form verification

**Usage:**
```bash
python nolearning_test.py --schedule_p {a1,a2,a3} --schedule_q {a1,a2,a3} [--skip_ode]
```

**Arguments:**
- `--schedule_p`: Schedule for distribution `p_t` (default: a1)
- `--schedule_q`: Schedule for distribution `q_t` (default: a2)
- `--skip_ode`: Skip ODE pipeline test for faster runs

### `experiment_pt2.py` - Part 2 Synthetic Bound Verification

**Usage:**
```bash
python experiment_pt2.py --schedule {a1,a2,a3} --delta_type {constant,sine} --delta_beta 0.0 0.05 0.1 [OPTIONS]
```

**Key arguments:**
- `--schedule`: Velocity schedule (required)
- `--delta_type`: Perturbation type: `constant` (δ(t)=β) or `sine` (δ(t)=β sin(2πt))
- `--delta_beta`: List of β values for perturbations (repeatable)
- `--K_eps`: Time points for ε computation (default: 101)
- `--N_eps`: Samples per time for ε (default: 4096)
- `--K_S`: Time points for S computation (default: 101)
- `--N_S`: Samples per time for S (default: 2048)
- `--N_kl`: Samples for KL at t=1 (default: 20000)
- `--rtol`, `--atol`: ODE tolerances (default: 1e-6, 1e-8)

**Examples:**
```bash
# Constant perturbations on a1
python experiment_pt2.py --schedule a1 --delta_type constant --delta_beta 0.0 0.05 0.1 0.2

# Sine perturbations on a2
python experiment_pt2.py --schedule a2 --delta_type sine --delta_beta 0.025 0.05 0.075 0.1
```

### `experiment_learn_pt2.py` - Part 2 Learned Bound Verification

**Usage:**
```bash
python experiment_learn_pt2.py --schedule {a1,a2,a3} [OPTIONS]
```

**Training arguments:**
- `--epochs`: Max training epochs (default: 400)
- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Training batch size (default: 4)
- `--batches_per_epoch`: Batches per epoch (default: 2)
- `--val_times`: Validation time points (default: 64)
- `--val_samples_per_time`: Validation samples (default: 2048)

**Evaluation arguments:**
- `--eval_checkpoints`: Checkpoints to evaluate: `"final,best"` or `"all"` (default: final,best)
- `--eval_val_times`: Eval time points for ε_θ (default: 101)
- `--eval_val_samples_per_time`: Eval samples for ε_θ (default: 1024)
- `--eval_K_S`: Time points for S_θ (default: 101)
- `--eval_N_S`: Samples per time for S_θ (default: 512)
- `--eval_N_kl`: Samples for KL (default: 5000)
- `--eval_rtol`, `--eval_atol`: Eval ODE tolerances (default: 1e-6, 1e-8)
- `--eval_chunk_size`: Batch size for evaluation (default: 1024)
- `--eval_seed`: Random seed for evaluation (default: 12345)
- `--eval_only`: Skip training, only evaluate existing checkpoints

**Examples:**
```bash
# Full training + evaluation
python experiment_learn_pt2.py --schedule a1 --epochs 400 --eval_checkpoints "all"

# Evaluate existing checkpoints only
python experiment_learn_pt2.py --schedule a1 --eval_only --eval_checkpoints "all"
```

## Testing

### LHS Pipeline Tests (`test_golden_path.py`)

Golden-path tests using `v_θ = u` (the true velocity):

- **A1**: Preimage computation (backward ODE)
- **A2**: Divergence accumulation sign
- **A3**: Log-density computation
- **B1**: Normalization check

```bash
python test_golden_path.py
```

### RHS Pipeline Tests (`test_rhs.py`)

Tests for the RHS integrand computation:

- **R0**: Trivial identity (`v = u` → `g(t) ≡ 0`)
- **R1(a)**: Analytic check with `v ≡ 0`
- **R1(b)**: Analytic check with scaled field `v = c·u`
- **R2**: Derivative consistency with LHS
- **R3**: Internal decomposition sanity

```bash
python test_rhs.py
```

### No-Learning Tests (`nolearning_test.py`)

Closed-form KL identity verification without learning:

- Computes LHS using analytical KL formula
- Computes RHS using analytical integrand
- Verifies they match up to quadrature error
- Checks derivative consistency

```bash
python nolearning_test.py --schedule_p a1 --schedule_q a2
```

### Part 2 Tests (`test_pt2.py`)

Tests for synthetic velocity field validation:

- **Synthetic velocity properties**: Forward and divergence correctness
- **ODE reversibility**: Backward-forward consistency
- **Score correctness**: Oracle comparison for linear fields
- **ε checks**: RMS flow-matching error validation
- **KL at t=1 checks**: KL divergence computation accuracy
- **S convergence**: Score-gap integral convergence
- **Bound verification**: KL ≤ ε√S across perturbation types

```bash
python test_pt2.py
```

### Part 2 (Learning) Tests (`test_learn_pt2.py`)

Tests for learned model validation:

- **T0**: Model wiring (forward, divergence)
- **T1**: Training learns (loss decreases)
- **T2**: ε_θ consistency (validate_model vs direct MC)
- **T3**: Score oracle at small times
- **T4**: Backward-ODE numerics
- **T5**: Bound holds and tightens
- **T6**: Reproducibility

```bash
python test_learn_pt2.py
```

## Automated Experiment Scripts

### Train models for all schedules and target MSEs

```bash
python run_all_experiments.py
```

This trains models for schedules a1, a2, a3 with target MSEs: 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8.

### Cross-schedule evaluations

```bash
bash run_all_cross_eval.sh
```

Evaluates each schedule with models trained on *different* schedules (e.g., a1 vs a2-trained model).

### Part 2 experiments (Synthetic)

Run all synthetic bound verification experiments:

```bash
python run_all_pt2_experiments.py
```

This tests all combinations of:
- Schedules: a1, a2, a3
- Perturbation types: constant and sine
- Perturbation strengths: β ∈ [0, 0.2] for constant, β ∈ [0.025, 0.1] for sine

## Output Files

All experiments write their artifacts (checkpoints, numeric summaries, and figures) into a data directory. Typical files include:

- Model checkpoints (`*.pth`) containing network weights.
- NumPy arrays (`t_grid_*.npy`, `kl_curve_*.npy`, `rhs_integrand_*.npy`, `rhs_cumulative_*.npy`) plus accompanying metadata JSON.
- Figures such as `kl_comparison_*.png`, training curves, velocity comparisons, and Part 2 scatter/ε/f̂ plots.
- Plot-data JSON files (`kl_comparison_{schedule}_mse_{TARGET}_TIMESTAMP.json`) that allow reproducible regeneration of figures.

To regenerate plots with different styling:
```bash
# Part 1 plots (from JSON data)
python regenerate_plots.py

# Part 2 plots (from CSV files - works for both learned and synthetic)
python regenerate_plots_from_csv.py path/to/results.csv --schedule a1
```

### ε-Curves Plotting

The `plot_eps_curves.py` utility generates plots showing how LHS (KL divergence) and RHS (ε√S) components vary with ε (RMS flow-matching loss). This is useful for visualizing bound behavior across different checkpoints (learned) or perturbation strengths (synthetic).

**Usage:**
```bash
# For learned data
from plot_eps_curves import plot_lhs_rhs_vs_eps
plot_lhs_rhs_vs_eps('path/to/learned_results.csv',
                    'output.png', schedule='a1', ylog=True, annotate=True)

# For synthetic data
plot_lhs_rhs_vs_eps('path/to/synthetic_results.csv',
                    'output.png', schedule='a1', ylog=True, annotate=True)
```

The plots automatically:
- Detect CSV format (learned vs synthetic)
- Handle zero epsilon values for log scale
- Use dark green for LHS and dark red for RHS
- Apply log scales to both axes
- Annotate points with epochs (learned) or delta labels (synthetic)

### Part 2 Outputs
Synthetic experiments produce CSV/JSON summaries alongside scatter, bar, ε-curve, and f̂(t) figures for each perturbation setting. Learned experiments create schedule-specific bundles containing checkpoints (`ckpt__sched=*__epoch=*__valmse=*_TIMESTAMP.pt`), per-checkpoint metrics (`bound_*_TIMESTAMP.{csv,json}`), and plots (`bound_scatter_*_TIMESTAMP.png`, `eps_curves_*_TIMESTAMP.png`, `fhat_curves_*_TIMESTAMP.png`). These assets reside inside the data directory so you can regenerate or archive them without affecting the repository.

## Key Implementation Details

### Divergence Computation
The divergence `∇·v_θ` is computed analytically by summing the diagonal of the Jacobian matrix:

```python
div = sum(dv_i/dx_i) for i in range(d)
```

This is exact for neural networks and differentiable.

### Backward ODE
To compute `q_t(x)`, we solve the backward ODE:
```
ẋ(s) = v_θ(x(s), s)  [forward direction]
```

using `odeint` with reversed time: integrating from `s=t` down to `s=0`.

### Log-Density Computation
```
log q_t(x) = log p_0(x₀) - ℓ
```
where:
- `x₀` is the preimage (from backward ODE)
- `ℓ = ∫₀ᵗ ∇·v_θ ds` is the accumulated divergence

Note the **subtraction** sign (not addition), which accounts for the change of variables.

### Score Computation
The score `∇log q_t(x)` is computed by:
1. Setting `x.requires_grad_(True)`
2. Computing `log_q = log_q_t(x, ...)`
3. Taking autograd gradient: `grad(log_q.sum(), x)`

This gives the full Jacobian in one backprop pass.

## Mathematical Background

### Time-Evolving Distributions

Given velocity field `a(t)`, the distribution evolves as:
```
p_t = N(0, σ_p(t)² I)
```
where `σ_p(t) = e^A(t)` and `A(t) = ∫₀ᵗ a(s) ds`.

This describes a Gaussian that spreads over time according to the velocity schedule.

### KL Divergence Between Gaussian Distributions

For `p = N(0, σ_p²I)` and `q = N(0, σ_q²I)`:
```
KL(p|q) = (d/2) [r - 1 - log(r)]
```
where `r = σ_p²/σ_q²` and `d` is the dimension.

### The Identity

The KL divergence identity relates the global divergence to local misalignment:
```
KL(p_t|q_t) = ∫₀ᵗ E_{x~p_s}[(u(x,s)-v_θ(x,s))ᵀ(∇log p_s(x)-∇log q_s(x))] ds
```

This is useful because:
- **LHS**: Requires solving ODEs and computing densities
- **RHS**: Only requires evaluating velocities and scores at sampled points

### Part 2: The Bound

Part 2 validates a related bound on the KL divergence at terminal time t=1:
```
KL(p₁|q₁) ≤ ε√S
```

where:
- **ε** = √(E_{t~U[0,1], x~p_t} |v(x,t) - u(x,t)|²) is the RMS flow-matching error
- **S** = ∫₀¹ E_{x~p_t} |∇log p_t(x) - ∇log q_t(x)|² dt is the score-gap integral

This bound provides a certificate of model quality: if ε and S are small, then the KL divergence at t=1 is guaranteed to be small. This is particularly useful because ε can be computed from validation data without solving ODEs.

**Part 2 (Synthetic)**: Uses synthetic velocity fields v(x,t) = (a(t) + δ(t))x with perturbations δ(t) to systematically explore bound behavior across different model quality regimes.

**Part 2 (Learning)**: Trains velocity MLPs and verifies the bound holds (and tightens) as training progresses, demonstrating practical applicability to learned models.

## Troubleshooting

### Model not converging

1. **Increase training epochs**: Use `--epochs 500`
2. **Lower target MSE**: Use `--target_mse 0.2` first
3. **Check model capacity**: Larger hidden dims may help

### Memory issues

1. **Reduce batch size**: Use `--batch_size 64`
2. **Reduce evaluation samples**: Use `--num_samples 1000`
3. **Process in chunks**: Modify evaluation loop

### ODE solver failures

1. **Loosen tolerances**: Use `--rtol 1e-5 --atol 1e-7`
2. **Reduce time span**: Check if issue is near t=0 or t=1
3. **Increase max steps**: Modify in `eval.py`

## File Naming Convention

All output files use a consistent naming scheme:

```
{filename}_{schedule}_mse_{TARGET}_{YYYYMMDD_HHMMSS}.{ext}
```

Example:
```
kl_comparison_a2_mse_0-05_20251027_231335.png
```

Where:
- `filename`: Plot/data name
- `schedule`: a1, a2, or a3
- `TARGET`: Target MSE (replaced `.` with `-`)
- `TIMESTAMP`: When the experiment was run

## Dependencies

- `torch>=2.0.0`: Neural networks and autograd
- `torchdiffeq>=0.2.3`: ODE solving
- `numpy>=1.24.0`: Numerical computation
- `matplotlib>=3.7.0`: Plotting
- `scipy>=1.10.0`: Scientific computing
- `tqdm>=4.65.0`: Progress bars
- `seaborn>=0.12.0`: Statistical plots

<!-- ## License

[Specify your license here]

## Citation

If you use this code, please cite:
```bibtex
[Add citation information]
```

## Contact

[Add contact information if relevant] -->

