# Flow Matching KL Divergence Bounds – Experiments

This repository hosts the implementation of the numerical studies reported in “On Flow Matching KL Divergence,” Su et al., 2025. The code reproduces the empirical evidence supporting the paper’s KL error bounds.

For reference, the paper builds on the KL evolution identity:
$$
\mathrm{KL}(p_t \mid q_t) = \int_0^t \mathbb{E}_{x \sim p_s} \left[ \big(u(x,s) - v_{\theta}(x,s)\big)^{\top} \big(\nabla \log p_s(x) - \nabla \log q_s(x)\big) \right] ds
$$

where:
- $p_t$ evolves under velocity field $u(x,t) = a(t) \, x$
- $q_t$ evolves under learned velocity field $v_{\theta}(x,t)$
- Both start as standard Gaussians: $p_0 = q_0 = \mathcal{N}(0, I)$

## Project Structure

```
flow_kl/
├── README.md
├── requirements.txt
├── core/                        # Shared schedules, densities, utilities
│   ├── __init__.py
│   ├── true_path.py             # Schedules, sampling, densities, scores
│   └── utils.py                 # Seeding, device helpers, plotting I/O
├── part1/                       # Part 1: learned velocity identity experiments
│   ├── __init__.py
│   ├── experiment.py            # CLI: python -m part1.experiment
│   ├── eval.py                  # LHS / RHS evaluation routines
│   ├── model.py                 # Velocity MLP
│   └── train.py                 # Training loop and plotting helpers
├── part2/
│   ├── __init__.py
│   ├── synthetic/               # Part 2A: synthetic perturbation studies
│   │   ├── __init__.py
│   │   ├── experiment.py        # CLI: python -m part2.synthetic.experiment
│   │   ├── eval.py              # Part 2 evaluation helpers
│   │   ├── synthetic_velocity.py
│   │   └── run_all_experiments.py
│   └── learned/                 # Part 2B: learned perturbation studies
│       ├── __init__.py
│       ├── experiment.py        # CLI: python -m part2.learned.experiment
│       ├── eval.py
│       ├── model.py
│       └── train.py
├── plotting/                    # Plot regeneration & epsilon-curve utilities
│   ├── __init__.py
│   ├── plot_eps_curves.py
│   ├── regenerate_plots.py
│   └── regenerate_plots_from_csv.py
├── scripts/                     # Automation & shell entry points
│   ├── run_all_experiments.py / .ps1
│   ├── run_all_cross_eval.sh / .ps1
│   ├── run_all_nolearning.sh / .ps1
│   ├── run_all_pt2_experiments.ps1
│   ├── run_all_pt2_learn_experiments.ps1
│   └── nolearning_test.py
├── tests/                       # Unit / integration tests
│   ├── __init__.py
│   ├── test_golden_path.py
│   ├── test_rhs.py
│   ├── test_pt2.py
│   ├── test_learn_pt2.py
│   └── test_eps_curves.py
└── data/                        # Generated checkpoints, plots, metrics
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
python scripts/nolearning_test.py --schedule_p a1 --schedule_q a2 --skip_ode
```

For all 6 schedule permutations:
```bash
bash scripts/run_all_nolearning.sh      # or: pwsh scripts/run_all_nolearning.ps1
```

### 2. Train and evaluate a model

Train a model to learn velocity field $v_{\theta}$:

```bash
python -m part1.experiment --schedule a1 --target_mse 0.05
```

This will reproduce the Section 5.2 checkpoints:
- Train a neural network to match the true velocity
- Write checkpoints, metrics, and plots into the configured output directory 
- Evaluate the KL identity
- Generate plots showing LHS vs RHS

### 3. Load a trained model and re-evaluate

```bash
python -m part1.experiment --schedule a1 --load_model path/to/vtheta_schedule_a1_mse_0-05_TIMESTAMP.pth
```

### 4. Part 2: Synthetic Bound Verification

Validate the bound $\mathrm{KL}(p_1 \mid q_1) \le \epsilon \sqrt{S}$ using synthetic velocity fields:

```bash
python -m part2.synthetic.experiment --schedule a1 --delta_beta 0.0 0.05 0.1 0.2
```Run all Part 2 experiments:
```bash
python -m part2.synthetic.run_all_experiments
```

### 5. Part 2 (Learning): Learned Bound Verification

Train a velocity MLP and verify the bound across training checkpoints:

```bash
python -m part2.learned.experiment --schedule a1 --epochs 400 --eval_checkpoints "all"
```

This will:
- Train a neural network for up to 400 epochs
- Save multiple checkpoints (best, final, and on improvement)
- Evaluate the bound for all saved checkpoints
- Generate scatter plots showing bound tightening with training

## Dependencies

- `torch>=2.0.0`: Neural networks and autograd
- `torchdiffeq>=0.2.3`: ODE solving
- `numpy>=1.24.0`: Numerical computation
- `matplotlib>=3.7.0`: Plotting
- `scipy>=1.10.0`: Scientific computing
- `tqdm>=4.65.0`: Progress bars
- `seaborn>=0.12.0`: Statistical plots

## License

This project is distributed under the [MIT License](LICENSE).

<!-- ## Citation

If you use this code, please cite:
```bibtex
[TBD citation information]
```





