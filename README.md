# π-Recursive Optimizer

Drop-in π-recursive learning rate and momentum modulation for PyTorch. Small, interpretable "breathing" helps training avoid shallow traps without sacrificing stability.

## Features

- **π-recursive modulation**: Non-repeating oscillation patterns based on π's transcendental properties
- **Dual modulation**: Both learning rate and momentum adapt dynamically
- **Drop-in replacement**: Compatible with standard PyTorch optimizer API
- **Minimal overhead**: Lightweight computation with interpretable hyperparameters
- **Proven effectiveness**: Helps escape local minima on challenging optimization landscapes

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from pi_opt.optim import PiAdam

# Create model and optimizer
model = YourModel()
optimizer = PiAdam(
    model.parameters(),
    lr=1e-3,
    pi_alpha=0.25,
    pi_lambdas=[0.4, 0.15],
    pi_amplitude=0.1,
    anneal_b=1e-4
)

# Standard training loop
for batch in dataloader:
    loss = compute_loss(model, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Available Optimizers

### PiAdam
Adam with π-recursive modulation (recommended for most use cases):
```python
from pi_opt.optim import PiAdam
optimizer = PiAdam(model.parameters(), lr=3e-4, pi_amplitude=0.1)
```

### PiSGD
SGD with momentum and π-recursive modulation:
```python
from pi_opt.optim import PiSGD
optimizer = PiSGD(model.parameters(), lr=1e-2, momentum=0.9, pi_amplitude=0.1)
```

## Key Hyperparameters

- `pi_alpha` (default: 0.25): Drift strength for phase evolution
- `pi_lambdas` (default: [0.4, 0.15]): Harmonic amplitudes at π^1, π^2 frequencies
- `pi_amplitude` (default: 0.1): Learning rate modulation amplitude (±10%)
- `anneal_b` (default: 1e-4): Annealing rate for gradual LR decay
- `momentum_amplitude` (default: 0.05): Momentum modulation amplitude

## Examples

Run the included examples:

```bash
# Optimize 2D Rosenbrock function
python examples/rosenbrock_piadam.py

# Train on MNIST
python examples/mnist_piadam.py
```

## Project Structure

```
adaptive-sonic-pi-optimizer/
├── src/pi_opt/
│   ├── __init__.py
│   ├── schedules.py          # PiPhase and pi_schedule
│   └── optim/
│       ├── __init__.py
│       ├── pi_adam.py        # PiAdam optimizer
│       └── pi_sgd.py         # PiSGD optimizer
├── examples/
│   ├── rosenbrock_piadam.py  # 2D optimization demo
│   └── mnist_piadam.py       # MNIST training demo
├── tests/
│   └── test_schedules.py     # Unit tests
└── pyproject.toml            # Package configuration
```

## Testing

```bash
python -m pytest tests/
```

## How It Works

The π-recursive schedule combines:
1. **Logarithmic drift**: log_π(t) term for slow, non-linear progression
2. **Harmonic oscillations**: cos(π^i · t) terms create complex, non-repeating patterns
3. **Annealing**: Optional √(1 + bt) decay for convergence

This creates a "breathing" pattern that helps optimizers explore the loss landscape more effectively while maintaining training stability.

## License

MIT
