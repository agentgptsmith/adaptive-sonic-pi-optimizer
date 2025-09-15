
# π-Recursive Optimizer (pi-recursive-optimizer)

Drop-in π-recursive learning-rate/momentum modulation for PyTorch. Small, interpretable “breathing” helps training avoid shallow traps without blowing up stability.

## Quick Start
```bash
pip install -e .
python examples/rosenbrock_piadam.py
```

## Minimal Usage
```python
from pi_opt.optim import PiAdam
opt = PiAdam(model.parameters(), lr=1e-3, pi_alpha=0.25, pi_lambdas=[0.4, 0.15], pi_amplitude=0.1, anneal_b=1e-4)
```

## Files
- src/pi_opt/schedules.py — PiPhase + pi_schedule
- src/pi_opt/optim/pi_adam.py — AdamW-style with π modulation
- src/pi_opt/optim/pi_sgd.py  — SGD+momentum with π modulation
- examples/rosenbrock_piadam.py, examples/mnist_piadam.py
- tests/test_schedules.py

## License
MIT


---

## 📦 Note on Flat Upload Version
This is a **flat layout** for easy GitHub web uploads (e.g., from a tablet).
Files are prefixed to indicate original folders:

- `pi_opt_*.py` → from `src/pi_opt/`
- `pi_opt_optim_*.py` → from `src/pi_opt/optim/`
- `example_*.py` → from `examples/`
- `test_*.py` → from `tests/`
- `GHWORKFLOW_*.yml` → from `.github/workflows/`

## 🔗 Structured Version
For proper development or `pip install -e .`, use the structured repo layout:
https://github.com/<your-username>/pi-recursive-optimizer
