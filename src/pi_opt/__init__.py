
from .schedules import PiPhase, pi_schedule
from .optim.pi_adam import PiAdam
from .optim.pi_sgd import PiSGD

__version__ = "0.2.0"
__all__ = ["PiPhase", "pi_schedule", "PiAdam", "PiSGD"]

# Visualization is optional (requires matplotlib)
try:
    from .visualize import plot_phase_evolution, plot_lr_schedule, plot_optimizer_trajectory_2d
    __all__.extend(["plot_phase_evolution", "plot_lr_schedule", "plot_optimizer_trajectory_2d"])
except ImportError:
    pass
