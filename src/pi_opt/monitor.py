"""
Training monitor for π-recursive optimizers.

Track optimizer behavior, learning rate evolution, and phase dynamics during training.
"""

from typing import Dict, List, Optional, Any
import json
from pathlib import Path


class PiOptimizerMonitor:
    """
    Monitor and log π-recursive optimizer behavior during training.

    Tracks learning rate, momentum, phase values, and gradients to help
    understand and debug the breathing patterns.

    Example:
        >>> monitor = PiOptimizerMonitor()
        >>> optimizer = PiAdam(model.parameters(), lr=1e-3)
        >>>
        >>> for step in range(num_steps):
        >>>     loss = compute_loss()
        >>>     optimizer.zero_grad()
        >>>     loss.backward()
        >>>
        >>>     # Log before step
        >>>     monitor.log_step(step, optimizer, loss.item())
        >>>     optimizer.step()
        >>>
        >>> monitor.save("training_log.json")
        >>> monitor.plot_summary()
    """

    def __init__(self):
        self.history: Dict[str, List[Any]] = {
            "step": [],
            "loss": [],
            "lr": [],
            "phase": [],
            "grad_norm": []
        }

    def log_step(
        self,
        step: int,
        optimizer,
        loss: float,
        grad_norm: Optional[float] = None
    ):
        """
        Log optimizer state at current training step.

        Args:
            step: Current training step
            optimizer: PiAdam or PiSGD optimizer instance
            loss: Current loss value
            grad_norm: Optional gradient norm (computed if None)
        """
        self.history["step"].append(step)
        self.history["loss"].append(loss)

        # Extract learning rate from optimizer
        if hasattr(optimizer, 'param_groups'):
            lr = optimizer.param_groups[0]['lr']
            self.history["lr"].append(lr)

        # Extract phase if π-recursive optimizer
        if hasattr(optimizer, '_phase') and hasattr(optimizer, '_t'):
            phase_val = optimizer._phase.phi(optimizer._t)
            self.history["phase"].append(phase_val)
        else:
            self.history["phase"].append(None)

        # Compute gradient norm if not provided
        if grad_norm is None and hasattr(optimizer, 'param_groups'):
            total_norm = 0.0
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

        self.history["grad_norm"].append(grad_norm)

    def save(self, filepath: str):
        """Save monitoring history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training log to {filepath}")

    def load(self, filepath: str):
        """Load monitoring history from JSON file."""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        print(f"Loaded training log from {filepath}")

    def plot_summary(self, figsize=(14, 10), save_path: Optional[str] = None):
        """
        Plot comprehensive summary of training dynamics.

        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib required for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        steps = self.history["step"]

        # Loss curve
        axes[0, 0].plot(steps, self.history["loss"], linewidth=2, color='#2E86AB')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].grid(alpha=0.3, linestyle='--')
        axes[0, 0].set_yscale('log')

        # Learning rate evolution
        if self.history["lr"]:
            axes[0, 1].plot(steps, self.history["lr"], linewidth=2, color='#A23B72')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('LR Schedule (π-breathing)', fontweight='bold')
            axes[0, 1].grid(alpha=0.3, linestyle='--')

        # Phase evolution
        if any(p is not None for p in self.history["phase"]):
            phase_vals = [p if p is not None else 0 for p in self.history["phase"]]
            axes[1, 0].plot(steps, phase_vals, linewidth=1.5, color='#F18F01', alpha=0.8)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Phase φ(t)')
            axes[1, 0].set_title('π-Recursive Phase', fontweight='bold')
            axes[1, 0].grid(alpha=0.3, linestyle='--')

        # Gradient norm
        if self.history["grad_norm"] and any(g is not None for g in self.history["grad_norm"]):
            grad_norms = [g if g is not None else 0 for g in self.history["grad_norm"]]
            axes[1, 1].plot(steps, grad_norms, linewidth=2, color='#6C757D', alpha=0.7)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_title('Gradient Magnitude', fontweight='bold')
            axes[1, 1].grid(alpha=0.3, linestyle='--')
            axes[1, 1].set_yscale('log')

        plt.suptitle('π-Recursive Optimizer Training Summary',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved summary plot to {save_path}")

        plt.show()

    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics from training run."""
        import numpy as np

        stats = {
            "total_steps": len(self.history["step"]),
            "final_loss": self.history["loss"][-1] if self.history["loss"] else None,
            "min_loss": min(self.history["loss"]) if self.history["loss"] else None,
        }

        if self.history["lr"]:
            stats["avg_lr"] = np.mean(self.history["lr"])
            stats["lr_std"] = np.std(self.history["lr"])

        if self.history["grad_norm"] and any(g is not None for g in self.history["grad_norm"]):
            valid_grads = [g for g in self.history["grad_norm"] if g is not None]
            stats["avg_grad_norm"] = np.mean(valid_grads)
            stats["max_grad_norm"] = max(valid_grads)

        return stats
