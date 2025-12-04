"""
Advanced example: Ackley function optimization with trajectory visualization.

The Ackley function has a nearly-flat outer region and a large hole at the center.
It's excellent for testing whether π-breathing helps escape plateaus.
"""

import torch
import math
from pi_opt.optim import PiAdam


def ackley(x: torch.Tensor, a: float = 20, b: float = 0.2, c: float = 2*math.pi) -> torch.Tensor:
    """
    Ackley function: challenging multimodal benchmark.
    Global minimum: f(0, ..., 0) = 0
    """
    d = x.numel()
    sum1 = -a * torch.exp(-b * torch.sqrt(torch.mean(x**2)))
    sum2 = -torch.exp(torch.mean(torch.cos(c * x)))
    return sum1 + sum2 + a + math.e


def optimize_with_tracking(optimizer, x, steps=3000):
    """Optimize and track full trajectory."""
    trajectory = []
    losses = []

    for step in range(steps):
        trajectory.append(x.detach().clone())

        loss = ackley(x)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"Step {step+1:4d} | Loss: {loss.item():.6f} | "
                  f"x: [{x[0].item():.3f}, {x[1].item():.3f}]")

    return trajectory, losses


def main():
    print("=" * 60)
    print("ADVANCED EXAMPLE: Ackley Function (2D)")
    print("π-recursive breathing helps escape the plateau!")
    print("=" * 60)

    # Start from a challenging position
    x = torch.tensor([3.5, 3.5], requires_grad=True)

    optimizer = PiAdam(
        [x],
        lr=5e-2,
        pi_alpha=0.35,
        pi_lambdas=[0.5, 0.2, 0.05],  # 3 harmonics for extra breathing
        pi_amplitude=0.2,
        anneal_b=5e-5
    )

    print("\nOptimizing from initial position: [{:.2f}, {:.2f}]".format(x[0].item(), x[1].item()))
    print()

    trajectory, losses = optimize_with_tracking(optimizer, x, steps=3000)

    print(f"\nFinal position: [{x[0].item():.6f}, {x[1].item():.6f}]")
    print(f"Final loss: {losses[-1]:.6f}")

    # Visualize if matplotlib available
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from pi_opt import plot_optimizer_trajectory_2d

        # Plot trajectory on Ackley surface
        trajectory_2d = [(t[0].item(), t[1].item()) for t in trajectory]

        fig = plot_optimizer_trajectory_2d(
            trajectory_2d,
            loss_function=lambda x, y: ackley(torch.tensor([x, y])).item(),
            title="PiAdam Trajectory on Ackley Function"
        )
        plt.savefig('ackley_trajectory.png', dpi=150)
        print("\nSaved trajectory plot to ackley_trajectory.png")

        # Plot convergence
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(losses, linewidth=2, color='#2E86AB')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss (Ackley)', fontsize=12)
        ax.set_title('Convergence: PiAdam on Ackley Function', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('ackley_convergence.png', dpi=150)
        print("Saved convergence plot to ackley_convergence.png")

        plt.show()

    except ImportError:
        print("\n(matplotlib not available - skipping visualization)")


if __name__ == "__main__":
    main()
