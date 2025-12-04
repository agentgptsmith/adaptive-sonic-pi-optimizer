"""
Benchmark PiAdam vs vanilla Adam on the Rastrigin function.

Rastrigin is a highly multimodal function that's notoriously difficult to optimize
due to its many local minima. Perfect for testing π-recursive breathing!
"""

import torch
import math
import time
from pi_opt.optim import PiAdam


def rastrigin(x: torch.Tensor, A: float = 10.0) -> torch.Tensor:
    """
    N-dimensional Rastrigin function: f(x) = A*n + Σ[x_i² - A*cos(2π*x_i)]
    Global minimum: f(0, 0, ..., 0) = 0
    """
    n = x.numel()
    return A * n + torch.sum(x**2 - A * torch.cos(2 * math.pi * x))


def optimize_rastrigin(optimizer_class, dim=10, steps=2000, **opt_kwargs):
    """Run optimization and return trajectory."""
    x = torch.randn(dim, requires_grad=True) * 5.0  # Random start
    optimizer = optimizer_class([x], **opt_kwargs)

    losses = []
    start_time = time.time()

    for step in range(steps):
        loss = rastrigin(x)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1:4d} | Loss: {loss.item():.6f}")

    elapsed = time.time() - start_time
    final_loss = losses[-1]

    return {
        "losses": losses,
        "final_loss": final_loss,
        "final_x": x.detach().clone(),
        "time": elapsed
    }


def main():
    print("=" * 60)
    print("BENCHMARK: Rastrigin Function (10D)")
    print("=" * 60)

    lr = 1e-2
    dim = 10
    steps = 2000

    # Vanilla Adam
    print("\n[1] Vanilla Adam (lr={})".format(lr))
    adam_result = optimize_rastrigin(
        torch.optim.Adam,
        dim=dim,
        steps=steps,
        lr=lr
    )

    # PiAdam with moderate breathing
    print("\n[2] PiAdam (lr={}, pi_amplitude=0.15)".format(lr))
    piadam_result = optimize_rastrigin(
        PiAdam,
        dim=dim,
        steps=steps,
        lr=lr,
        pi_alpha=0.3,
        pi_lambdas=[0.4, 0.15],
        pi_amplitude=0.15,
        anneal_b=1e-4
    )

    # Results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Vanilla Adam:")
    print(f"  Final Loss: {adam_result['final_loss']:.6f}")
    print(f"  Time: {adam_result['time']:.2f}s")
    print(f"\nPiAdam:")
    print(f"  Final Loss: {piadam_result['final_loss']:.6f}")
    print(f"  Time: {piadam_result['time']:.2f}s")

    improvement = ((adam_result['final_loss'] - piadam_result['final_loss']) /
                   adam_result['final_loss'] * 100)
    print(f"\nImprovement: {improvement:+.1f}%")

    # Plot convergence
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(adam_result['losses'], label='Vanilla Adam', linewidth=2, alpha=0.8)
        ax.plot(piadam_result['losses'], label='PiAdam (π-breathing)', linewidth=2, alpha=0.8)
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss (Rastrigin)', fontsize=12)
        ax.set_title('Optimization Convergence: Rastrigin Function (10D)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('benchmark_rastrigin_convergence.png', dpi=150)
        print("\nSaved convergence plot to benchmark_rastrigin_convergence.png")
        plt.show()
    except ImportError:
        print("\n(matplotlib not available - skipping plot)")


if __name__ == "__main__":
    main()
