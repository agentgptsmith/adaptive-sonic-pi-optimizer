"""
Visualization tools for π-recursive optimizer behavior.

Plot phase evolution, learning rate schedules, and optimizer dynamics to understand
and debug the breathing patterns.
"""

import math
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from .schedules import PiPhase, pi_schedule


def plot_phase_evolution(
    alpha: float = 0.25,
    beta: float = 1.0,
    lambdas: Optional[List[float]] = None,
    steps: int = 1000,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot the π-recursive phase evolution over time.

    Args:
        alpha: Phase drift coefficient
        beta: Phase time offset
        lambdas: Harmonic amplitudes
        steps: Number of timesteps to plot
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    phase = PiPhase(alpha=alpha, beta=beta, lambdas=lambdas)
    t_vals = np.arange(steps)
    phi_vals = [phase.phi(t) for t in t_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Phase evolution
    ax1.plot(t_vals, phi_vals, linewidth=1.5, color='#2E86AB')
    ax1.set_xlabel('Training Step', fontsize=11)
    ax1.set_ylabel('Phase φ(t) [radians]', fontsize=11)
    ax1.set_title('π-Recursive Phase Evolution', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')

    # Phase modulation (cos and sin carriers)
    cos_vals = [math.cos(phi) for phi in phi_vals]
    sin_vals = [math.sin(phi) for phi in phi_vals]

    ax2.plot(t_vals, cos_vals, label='cos(φ)', alpha=0.8, color='#A23B72')
    ax2.plot(t_vals, sin_vals, label='sin(φ)', alpha=0.8, color='#F18F01')
    ax2.set_xlabel('Training Step', fontsize=11)
    ax2.set_ylabel('Carrier Value', fontsize=11)
    ax2.set_title('Phase Carriers (Modulation Patterns)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_ylim(-1.2, 1.2)

    plt.tight_layout()
    return fig


def plot_lr_schedule(
    base_lr: float = 1e-3,
    pi_amplitude: float = 0.1,
    anneal_b: float = 1e-4,
    alpha: float = 0.25,
    lambdas: Optional[List[float]] = None,
    steps: int = 1000,
    compare_vanilla: bool = True,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot learning rate schedule with and without π-modulation.

    Args:
        base_lr: Base learning rate
        pi_amplitude: Modulation amplitude
        anneal_b: Annealing coefficient
        alpha: Phase drift coefficient
        lambdas: Harmonic amplitudes
        steps: Number of timesteps to plot
        compare_vanilla: Include vanilla (non-modulated) schedule for comparison
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    phase = PiPhase(alpha=alpha, beta=1.0, lambdas=lambdas)
    t_vals = np.arange(steps)

    # π-modulated schedule
    pi_lr_vals = [pi_schedule(t, base_lr, pi_amplitude, phase, anneal_b) for t in t_vals]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(t_vals, pi_lr_vals, linewidth=2, color='#2E86AB',
            label=f'π-recursive (amp={pi_amplitude})', alpha=0.9)

    if compare_vanilla:
        # Vanilla schedule (no modulation, just annealing)
        vanilla_lr = [base_lr / math.sqrt(1 + anneal_b * t) if anneal_b > 0 else base_lr
                      for t in t_vals]
        ax.plot(t_vals, vanilla_lr, linewidth=2, color='#6C757D',
                label='Vanilla (no modulation)', linestyle='--', alpha=0.7)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule: π-Recursive Breathing',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')

    # Add breathing range annotation
    if pi_amplitude > 0:
        avg_lr = np.mean(pi_lr_vals[:100])  # Average of first 100 steps
        ax.axhline(avg_lr, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.fill_between([0, steps],
                        avg_lr * (1 - pi_amplitude),
                        avg_lr * (1 + pi_amplitude),
                        alpha=0.15, color='#2E86AB',
                        label=f'Breathing range (±{pi_amplitude*100:.0f}%)')

    plt.tight_layout()
    return fig


def plot_optimizer_trajectory_2d(
    optimizer_states: List[Tuple[float, float]],
    loss_function=None,
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Optimizer Trajectory"
) -> plt.Figure:
    """
    Plot 2D optimization trajectory (for visualizing optimizer path).

    Args:
        optimizer_states: List of (x, y) positions during optimization
        loss_function: Optional function(x, y) -> loss for contour plot
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    xs, ys = zip(*optimizer_states)

    # Plot contours if loss function provided
    if loss_function is not None:
        x_min, x_max = min(xs) - 0.5, max(xs) + 0.5
        y_min, y_max = min(ys) - 0.5, max(ys) + 0.5

        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.array([[loss_function(x, y) for x in x_grid] for y in y_grid])

        contours = ax.contour(X, Y, Z, levels=20, alpha=0.4, colors='gray')
        ax.clabel(contours, inline=True, fontsize=8)

    # Plot trajectory
    ax.plot(xs, ys, 'o-', linewidth=2, markersize=4, color='#2E86AB',
            alpha=0.7, label='Optimization path')
    ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start', zorder=5)
    ax.plot(xs[-1], ys[-1], 'r*', markersize=15, label='End', zorder=5)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def save_all_plots(output_dir: str = "plots", **kwargs):
    """
    Generate and save all visualization plots.

    Args:
        output_dir: Directory to save plots
        **kwargs: Additional arguments passed to plotting functions
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Phase evolution
    fig1 = plot_phase_evolution(**kwargs)
    fig1.savefig(f"{output_dir}/phase_evolution.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # LR schedule
    fig2 = plot_lr_schedule(**kwargs)
    fig2.savefig(f"{output_dir}/lr_schedule.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print(f"Saved plots to {output_dir}/")


if __name__ == "__main__":
    # Demo: generate example plots
    print("Generating π-recursive visualization plots...")

    # Phase evolution
    fig1 = plot_phase_evolution(alpha=0.25, lambdas=[0.4, 0.15], steps=1000)
    plt.show()

    # Learning rate schedule
    fig2 = plot_lr_schedule(base_lr=1e-3, pi_amplitude=0.15, steps=1000)
    plt.show()
