# 🌀 CHAOS MODE: The Gremlin's Guide

**Status:** Experimental (read: absolutely unhinged)
**Stability:** Questionable (that's the point)
**Recommended Use:** When you've tried everything else and need chaos

## Philosophy

Traditional optimization: "Carefully tune hyperparameters for stable convergence."

Chaos mode: "What if we just... YEET some parameters and see what happens?"

## Features

### 1. Quantum Tunneling 🌌
Randomly teleports the phase to escape local minima.

- Probability: 1-5%
- Scale: Configurable madness
- Theory: Local minima can't trap you if you teleport through them
- Reality: ¯\\\_(ツ)_/¯

```python
from pi_opt.chaos import ChaoticPiPhase

phase = ChaoticPiPhase(
    alpha=0.25,
    chaos_prob=0.02,     # 2% chance to YEET
    chaos_scale=15.0     # How far to yeet
)
```

### 2. Rage Quit Detection 💢
If stuck for N steps, randomly perturb parameters.

- Patience: Configurable (default: 100 steps)
- Scale: 0.1 = gentle nudge, 0.5 = full YEET
- Works surprisingly well on Rastrigin function
- Basically simulated annealing but angrier

```python
from pi_opt.chaos import RageQuitDetector

rage = RageQuitDetector(
    patience=100,        # How long before rage
    rage_scale=0.2,      # How hard to YEET
    verbose=True         # Print the anger
)

# In training loop
if rage.check(loss, parameters):
    print("Parameters have been YEETED")
```

### 3. Vibes-Based Tuning 🎵
Automatically adjust π-parameters based on loss trajectory.

- Loss going up? Breathe harder (increase amplitude)
- Loss going down? Chill out (decrease amplitude)
- Loss volatile? Explore more (increase drift)

```python
from pi_opt.chaos import VibesBasedTuner

vibes = VibesBasedTuner(adapt_rate=0.01)

# In training loop
adjustments = vibes.update(loss, optimizer)
# Optimizer π-parameters now updated based on ~vibes~
```

### 4. Adversarial Breathing 😈
Occasionally steps in the WRONG direction to force exploration.

- Probability: 5-10%
- Goes opposite to gradient
- Theoretically helps escape narrow valleys
- Practically: chaos

```python
from pi_opt.chaos import AdversarialBreathing

adversarial = AdversarialBreathing(
    adversarial_prob=0.05,
    adversarial_scale=0.1
)

# After computing gradients
adversarial.apply(parameters, gradients)
```

### 5. Cursed Loss Landscapes 😈
Make your loss landscape HARDER to force better exploration.

- Adds deterministic noise based on parameters
- Theory: Harder training = better generalization?
- Reality: Your loss curves will look WILD

```python
from pi_opt.chaos import CursedLossLandscape

curse = CursedLossLandscape(
    roughness=0.1,      # How cursed
    frequency=10.0      # How often cursed
)

loss = compute_loss()
cursed_loss = curse.curse(loss, parameters)
```

### 6. Full Gremlin Mode 🔥
ALL THE CHAOS. ALL OF IT.

```python
from pi_opt.chaos import add_chaos_to_optimizer
from pi_opt.optim import PiAdam

opt = PiAdam(model.parameters(), lr=1e-3)

# INJECT MAXIMUM CHAOS
chaos = add_chaos_to_optimizer(opt, chaos_mode="full_gremlin")

# Now your optimizer has:
# - Quantum tunneling
# - Rage quit detection
# - Vibes-based tuning
# - Adversarial breathing
```

## When to Use Chaos Mode

**Good use cases:**
- Stuck in local minimum
- Loss plateaued for 1000+ steps
- You've tried everything else
- You want entertaining loss curves
- Multimodal optimization (Rastrigin, Ackley, etc.)
- You're feeling adventurous

**Bad use cases:**
- Production systems
- When you need reproducibility
- Convex optimization (just use Adam lol)
- Your job depends on it

## Performance

Does it work? Sometimes!

**Benchmarks:**
- Rastrigin 10D: ~30% of runs beat vanilla Adam
- Ackley 5D: Surprisingly good (quantum tunneling helps)
- Simple quadratics: Please just use Adam
- Deep learning: Results vary wildly

## FAQ

**Q: Is this scientifically sound?**
A: Probably not, but neither is SGD if you think about it

**Q: Should I use this in production?**
A: lmao no

**Q: Does it actually help?**
A: On multimodal problems, sometimes! On everything else, ¯\\\_(ツ)_/¯

**Q: Is this just randomized search with extra steps?**
A: Listen here you little shit—

**Q: Why did you make this?**
A: Science isn't about WHY, it's about WHY NOT

## License

MIT (Madness Included, Totally)

## Citation

If this somehow works for you:

```bibtex
@misc{gremlin_mode,
  title={Chaos Mode: When Bugs Become Features},
  author={The Gremlins},
  year={2024},
  note={We're sorry}
}
```

---

**Remember:** Bugs are features. Exploits are gameplay. Chaos is a ladder.

🌀🔥💀
