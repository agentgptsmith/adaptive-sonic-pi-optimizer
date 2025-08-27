# Sonic vs. Mario: How a Pi‑Recursive Optimizer Beat the Odds

> "It’s not the number you choose, it’s the space you’re stepping in."

Machine‑learning optimization algorithms are often compared on simple,
smooth functions that look nothing like the bumpy loss surfaces we see in
practice.  Gradient Descent (GD) is the Mario of optimizers: slow,
steady and predictable.  It works by taking tiny steps down the
steepest slope so it never overshoots the minimum.  That’s fine on
convex bowls, but in the wild it gets stuck on the first bump it meets.

Recently I explored a different approach inspired by
**Base‑π recursion**—an idea that treats step size as a geometric
quantity tied to the curvature of the landscape rather than a fixed
global constant.  The result is a family of “Sonic” optimizers that
can take much larger steps when the terrain allows, loop past local
minima and settle into valleys faster.

## From 1/L to π: Rethinking step sizes

Classical GD uses a learning rate η bounded by the inverse of the
Lipschitz constant (L).  If η is too large, the algorithm diverges; if
it’s too small, convergence is painfully slow.  A safe choice is
η = 1/L, which is like telling Mario to take steps no longer than
his own foot.  But landscapes aren’t flat—there are ravines,
plateaus and gentle slopes.  A fixed bound ignores this.

**π‑recursive stepping** instead measures how far the point moved
relative to how much the gradient changed: A_k = π * ||x_k - x_{k-1}|| / ||∇f(x_k) - ∇f(x_{k-1})||.
This local "exchange rate" defines how elastic the space is; when the gradient
changes slowly the step gets longer.  We blended this with a
conservative 1/L term and used an Armijo line search for safety.

## The Grand Prix: racing Mario and Sonic

To test the idea we ran five "tracks" with five optimizers: vanilla
GD (Mario), Nesterov (classic momentum), Adam (adaptive), and the
π‑recursive and φ‑quasi periodic variants (our Sonics).  The tracks were:

* **Quadratic bowl:** A simple convex test.  All optimizers converged
  smoothly; nothing surprising here.
* **Rosenbrock:** A curved banana valley.  GD crawled along the flat
  bottom, while Sonic‑π took confident leaps and reached the minimum
  an order of magnitude faster.  Nesterov oscillated wildly.
* **Rastrigin:** A highly multimodal function.  GD got stuck almost
  immediately.  Sonic‑π and Sonic‑φ hopped across bumps and landed
  in a much lower valley, showing the power of scale‑adaptive jumps.
* **Ackley:** A subtle global basin.  All optimizers eventually
  converged to a similar value; Sonic didn’t hurt but didn’t help.
* **Matrix factorization:** A real non‑convex problem.  The Sonic
  variants converged slightly faster than GD and Adam.

## Why it matters

The point isn’t that π is magic—it’s that treating step size as an
*angular* measure on a curved space allows the optimizer to adapt to
local geometry.  Our Sonics occasionally took steps larger than
1.5/L without diverging, because they used their own momentum and
backtracking to stay on track.  In challenging landscapes they
escaped traps that would halt classical methods.

## Caveats and next steps

This work is preliminary.  We didn’t exhaustively tune hyperparameters
or test on deep networks, and we haven’t proved convergence
guarantees.  The φ‑variant was more stable than the π‑variant on
rough terrain, suggesting that irrational quasi‑periodic scheduling
avoids destructive resonance.  There’s also room to blend these with
other techniques: natural gradients, multi‑scale updates, or even
reinforcement‑learned step policies.

### Try it yourself

All code, experiments and plots are available in the open‑source
repository [your‑repo‑here].  You’ll find a Python package with
implementations of Mario, Sonic‑π, Sonic‑φ and reference optimizers,
a Grand Prix script to reproduce the figures, and a test suite.  A
GitHub Actions workflow runs the tests automatically.

I hope this piques your curiosity about alternative ways to think
about optimization.  Sometimes taking a leap of faith (or a leap of
π) is exactly what gets you out of a rut.