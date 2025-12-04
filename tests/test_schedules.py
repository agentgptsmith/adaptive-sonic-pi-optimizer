
import math
from pi_opt.schedules import PiPhase, pi_schedule

def test_phi_varies_and_finite():
    p = PiPhase(alpha=0.25, beta=1.0, lambdas=[0.2, 0.05])
    vals = [p.phi(t) for t in range(1, 100)]
    assert all(math.isfinite(v) for v in vals)
    assert any(abs(vals[i+1]-vals[i]) > 0 for i in range(len(vals)-1))

def test_schedule_bounds():
    base = 1e-3
    vals = [pi_schedule(t, base, amplitude=0.2, anneal_b=1e-3) for t in range(500)]
    assert all(v > 0 for v in vals)
    assert min(vals) < max(vals)
