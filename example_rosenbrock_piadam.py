
# Optimize the 2D Rosenbrock ("banana") function using PiAdam.
import torch
from pi_opt.optim import PiAdam

def rosenbrock(x, y, a=1.0, b=100.0):
    return (a - x)**2 + b*(y - x**2)**2

def main():
    xy = torch.randn(2, requires_grad=True)
    opt = PiAdam([xy], lr=3e-3, pi_alpha=0.35, pi_lambdas=[0.3, 0.1], pi_amplitude=0.12)
    for t in range(3000):
        x, y = xy[0], xy[1]
        loss = rosenbrock(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if (t+1) % 500 == 0:
            print(f"step {t+1:4d} | loss={loss.item():.6f} | x={x.item():.4f} y={y.item():.4f}")
    x, y = xy.detach().tolist()
    print(f"Converged near (1,1): x={x:.4f}, y={y:.4f}")

if __name__ == '__main__':
    main()
