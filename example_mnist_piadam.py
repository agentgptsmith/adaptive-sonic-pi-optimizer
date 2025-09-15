
# Train MNIST with PiAdam (requires torchvision).
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pi_opt.optim import PiAdam

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def main():
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test,  batch_size=256)
    device = torch.device("cpu")
    model = Net().to(device)
    opt = PiAdam(model.parameters(), lr=1e-3, pi_alpha=0.3, pi_lambdas=[0.25, 0.1], pi_amplitude=0.1)
    for epoch in range(1, 4):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb); loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred==yb).sum().item(); total += yb.size(0)
        print(f"epoch {epoch}: acc={correct/total:.4f}")

if __name__ == "__main__":
    main()
