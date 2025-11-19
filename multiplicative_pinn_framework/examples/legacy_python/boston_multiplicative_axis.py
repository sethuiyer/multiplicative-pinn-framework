# boston_multiplicative_axis_full.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# 1. Synthetic Boston Housing (realistic stats + poisoned inversions)
# -------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

n_samples = 506

features = {
    'CRIM': np.random.lognormal(np.log(3.61), 1, n_samples).clip(0.006, 88.98),
    'ZN': np.random.normal(11.36, 23.32, n_samples).clip(0, 100),
    'INDUS': np.random.normal(11.14, 6.86, n_samples).clip(0.46, 27.74),
    'CHAS': np.random.binomial(1, 0.07, n_samples),
    'NOX': np.random.normal(0.55, 0.12, n_samples).clip(0.385, 0.871),
    'RM': np.random.normal(6.28, 0.70, n_samples).clip(3.56, 8.78),
    'AGE': np.random.normal(68.57, 28.15, n_samples).clip(2.9, 100),
    'DIS': np.random.normal(3.79, 2.11, n_samples).clip(1.13, 12.13),
    'RAD': np.random.poisson(9.55, n_samples).clip(1, 24),
    'TAX': np.random.normal(408, 168, n_samples).clip(187, 711),
    'PTRATIO': np.random.normal(18.46, 2.16, n_samples).clip(12.6, 22),
    'B': np.random.normal(356.67, 91.29, n_samples).clip(0.32, 396.9),
    'LSTAT': np.random.normal(12.65, 7.14, n_samples).clip(1.73, 37.97),
}

X_np = np.stack(list(features.values()), axis=1)

# Price base: increases with RM, decreases with LSTAT, etc.
MEDV_base = (4.0 * features['RM'] 
             - 0.5 * features['LSTAT'] 
             + 0.3 * features['DIS']
             + 0.1 * np.random.randn(n_samples))

noise = np.random.normal(0, 4.5, n_samples)
inversion_mask = np.random.rand(n_samples) < 0.15
noise[inversion_mask] -= 12.0  # poison some monotonicity

MEDV = np.clip(MEDV_base + noise, 5, 50)
y_np = MEDV.reshape(-1, 1)

# Normalize features
scaler = StandardScaler()
X_np = scaler.fit_transform(X_np)

X = torch.from_numpy(X_np).float()
y = torch.from_numpy(y_np).float()

# Train/test split
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# -------------------------------------------------
# 2. Model
# -------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# 3. Constraint utilities
# -------------------------------------------------
def monotonic_violation(preds, rm_values):
    """Mean violation of non-decreasing prices w.r.t. RM"""
    sorted_idx = torch.argsort(rm_values.squeeze())
    sorted_preds = preds[sorted_idx]
    diffs = sorted_preds[1:] - sorted_preds[:-1]
    return torch.relu(-diffs).mean()

def exp_barrier(v, gamma=5.0):
    return torch.exp(gamma * v)

def euler_gate(v, primes=[2, 3, 5, 7, 11], tau=3.0):
    """From your paper — multiplicative attenuation"""
    out = torch.ones_like(v)
    for p in primes:
        out = out * (1 - p ** (-tau * v))
    return torch.clamp(out, min=1e-8, max=1.0)

# -------------------------------------------------
# 4. Training function
# -------------------------------------------------
def train_and_eval(name, use_barrier=False, use_gate=False, gamma=5.0, tau=3.0):
    print(f"\n=== {name} ===")
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    mse_loss = nn.MSELoss()

    for epoch in range(201):
        optimizer.zero_grad()
        preds = model(X_train)
        fidelity = mse_loss(preds, y_train)
        rm = X_train[:, 5]  # RM is index 5
        viol = monotonic_violation(preds, rm)

        if use_barrier:
            factor = exp_barrier(viol, gamma=gamma)
        elif use_gate:
            factor = euler_gate(viol, tau=tau)
        else:
            factor = torch.tensor(1.0)

        loss = fidelity * factor
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Fid {fidelity.item():6.4f} | Viol {viol.item():.6f} | Factor {factor.item():.4f}")

    # Evaluation
    with torch.no_grad():
        preds_test = model(X_test)
        test_mse = mse_loss(preds_test, y_test).item()
        rm_test = X_test[:, 5]
        test_viol = monotonic_violation(preds_test, rm_test).item()
        compliance = 100.0 if test_viol < 1e-6 else (1 - test_viol / preds_test.std().item()) * 100

        print(f"\n{name} FINAL RESULTS:")
        print(f"Test MSE       : {test_mse:.4f}")
        print(f"Test Violation : {test_viol:.6f}")
        print(f"Compliance     : {compliance:.2f}%")
        print(f"Min/Max Pred   : {preds_test.min().item():.2f} / {preds_test.max().item():.2f}")

# -------------------------------------------------
# 5. RUN ALL THREE
# -------------------------------------------------
train_and_eval("1. Unconstrained Baseline", use_barrier=False, use_gate=False)
train_and_eval("2. Exponential Barrier (γ=5)", use_barrier=True, gamma=5.0)
train_and_eval("3. Euler Gate Attenuator (τ=3)", use_gate=True, tau=3.0)
