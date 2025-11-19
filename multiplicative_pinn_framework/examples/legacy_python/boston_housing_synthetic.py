import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

# Synthetic Boston data gen (mimicking real stats)
np.random.seed(42)
torch.manual_seed(42)

n_samples = 506

# Features with real means/stds (approx from dataset)
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

X = np.stack(list(features.values()), axis=1)

# MEDV increasing with RM, but noisy + some inversions
MEDV_base = 4 * features['RM'] - 0.5 * features['LSTAT'] + 0.3 * features['DIS']
noise = np.random.normal(0, 5, n_samples)
inversion_mask = np.random.rand(n_samples) < 0.15
noise[inversion_mask] -= 10  # force some monotonic violations in data
MEDV = (MEDV_base + noise).clip(5, 50)

y = MEDV.reshape(-1, 1)

# Normalize X
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# Train/test split
n_train = 400
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Model
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

def monotonic_violation(preds, rm_values):
    sorted_idx = torch.argsort(rm_values.squeeze())
    sorted_preds = preds[sorted_idx]
    diffs = sorted_preds[1:] - sorted_preds[:-1]
    return torch.relu(-diffs).mean()  # positive if decreasing

def train_model(use_barrier=True):
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    mse = nn.MSELoss()
    gamma = 5.0 if use_barrier else 0.0

    for epoch in range(200):
        optimizer.zero_grad()
        preds = model(X_train)
        fidelity = mse(preds, y_train)
        rm = X_train[:, 5]  # RM index 5
        viol = monotonic_violation(preds, rm)
        factor = torch.exp(gamma * viol)
        loss = fidelity * factor
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Fidelity {fidelity:.4f}, Viol {viol:.6f}, Factor {factor:.2f}')

    with torch.no_grad():
        preds_test = model(X_test)
        mse_test = mse(preds_test, y_test).item()
        rm_test = X_test[:, 5]
        viol_test = monotonic_violation(preds_test, rm_test).item()
        compliance = 100.0 if viol_test == 0 else (1 - viol_test / preds_test.std().item()) * 100  # approx % compliance

    print(f'\n{"Barrier" if use_barrier else "Baseline"} Results:')
    print(f'Test MSE: {mse_test:.4f}')
    print(f'Test Violation: {viol_test:.6f}')
    print(f'Compliance: {compliance:.2f}%')

print('Unconstrained Baseline:')
train_model(use_barrier=False)

print('\nMultiplicative Barrier:')
train_model(use_barrier=True)
