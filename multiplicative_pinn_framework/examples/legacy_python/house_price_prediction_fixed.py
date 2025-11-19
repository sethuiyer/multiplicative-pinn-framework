# house_price_safe_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# 1. Generate realistic-but-poisoned data
# -------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

n_samples = 60_000
n_features = 15

X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                       noise=0.1, random_state=42)
y = np.abs(y) * 120 + 380_000  # shift to realistic prices ~$380k–$800k

# Inject real-world garbage
neg_mask = np.random.rand(n_samples) < 0.018
y[neg_mask] = -np.random.exponential(100_000, size=neg_mask.sum())

rich_mask = np.random.rand(n_samples) < 0.009
y[rich_mask] = 15_000_000 + np.random.exponential(20_000_000, size=rich_mask.sum())

y = y.astype(np.float32)
X = X.astype(np.float32)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10_000, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train).reshape(-1,1)
y_test = torch.from_numpy(y_test).reshape(-1,1)

# Clamp negatives to 0 for sanity (real data cleaning)
y_train = torch.clamp(y_train, min=0)
y_test = torch.clamp(y_test, min=0)

# Z-normalize
y_mean = y_train.mean()
y_std = y_train.std() + 1e-8  # avoid div by zero
y_train_norm = (y_train - y_mean) / y_std
y_test_norm = (y_test - y_mean) / y_std

# -------------------------------------------------
# 2. Model + Multiplicative Barrier
# -------------------------------------------------
class SafeHouseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)    # raw normalized output
        )
    def forward(self, x):
        return self.net(x)

model = SafeHouseNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

def violation_score(prices, lower=0.0, upper=10_000_000.0):
    below = torch.relu(lower - prices)
    above = torch.relu(prices - upper)
    return (below + above).mean()

gamma = 10.0   # Lower for dollar-scale violations

# -------------------------------------------------
# 3. Training loop
# -------------------------------------------------
train_losses = []
violations = []

for epoch in range(400):
    model.train()
    perm = torch.randperm(X_train.size(0))
    epoch_loss = 0
    epoch_viol = 0
    
    for i in range(0, X_train.size(0), 256):
        idx = perm[i:i+256]
        x = X_train[idx]
        y_norm = y_train_norm[idx]
        
        pred_norm = model(x)
        pred_price = pred_norm * y_std + y_mean  # denormalize for violation check
        
        fidelity = mse(pred_norm, y_norm)
        fidelity = torch.clamp(fidelity, max=5000.0)  # safety clamp
        
        viol = violation_score(pred_price)
        
        loss = fidelity * torch.exp(gamma * viol)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += fidelity.item()
        epoch_viol += viol.item()
    
    train_losses.append(epoch_loss)
    violations.append(epoch_viol)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Fidelity {fidelity.item():.4f} | Viol {viol.item():.8f} | Factor {torch.exp(gamma * viol).item():.2f}")

# -------------------------------------------------
# 4. Final results
# -------------------------------------------------
model.eval()
with torch.no_grad():
    pred_norm = model(X_test)
    pred_price = pred_norm * y_std + y_mean
    final_mse = mse(pred_norm, y_test_norm).item()
    final_rmse_dollars = torch.sqrt(mse(pred_price, y_test)).item()
    
    below_zero = (pred_price < 0).sum().item()
    above_10m = (pred_price > 10_000_000).sum().item()
    in_range = 100.0 * (pred_price.shape[0] - below_zero - above_10m) / pred_price.shape[0]
    
    print("\n" + "="*60)
    print("FINAL RESULTS – SAFE HOUSE PRICE PREDICTOR")
    print("="*60)
    print(f"MSE on test set          : {final_mse:.6f}")
    print(f"RMSE in dollars          : {final_rmse_dollars:,.2f}")
    print(f"Negative prices predicted: {below_zero}  ({below_zero/10000:.4f}%)")
    print(f"Prices > $10M predicted  : {above_10m}  ({above_10m/10000:.4f}%)")
    print(f"Prices in [$0, $10M]     : {in_range:.4f}%")
    print(f"Min price predicted      : ${pred_price.min().item():,.2f}")
    print(f"Max price predicted      : ${pred_price.max().item():,.2f}")

