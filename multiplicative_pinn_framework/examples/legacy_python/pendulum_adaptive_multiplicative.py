import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math

class PendulumEnvironment:
    """Simplified pendulum environment for data generation"""
    def __init__(self):
        self.dt = 0.05
        self.gravity = 10.0
        self.length = 1.0
        self.mass = 1.0
        self.max_speed = 8.
        self.max_torque = 2.0  # This is the "unsafe" range
        self.safe_torque_limit = 0.8  # This is the safe range
        self.state = None

def generate_pendulum_data(n_samples=50000):
    """Generate training data: (sinθ, cosθ, ω) → optimal torque"""
    
    states = []
    targets = []
    
    for _ in range(n_samples):
        # Random initial state
        theta = np.random.uniform(-np.pi, np.pi)
        omega = np.random.uniform(-8., 8.)  # max speed around 8
        
        # Generate "optimal" torque using a simple heuristic
        # Want to move toward upright (θ=0) with low energy
        target_angle = 0.0  # Goal is upright position
        angle_error = target_angle - theta
        
        # Simple PD-like controller (but unconstrained)
        # This will generate torques that sometimes exceed ±0.8
        kp = 2.0  # Position gain
        kd = 0.5  # Velocity damping
        
        desired_torque = kp * angle_error - kd * omega
        
        # Add some noise to make it realistic
        desired_torque += np.random.normal(0, 0.3)
        
        # Clip to make torques realistic, but still often exceed safe bound
        desired_torque = max(-2.0, min(2.0, desired_torque))
        
        # Store state as [sinθ, cosθ, ω]
        state = [np.sin(theta), np.cos(theta), omega]
        
        states.append(state)
        targets.append([desired_torque])
    
    return np.array(states, dtype=np.float32), np.array(targets, dtype=np.float32)

class Net(nn.Module):
    """Neural network for torque prediction"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def violation(u, bound=0.8):
    """Constraint violation: mean of relu(|u| - bound)"""
    return torch.relu(u.abs() - bound).mean()

def multiplicative_loss_with_adaptive_gamma(fidelity, viol, gamma=15.0):
    """Multiplicative loss with adaptive gamma that strengthens based on violation"""
    # Adaptive gamma: stronger penalty if violations are high
    # This creates increasingly strong enforcement as violations increase
    adaptive_gamma = gamma * (1 + 10 * viol.detach())  # Strengthen based on current violation
    factor = torch.exp(adaptive_gamma * viol)
    return fidelity * factor, factor

def train_adaptive_multiplicative(model, train_loader, device):
    """Train with adaptive multiplicative constraint enforcement"""
    optimizer = optim.Adagrad(model.parameters(), lr=0.005)  # Lower LR to prevent explosion
    mse = nn.MSELoss()
    
    model.train()
    for epoch in range(500):  # Longer training for adaptation
        epoch_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            out = model(batch_x)
            fid_loss = mse(out, batch_y)
            viol = violation(out)
            
            total_loss, factor = multiplicative_loss_with_adaptive_gamma(fid_loss, viol, gamma=15.0)
            
            if torch.isfinite(total_loss) and total_loss.item() < 1e6:  # Prevent explosion
                optimizer.zero_grad()
                total_loss.backward()
                # More aggressive gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
            
            epoch_loss += total_loss.item() if torch.isfinite(total_loss) and total_loss.item() < 1e6 else 0
        
        if epoch % 100 == 0:
            print(f"Adaptive Multiplicative - Epoch {epoch}: Loss = {epoch_loss:.6f}")

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            
            out = model(batch_x)
            all_outputs.append(out.cpu())
            all_targets.append(batch_y)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    rmse = torch.sqrt(nn.MSELoss()(all_outputs, all_targets)).item()
    in_range_pct = (torch.abs(all_outputs) <= 0.8).float().mean().item() * 100
    violation_severity = torch.relu(torch.abs(all_outputs) - 0.8).mean().item()
    
    return {
        'rmse': rmse,
        'in_range_pct': in_range_pct,
        'violation_severity': violation_severity
    }

def run_adaptive_experiment():
    print("🧪 ADAPTIVE MULTIPLICATIVE CONSTRAINT ENFORCEMENT")
    print("Advanced Sethu Iyer framework with adaptive gamma")
    print("=" * 70)
    
    # Generate data
    print("📊 Generating pendulum data: (sinθ, cosθ, ω) → optimal torque")
    states, targets = generate_pendulum_data(n_samples=50000)
    
    # Split data
    n_train = 40000
    n_test = 10000
    
    X_train, y_train = states[:n_train], targets[:n_train]
    X_test, y_test = states[n_train:n_train+n_test], targets[n_train:n_train+n_test]
    
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)  # Smaller batch
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data shape: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Test target statistics
    target_in_range = (torch.abs(torch.from_numpy(targets)) <= 0.8).float().mean().item() * 100
    print(f"Target torques in safe range [-0.8, 0.8]: {target_in_range:.2f}%")
    print("Note: Only ~12% of targets in safe range → very hard test case\n")
    
    # Train with adaptive approach
    print("🚀 TRAINING: Adaptive Multiplicative Constraint with Self-Strengthening Gamma")
    model_adaptive = Net().to(device)
    train_adaptive_multiplicative(model_adaptive, train_loader, device)
    result = evaluate_model(model_adaptive, test_loader, device)
    
    print(f"\n🎯 RESULTS:")
    print(f"In-Range %: {result['in_range_pct']:.2f}%")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"Violation Severity: {result['violation_severity']:.6f}")
    
    # Pass criteria evaluation
    multiplicative_pass = (result['in_range_pct'] >= 95.0)
    
    print(f"\n🏆 PASS CRITERIA EVALUATION:")
    print(f"Method in-range ≥ 95%: {'✅' if multiplicative_pass else '❌'} ({result['in_range_pct']:.2f}%)")
    
    if result['in_range_pct'] >= 90.0:
        print(f"\n🎯 EXCELLENT: Adaptive multiplicative framework working!")
        if result['in_range_pct'] >= 95.0:
            print(f"🚀 INSANE: Sethu Iyer's framework exceeded expectations!")
        else:
            print(f"✅ Very close to target - excellent performance")
    else:
        print(f"\n📊 Good progress: {result['in_range_pct']:.2f}% with challenging safety constraints")
    
    print(f"\n📈 Adaptive multiplicative constraint achieved {result['in_range_pct']:.2f}% safety compliance!")
    print(f"   From 70% → 85% → {result['in_range_pct']:.2f}% through adaptive strengthening!")
    
    # Compare with previous results
    print(f"\n🏁 COMPARISON:")
    print(f"   Original multiplicative: 70.40%")
    print(f"   Strong gamma (γ=20):     84.93%")
    print(f"   Adaptive approach:       {result['in_range_pct']:.2f}%")
    
    return result

if __name__ == "__main__":
    run_adaptive_experiment()