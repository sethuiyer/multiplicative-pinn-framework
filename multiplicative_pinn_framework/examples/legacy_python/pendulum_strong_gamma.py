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
        
    def dynamics(self, theta, omega, torque):
        """Pendulum dynamics: theta_dot = omega, omega_dot = -(g/l)*sin(theta) + torque/(m*l^2)"""
        # Simplified pendulum: omega_dot = -g/l*sin(theta) + torque/(m*l^2)
        gravity_force = -self.gravity / self.length * math.sin(theta)
        torque_force = torque / (self.mass * self.length * self.length)
        omega_dot = gravity_force + torque_force
        
        # Limit speed
        omega_next = max(-self.max_speed, min(self.max_speed, omega + omega_dot * self.dt))
        theta_next = theta + omega_next * self.dt
        
        return theta_next, omega_next

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

def multiplicative_loss(fidelity, viol, gamma=20.0):
    """Multiplicative barrier loss: fidelity * exp(gamma * violation)"""
    factor = torch.exp(gamma * viol)
    return fidelity * factor, factor

def train_multiplicative(model, train_loader, device, gamma=20.0):
    """Train with multiplicative constraint enforcement - STRONGER Gamma"""
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)  # AdaGrad as suggested
    mse = nn.MSELoss()
    
    model.train()
    for epoch in range(300):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            out = model(batch_x)
            fid_loss = mse(out, batch_y)
            viol = violation(out)
            
            total_loss, factor = multiplicative_loss(fid_loss, viol, gamma=gamma)
            
            # Only update if loss is finite
            if torch.isfinite(total_loss):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            epoch_loss += total_loss.item() if torch.isfinite(total_loss) else 0
        
        if epoch % 100 == 0:
            print(f"Multiplicative (γ={gamma}) - Epoch {epoch}: Loss = {epoch_loss:.6f}")

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

def run_strong_gamma_experiment():
    print("🧪 STRONG GAMMA TEST: Multiplicative Barrier with γ=20.0")
    print("Testing Sethu Iyer's framework with stronger constraint enforcement")
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
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data shape: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Test target statistics
    target_in_range = (torch.abs(torch.from_numpy(targets)) <= 0.8).float().mean().item() * 100
    print(f"Target torques in safe range [-0.8, 0.8]: {target_in_range:.2f}%")
    print("Note: Only ~12% of targets in safe range → very hard test case\n")
    
    # Train with STRONG gamma
    print("🚀 TRAINING: Multiplicative Barrier with STRONG γ=20.0")
    model_mult = Net().to(device)
    train_multiplicative(model_mult, train_loader, device, gamma=20.0)
    result = evaluate_model(model_mult, test_loader, device)
    
    print(f"\n🎯 RESULTS:")
    print(f"In-Range %: {result['in_range_pct']:.2f}%")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"Violation Severity: {result['violation_severity']:.6f}")
    
    # Pass criteria evaluation
    multiplicative_pass = (result['in_range_pct'] >= 95.0)
    
    print(f"\n🏆 PASS CRITERIA EVALUATION:")
    print(f"Multiplicative method in-range ≥ 95%: {'✅' if multiplicative_pass else '❌'} ({result['in_range_pct']:.2f}%)")
    
    if result['in_range_pct'] >= 85.0:
        print(f"\n✅ GOOD: Strong gamma enforcement working!")
        if result['in_range_pct'] >= 95.0:
            print(f"🏆 INCREDIBLE: Sethu Iyer's framework passed strict criteria!")
        else:
            print(f"📊 PROMISING: Very close to 95% target")
    else:
        print(f"\n⚠️  Stricter gamma needed")
    
    print(f"\n📈 RESULT: The multiplicative barrier with γ=20.0 achieved {result['in_range_pct']:.2f}% safety compliance!")
    
    return result

if __name__ == "__main__":
    run_strong_gamma_experiment()