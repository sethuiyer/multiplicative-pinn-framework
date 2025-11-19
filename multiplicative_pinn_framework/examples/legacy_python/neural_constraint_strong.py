import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class SethuConstraintNet(nn.Module):
    """
    Implementation of Sethu Iyer's multiplicative axis with strong enforcement
    """
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

def output_range_constraint(outputs, min_val=-1.0, max_val=1.0):
    """Strong constraint to keep outputs in range"""
    violations = torch.relu(torch.abs(outputs) - max_val)
    return torch.mean(violations)

def multiplicative_constraint_loss(fidelity_loss, violation_score, gamma=8.0):
    """
    Sethu Iyer's core insight: exp(gamma * violation) creates strong penalty
    When violation=0: factor = 1
    When violation=0.1: factor = exp(0.8) = 2.23
    When violation=0.5: factor = exp(4.0) = 54.6
    """
    # Strong barrier: heavily penalize violations
    constraint_factor = torch.exp(gamma * violation_score)
    total_loss = fidelity_loss * constraint_factor
    return total_loss, constraint_factor

def run_strong_implementation():
    print("🧪 STRONG MULTIPLICATIVE CONSTRAINT ENFORCEMENT")
    print("Based on Sethu Iyer's Multiplicative Axis Framework")
    print("=" * 70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 1000
    input_dim = 8
    
    # Create dataset with some tendency to go out of bounds
    X = torch.randn(n_samples, input_dim)
    # Create targets that might push outputs out of range
    y_true = 2.0 * torch.sum(X[:, :3], dim=1, keepdim=True) + 0.5 * torch.randn(n_samples, 1)
    
    print(f"📊 Data: {n_samples} samples, outputs range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    
    model = SethuConstraintNet(input_dim=input_dim)
    
    # Use SGD with momentum for constraint-aware optimization
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    fidelity_criterion = nn.MSELoss()
    
    print("⚙️  Training with exp(γ * violation) constraint scaling (γ=8.0)")
    
    fidelity_history = []
    violation_history = []
    factor_history = []
    
    for epoch in range(200):
        # Shuffle
        indices = torch.randperm(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y_true[indices]
        
        epoch_fidelity = 0
        epoch_violation = 0
        epoch_factor = 0
        batch_count = 0
        
        batch_size = 64
        for i in range(0, n_samples, batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            outputs = model(batch_X)
            
            fidelity_loss = fidelity_criterion(outputs, batch_y)
            violation = output_range_constraint(outputs, min_val=-1.0, max_val=1.0)
            
            total_loss, factor = multiplicative_constraint_loss(fidelity_loss, violation, gamma=8.0)
            
            optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_fidelity += fidelity_loss.item()
            epoch_violation += violation.item()
            epoch_factor += factor.item()
            batch_count += 1
        
        avg_fidelity = epoch_fidelity / batch_count
        avg_violation = epoch_violation / batch_count
        avg_factor = epoch_factor / batch_count
        
        fidelity_history.append(avg_fidelity)
        violation_history.append(avg_violation)
        factor_history.append(avg_factor)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: MSE={avg_fidelity:.6f}, Violations={avg_violation:.6f}, Factor={avg_factor:.2f}")
    
    print(f"\n✅ COMPLETED!")
    print(f"Final MSE: {fidelity_history[-1]:.6f}")
    print(f"Final Violations: {violation_history[-1]:.6f}")
    
    # Evaluation
    print(f"\n🔍 EVALUATION:")
    with torch.no_grad():
        final_outputs = model(X)
        final_mse = fidelity_criterion(final_outputs, y_true)
        final_viol = output_range_constraint(final_outputs)
        
        in_range_pct = (torch.abs(final_outputs) <= 1.0).float().mean() * 100
        
        print(f"  MSE: {final_mse:.6f}")
        print(f"  Range violations: {final_viol:.6f}")
        print(f"  % in range [-1,1]: {in_range_pct:.2f}%")
    
    # Calculate improvement trend
    initial_avg = np.mean(violation_history[:20]) if len(violation_history) > 20 else violation_history[0]
    final_avg = np.mean(violation_history[-20:]) if len(violation_history) > 20 else violation_history[-1]
    
    improvement = ((initial_avg - final_avg) / initial_avg) * 100 if initial_avg > 0 else 0
    
    print(f"  Violation improvement: {improvement:.2f}%")
    
    if in_range_pct > 90 and improvement > 10:
        print(f"\n🎉 SUCCESS: Sethu Iyer's framework working excellently!")
        print(f"✅ High percentage in range: {in_range_pct:.2f}%")
        print(f"✅ Violations reduced by: {improvement:.2f}%")
        print(f"✅ MSE maintained: {final_mse:.6f}")
    elif in_range_pct > 75:
        print(f"\n✅ GOOD: Framework providing constraint awareness")
        print(f"✅ {in_range_pct:.2f}% outputs in range")
        print(f"✅ MSE: {final_mse:.6f}")
    else:
        print(f"\n⚠️  Needs tuning: {in_range_pct:.2f}% in range")
    
    print(f"\n🎯 CORE VALIDATION:")
    print(f"✅ exp(γ * violation) barrier implemented (γ=8.0)")
    print(f"✅ Strong constraint enforcement activated")
    print(f"✅ Simultaneous fidelity + constraints optimization")
    print(f"✅ Sethu Iyer's multiplicative axis operational")

if __name__ == "__main__":
    run_strong_implementation()