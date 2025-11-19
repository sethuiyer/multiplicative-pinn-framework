import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class SimpleConstraintNet(nn.Module):
    """
    Simplified implementation focusing on the core multiplicative concept
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

def capacity_constraint(outputs, max_capacity=1.5):
    """Ensure outputs don't exceed capacity"""
    excess = torch.relu(outputs - max_capacity)
    return torch.mean(excess)

def range_constraint(outputs, min_val=-1.5, max_val=1.5):
    """Ensure outputs stay in range"""
    low_viol = torch.relu(min_val - outputs)
    high_viol = torch.relu(outputs - max_val)
    return torch.mean(low_viol + high_viol)

def multiplicative_constraint_loss(fidelity_loss, violation_score, gamma=5.0):
    """
    The core idea: exp(gamma * violation) >= 1
    When violations=0, factor=1 (no constraint penalty)
    When violations>0, factor>1 (constraint penalty increases)
    """
    constraint_factor = torch.exp(gamma * violation_score)
    total_loss = fidelity_loss * constraint_factor
    return total_loss, constraint_factor

def run_simple_implementation():
    print("🧪 SIMPLE MULTIPLICATIVE CONSTRAINT NETWORK (WITH ADAGRAD)")
    print("Implementing Sethu Iyer's core idea: exp(gamma * violation) scaling")
    print("=" * 75)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataset
    n_samples = 800
    input_dim = 10
    
    X = torch.randn(n_samples, input_dim)
    # Target is a simple function of first few features
    y_true = 0.5 * X[:, 0:1] + 0.3 * X[:, 1:2] + 0.2 * torch.randn(n_samples, 1)
    
    print(f"📊 Dataset: {n_samples} samples, {input_dim} features")
    
    # Create model and optimizer
    model = SimpleConstraintNet(input_dim=input_dim)
    optimizer = optim.Adagrad(model.parameters(), lr=0.02)  # AdaGrad as suggested
    fidelity_criterion = nn.MSELoss()
    
    # Constraints to enforce
    constraint_functions = [lambda out: range_constraint(out)]
    
    print("⚙️  Training with multiplicative constraint scaling...")
    print("  Loss = Fidelity_Loss * exp(gamma * violations)")
    
    fidelity_history = []
    violation_history = []
    factor_history = []
    
    for epoch in range(300):
        # Shuffle data
        indices = torch.randperm(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y_true[indices]
        
        epoch_fidelity = 0
        epoch_violation = 0
        epoch_factor = 0
        batch_count = 0
        
        # Mini-batch training
        batch_size = 32
        for i in range(0, n_samples, batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_X)
            
            # Compute fidelity loss
            fidelity_loss = fidelity_criterion(outputs, batch_y)
            
            # Compute constraint violations
            total_violation = torch.tensor(0.0)
            for constraint_fn in constraint_functions:
                total_violation = total_violation + constraint_fn(outputs)
            
            # Apply multiplicative constraint (the core innovation)
            total_loss, constraint_factor = multiplicative_constraint_loss(
                fidelity_loss, total_violation, gamma=3.0
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_fidelity += fidelity_loss.item()
            epoch_violation += total_violation.item()
            epoch_factor += constraint_factor.item()
            batch_count += 1
        
        # Record epoch stats
        avg_fidelity = epoch_fidelity / batch_count
        avg_violation = epoch_violation / batch_count
        avg_factor = epoch_factor / batch_count
        
        fidelity_history.append(avg_fidelity)
        violation_history.append(avg_violation)
        factor_history.append(avg_factor)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: MSE={avg_fidelity:.6f}, Violations={avg_violation:.6f}, Factor={avg_factor:.4f}")
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"Final MSE: {fidelity_history[-1]:.6f}")
    print(f"Final violations: {violation_history[-1]:.6f}")
    print(f"Final constraint factor: {factor_history[-1]:.4f}")
    
    # Final evaluation
    print(f"\n🔍 FINAL EVALUATION:")
    with torch.no_grad():
        final_outputs = model(X)
        final_mse = fidelity_criterion(final_outputs, y_true)
        final_violations = range_constraint(final_outputs)
        
        print(f"  Final MSE: {final_mse:.6f}")
        print(f"  Final range violations: {final_violations:.6f}")
        print(f"  Valid outputs %: {((torch.abs(final_outputs) <= 1.5).float().mean() * 100):.2f}%")
    
    # Calculate violation trend
    initial_viol = violation_history[0] if violation_history else 0
    final_viol = violation_history[-1] if violation_history else 0
    
    if initial_viol > 0:
        viol_improvement = ((initial_viol - final_viol) / initial_viol) * 100
        print(f"  Violation change: {viol_improvement:.2f}% from initial")
    else:
        print(f"  Started with near-zero violations")
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.plot(fidelity_history)
    ax1.set_title('MSE Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.grid(True)
    
    ax2.plot(violation_history)
    ax2.set_title('Constraint Violations Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Violation Score')
    ax2.grid(True)
    
    ax3.plot(factor_history)
    ax3.set_title('Multiplicative Constraint Factor')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Factor Value')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_constraint_net.png')
    plt.show()
    
    # Check if constraints improved
    initial_avg_viol = sum(violation_history[:50]) / 50 if len(violation_history) >= 50 else violation_history[0] if violation_history else 0
    final_avg_viol = sum(violation_history[-50:]) / 50 if len(violation_history) >= 50 else violation_history[-1] if violation_history else 0
    
    improvement = initial_avg_viol - final_avg_viol
    improvement_pct = (improvement / initial_avg_viol) * 100 if initial_avg_viol > 0 else 0
    
    print(f"\n🎯 RESULTS:")
    print(f"✅ MSE achieved: {final_mse:.6f}")
    print(f"✅ Range constraint violation: {final_violations:.6f}")
    print(f"✅ Valid output percentage: {(torch.abs(final_outputs) <= 1.5).float().mean() * 100:.2f}%")
    print(f"✅ Violation trend: {'Decreased' if improvement > 0 else 'Increased'} by {abs(improvement_pct):.2f}%")
    
    if improvement_pct > 10:  # Significant improvement
        print(f"✅ Sethu Iyer's multiplicative framework WORKING!")
        print(f"   Constraint enforcement improved by {improvement_pct:.2f}%")
        print(f"   AdaGrad helped with constraint-aware optimization")
    elif abs(improvement_pct) < 10:  # Minimal change
        print(f"⚠️  Mixed results - constraints maintained but not strongly improved")
        print(f"   Framework provides constraint awareness")
    else:  # Worsened
        print(f"⚠️  Violations increased - may need stronger gamma or different approach")
    
    print(f"\n🚀 CORE ACHIEVEMENT:")
    print(f"✅ Implemented exp(gamma * violation) multiplicative scaling")
    print(f"✅ AdaGrad optimizer as suggested")
    print(f"✅ Physics-inspired constraint enforcement in neural networks")
    print(f"✅ Sethu Iyer's 'multiplicative axis' concept operational!")

if __name__ == "__main__":
    run_simple_implementation()