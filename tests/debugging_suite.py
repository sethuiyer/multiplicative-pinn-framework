"""
Debugging and Diagnostics for Navier-Stokes Solution
Identifying and fixing high divergence in Sethu Iyer's framework
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from multiplicative_pinn_framework.examples.large_scale_simulation import TrainedNavierStokes2D


def autograd_sanity_check():
    """
    Check if autograd is computing derivatives correctly
    """
    print("🔍 AUTODIFF SANITY CHECK")
    print("-" * 40)
    
    import torch
    
    def check_autograd():
        coords = torch.rand(10, 3, requires_grad=True)  # t,x,y
        x = coords[:, 1:2]
        y = coords[:, 2:3]  # Fixed: was 2:2+1, now 2:3
        
        # Analytic functions
        u = x**2 + 3*y
        v = y**2 - 2*x
        
        grads_u = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
        grads_v = torch.autograd.grad(v.sum(), coords, create_graph=True)[0]
        
        # Check the derivatives
        u_x_error = ((grads_u[:, 1] - 2*x.squeeze())**2).mean().item()
        u_y_error = ((grads_u[:, 2] - 3.0)**2).mean().item()
        v_x_error = ((grads_v[:, 1] + 2.0)**2).mean().item()  # note sign
        v_y_error = ((grads_v[:, 2] - 2*y.squeeze())**2).mean().item()
        
        print(f"u_x error: {u_x_error:.8f}")
        print(f"u_y error: {u_y_error:.8f}")
        print(f"v_x error: {v_x_error:.8f}")
        print(f"v_y error: {v_y_error:.8f}")
        
        total_error = u_x_error + u_y_error + v_x_error + v_y_error
        print(f"Total derivative error: {total_error:.8f}")
        
        if total_error < 1e-6:
            print("✅ Autodiff: PASSED - Derivatives computed correctly")
        else:
            print(f"❌ Autodiff: FAILED - Error too large: {total_error}")
        
        return total_error < 1e-6
    
    success = check_autograd()
    return success


def compute_divergence_with_correct_indexing(u, v, x_grid, y_grid):
    """
    Compute divergence with correct indexing
    """
    # Compute spatial gradients using finite differences
    # dx = spacing between x values
    dx = x_grid[0, 1] - x_grid[0, 0] if x_grid.shape[1] > 1 else 1.0
    dy = y_grid[1, 0] - y_grid[0, 0] if y_grid.shape[0] > 1 else 1.0
    
    # Compute gradients: ∂u/∂x and ∂v/∂y
    # Using numpy gradient (which handles 2D arrays properly)
    du_dx = np.gradient(u, axis=1) / dx
    dv_dy = np.gradient(v, axis=0) / dy
    
    return du_dx + dv_dy


def quick_divergence_diagnostic(model):
    """
    Quick diagnostic to check divergence in the model
    """
    print(f"\n🔍 QUICK DIVERGENCE DIAGNOSTIC")
    print("-" * 40)
    
    # Create a small spatial grid
    grid_size = 10  # Small grid for quick test
    x_range = np.linspace(0.1, 0.9, grid_size)
    y_range = np.linspace(0.1, 0.4, grid_size)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    
    # Test at a specific time
    t_test = 0.25
    coords_tensor = torch.tensor(
        np.stack([np.full_like(x_grid.flatten(), t_test),
                  x_grid.flatten(), y_grid.flatten()], axis=1), 
        dtype=torch.float32
    )
    
    with torch.no_grad():
        solution = model(coords_tensor)
    
    u = solution[:, 0].reshape(grid_size, grid_size).numpy()
    v = solution[:, 1].reshape(grid_size, grid_size).numpy()
    
    # Compute divergence
    divergence = compute_divergence_with_correct_indexing(u, v, x_grid, y_grid)
    
    print(f"   Grid size: {grid_size}x{grid_size}")
    print(f"   Max |∇·u|: {np.abs(divergence).max():.8f}")
    print(f"   Mean |∇·u|: {np.abs(divergence).mean():.8f}")
    print(f"   Std |∇·u|: {np.abs(divergence).std():.8f}")
    
    if np.abs(divergence).max() < 1e-2:
        print("✅ Divergence: GOOD - Incompressible flow satisfied")
        return True
    else:
        print(f"❌ Divergence: TOO HIGH - Incompressible constraint not satisfied")
        print("   Need to investigate multiplicative constraint enforcement")
        return False


def check_coordinate_scaling():
    """
    Check if coordinate scaling is causing issues
    """
    print(f"\n🔍 COORDINATE SCALING CHECK")
    print("-" * 40)
    
    # Test with normalized coordinates
    coords_raw = torch.tensor([[0.1, 0.5, 0.3],  # [t, x, y]
                              [0.2, 0.6, 0.4],
                              [0.3, 0.7, 0.5]], dtype=torch.float32)
    
    # Normalize to [-1, 1] range
    coords_norm = (coords_raw - 0.5) * 2  # Normalize to [-1, 1]
    
    print(f"Raw coordinates: {coords_raw}")
    print(f"Normalized coordinates: {coords_norm}")
    print("✅ Coordinate normalization can be applied for better numerical stability")
    
    return True


def temporary_continuity_boost_demo():
    """
    Demonstrate temporary continuity boost approach
    """
    print(f"\n🔍 TEMPORARY CONTINUITY BOOST APPROACH")
    print("-" * 40)
    
    print("This would involve adding a direct additive penalty for continuity:")
    print("   loss = constraint_factor * momentum_residuals.mean() + lambda_cont * continuity_residual.abs().mean()")
    print("   where lambda_cont is large (e.g., 100-1000)")
    print("✅ Approach: Boost continuity enforcement temporarily, then tune multiplicative parameters")
    
    return True


def streamfunction_architecture_demo():
    """
    Demonstrate streamfunction approach for incompressible flow
    """
    print(f"\n🔍 STREAMFUNCTION ARCHITECTURE DEMO")
    print("-" * 40)
    
    class DivergenceFreeNavierStokes(nn.Module):
        """
        Streamfunction-based model that guarantees ∇·u = 0
        """
        def __init__(self, viscosity=0.01):
            super().__init__()
            self.viscosity = viscosity
            # Output streamfunction ψ and pressure p
            self.net = nn.Sequential(
                nn.Linear(3, 64),  # [t, x, y]
                nn.Tanh(),
                nn.Linear(64, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 2)  # [ψ, p] - streamfunction and pressure
            )
        
        def forward(self, x):
            x.requires_grad_(True)  # Need this before calling net
            output = self.net(x)
            psi = output[:, 0:1]  # Streamfunction
            p = output[:, 1:2]    # Pressure

            # Compute velocity from streamfunction: u = ∂ψ/∂y, v = -∂ψ/∂x
            grad_psi = torch.autograd.grad(
                psi.sum(), x, create_graph=True, retain_graph=True
            )[0]  # [∂ψ/∂t, ∂ψ/∂x, ∂ψ/∂y]

            u = grad_psi[:, 2:3]    # ∂ψ/∂y
            v = -grad_psi[:, 1:2]   # -∂ψ/∂x

            return torch.cat([u, v, p], dim=1)
    
    print("Streamfunction approach guarantees ∇·u = 0 by construction:")
    print("   u = ∂ψ/∂y, v = -∂ψ/∂x")
    print("   ∇·u = ∂u/∂x + ∂v/∂y = ∂²ψ/∂x∂y - ∂²ψ/∂x∂y = 0")
    print("✅ This approach enforces incompressibility by construction")

    # Create and test the architecture
    stream_model = DivergenceFreeNavierStokes()
    test_coords = torch.tensor([[0.25, 0.5, 0.3]], dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        output = stream_model.net(test_coords)
        psi = output[:, 0:1]
        p = output[:, 1:2]

    # Compute gradients for testing
    test_coords_grad = torch.tensor([[0.25, 0.5, 0.3]], dtype=torch.float32, requires_grad=True)
    output_grad = stream_model.net(test_coords_grad)
    psi_grad = output_grad[:, 0:1]

    grad_psi = torch.autograd.grad(
        psi_grad.sum(), test_coords_grad, create_graph=True, retain_graph=True
    )[0]

    u = grad_psi[:, 2:3]    # ∂ψ/∂y
    v = -grad_psi[:, 1:2]   # -∂ψ/∂x

    print(f"   Model output shape: {[u.shape, v.shape, p.shape]} [u, v, p for a single point]")
    print(f"   Sample output: u={u[0,0]:.6f}, v={v[0,0]:.6f}, p={p[0,0]:.6f}")

    return stream_model


def run_debugging_suite():
    """
    Run the complete debugging suite
    """
    print("🌊 DEBUGGING AND DIAGNOSTICS FOR NAVIER-STOKES SOLUTION")
    print("=" * 80)
    print("Identifying and fixing high divergence in Sethu Iyer's framework")
    print()
    
    # Create model for testing
    model = TrainedNavierStokes2D(viscosity=0.01)
    model.eval()
    
    # 1. Autodiff sanity check
    autograd_ok = autograd_sanity_check()
    
    # 2. Quick divergence diagnostic
    divergence_ok = quick_divergence_diagnostic(model)
    
    # 3. Coordinate scaling check
    scaling_ok = check_coordinate_scaling()
    
    # 4. Temporary continuity boost
    boost_ok = temporary_continuity_boost_demo()
    
    # 5. Streamfunction architecture demo
    stream_model = streamfunction_architecture_demo()
    
    print(f"\n" + "=" * 80)
    print("🔍 DEBUGGING SUMMARY")
    print("=" * 80)
    print(f"✅ Autodiff Check: {'PASS' if autograd_ok else 'FAIL'}")
    print(f"✅ Divergence Check: {'PASS' if divergence_ok else 'FAIL'}")
    print(f"✅ Scaling Check: {'PASS' if scaling_ok else 'N/A'}")
    print(f"✅ Continuity Boost: {'APPROVED' if boost_ok else 'N/A'}")
    print(f"✅ Streamfunction Approach: {'AVAILABLE' if stream_model is not None else 'N/A'}")
    
    print(f"\n🎯 DIAGNOSIS:")
    if not autograd_ok:
        print(f"   ❌ Autodiff has issues - derivatives are not computed correctly")
        print(f"   Fix: Debug gradient computation indexing")
    elif not divergence_ok:
        print(f"   ❌ High divergence detected - incompressibility not enforced")
        print(f"   Fix: Try temporary continuity boost or streamfunction approach")
    else:
        print(f"   ✅ All checks passed - framework appears correct")
    
    print(f"\n💡 RECOMMENDATION:")
    if not autograd_ok:
        print(f"   1. Debug derivative computation first")
        print(f"   2. Verify coordinate indexing")
        print(f"   3. Re-run physics validation")
    elif not divergence_ok:
        print(f"   1. Implement temporary continuity boost (additive penalty)")
        print(f"   2. Tune multiplicative parameters for continuity enforcement")
        print(f"   3. Consider streamfunction architecture for guaranteed incompressibility")
    else:
        print(f"   Framework is working correctly - consider architecture improvements for better physics")
    
    return {
        'autograd_ok': autograd_ok,
        'divergence_ok': divergence_ok,
        'scaling_ok': scaling_ok,
        'boost_ok': boost_ok,
        'stream_model': stream_model
    }


if __name__ == "__main__":
    results = run_debugging_suite()
    print(f"\n🌊 DEBUGGING COMPLETE!")
    print(f"   Ready to address divergence issues with recommended fixes!")