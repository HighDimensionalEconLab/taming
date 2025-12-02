import numpy as np
import torch
from joblib import Memory
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root

# Standard methods on the hypercube here are very slow in python,
# even with JAX or Pytorch.  So we cache the results after the first run.
# Subsequent runs will be instantaneous unless you change the arguments.
memory = Memory(location=".cache", verbose=0)


@memory.cache
def full_euler_iteration_torch(
    alpha,
    delta,
    sigma,
    rho,
    nu_nodes,
    beta,
    nu_weights,
    k_grid,
    z_grid,
    s_ss,
    method,
    verbose,
):
    z_grid_min = z_grid.min()
    z_grid_max = z_grid.max()
    k_grid_min = k_grid.min()
    k_grid_max = k_grid.max()

    def c(state, k_prime_func):
        k, z = state[..., 0], state[..., 1]
        return torch.relu(
            torch.exp(z) ** (1 - alpha) * k**alpha
            + (1 - delta) * k
            - k_prime_func(state)
        )

    def euler_residual(state, k_prime_func):
        c_t = c(state, k_prime_func)
        k_t, z_t = state[..., 0], state[..., 1]
        k_tp1 = torch.clamp(k_prime_func(state), k_grid_min, k_grid_max)
        z_tp1 = torch.clamp(
            rho * z_t.unsqueeze(-1) + sigma * torch.tensor(nu_nodes, dtype=k_t.dtype),
            z_grid_min,
            z_grid_max,
        )
        k_tp1_b = k_tp1.unsqueeze(-1).expand(-1, len(nu_nodes))
        states_tp1 = torch.stack([k_tp1_b, z_tp1], dim=-1)
        c_tp1 = c(states_tp1, k_prime_func)
        term_val = (c_t.unsqueeze(-1) / c_tp1) * (
            1
            - delta
            + alpha * (torch.exp(z_tp1) ** (1 - alpha)) * (k_tp1_b ** (alpha - 1))
        )
        exp_val = torch.sum(
            torch.tensor(nu_weights, dtype=term_val.dtype) * term_val, dim=-1
        )
        return 1 - beta * exp_val

    # Initial condition with Solow
    def k_prime_solow(state):
        k, z = state[..., 0], state[..., 1]
        return s_ss * (torch.exp(z) ** (1 - alpha)) * k**alpha + (1 - delta) * k

    # Create meshgrid for k and z
    k_long, z_long = np.meshgrid(k_grid, z_grid, indexing="ij")
    k_flat = k_long.ravel()
    z_flat = z_long.ravel()
    data_grid = np.stack((k_flat, z_flat), axis=-1)
    k_prime_init = (
        k_prime_solow(torch.from_numpy(data_grid).float()).numpy().reshape(k_long.shape)
    )

    # Residuals function for root finding
    def residuals(k_prime_states_flat):
        k_prime_states = k_prime_states_flat.reshape(k_long.shape)
        interp = RegularGridInterpolator(
            (k_grid, z_grid),
            k_prime_states,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        def k_prime_func(state):
            # state: torch tensor of shape (..., 2)
            pts = state.detach().cpu().numpy()
            return torch.from_numpy(interp(pts)).to(state.device)

        # Evaluate on all grid points
        state_tensor = torch.from_numpy(data_grid).float()
        res = euler_residual(state_tensor, k_prime_func)
        return res.detach().cpu().numpy().ravel()

    print(
        "Solving for baseline k_prime using classical methods (slow on first run, then cached)"
    )
    sol = root(residuals, k_prime_init.ravel(), method=method)
    if verbose:
        print(
            f"Solved: {sol.nfev} evals, ||residuals||_2 = {np.linalg.norm(sol.fun):.2e}"
        )
    k_prime_values = sol.x.reshape(k_long.shape)
    euler_residuals = sol.fun.reshape(k_long.shape)
    return k_grid, z_grid, k_prime_values, euler_residuals, sol.success


def stochastic_growth_baseline(
    beta,
    alpha,
    delta,
    rho,
    sigma,
    num_quad_nodes,
    k_grid_min_mul,
    k_grid_max_mul,
    z_grid_mul,
    num_z_points,
    num_k_points,
    z_sd_grid,  # possibly use different than true sigma
    verbose,
    method,
):
    # Non-stochastic steady state
    k_ss = (alpha / (1 / beta - 1 + delta)) ** (1 / (1 - alpha))
    c_ss = k_ss**alpha - delta * k_ss
    s_ss = delta * k_ss ** (1 - alpha)
    k_max = (1 / delta) ** (1 / (1 - alpha))
    z_ergodic_sd = np.sqrt(sigma**2 / (1 - rho**2))

    assert sigma > 0 or z_sd_grid is not None

    k_grid_min = k_grid_min_mul * k_ss
    k_grid_max = min(k_grid_max_mul * k_ss, k_max - 1e-6)
    k_grid = np.linspace(k_grid_min, k_grid_max, num_k_points)

    z_grid, _ = np.polynomial.hermite.hermgauss(num_z_points)
    z_grid_sd = z_ergodic_sd if z_sd_grid is None else z_sd_grid
    z_grid = z_grid * z_grid_mul * z_grid_sd / (np.max(np.abs(z_grid)))

    if sigma > 0:
        nu_nodes, nu_weights = np.polynomial.hermite.hermgauss(num_quad_nodes)
        nu_weights = nu_weights / np.sqrt(np.pi)
        nu_nodes = nu_nodes * np.sqrt(2)
    else:
        nu_nodes = np.array([0.0])
        nu_weights = np.array([1.0])

    k_grid, z_grid, k_prime_values, euler_residuals, success = (
        full_euler_iteration_torch(
            alpha,
            delta,
            sigma,
            rho,
            nu_nodes,
            beta,
            nu_weights,
            k_grid,
            z_grid,
            s_ss,
            method,
            verbose,
        )
    )

    # Note that this is silently extrapolating.  Careful comparing
    # the baseline to the NN solutions (which are not similarly constrained)
    interp = RegularGridInterpolator(
        (k_grid, z_grid),
        k_prime_values,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    def k_prime(states):
        k = states[..., 0]
        z = states[..., 1]
        pts = np.stack(
            [
                np.atleast_1d(k.detach().cpu().numpy()),
                np.atleast_1d(z.detach().cpu().numpy()),
            ],
            axis=-1,
        )
        out = interp(pts)
        return torch.from_numpy(out).float().to(states.device).unsqueeze(-1)

    results = {
        "success": success,
        "k_prime": k_prime,
        "k_ss": k_ss,
        "k_max": k_max,
        "c_ss": c_ss,
        "s_ss": s_ss,
        "z_ergodic_sd": z_ergodic_sd,
        "k_grid": k_grid,
        "z_grid": z_grid,
        "baseline_abs_euler_residual_mean": np.abs(euler_residuals).mean(),
    }
    return results
