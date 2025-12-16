import time
from dataclasses import dataclass
from typing import Optional

import jsonargparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Wrapper for a 2D RegularGridInterpolator using grid_sample
# The values of the function are "learnable" and differentiable
class RegularGridInterpolator(nn.Module):
    def __init__(self, points, values):
        super().__init__()
        k_grid, z_grid = points
        mins = torch.stack((k_grid[0], z_grid[0]))
        ranges = torch.stack((k_grid[-1] - k_grid[0], z_grid[-1] - z_grid[0]))
        self.register_buffer("mins", mins)
        self.register_buffer("ranges", ranges)
        self.values = nn.Parameter(values.view(1, 1, *values.shape))

    def forward(self, xi):
        original_shape = xi.shape[:-1]
        norm_xi = 2 * (xi - self.mins) / self.ranges - 1
        grid = norm_xi.flip(-1).view(1, 1, -1, 2)
        out = F.grid_sample(
            self.values,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return out.view(*original_shape, 1)


@dataclass
class BaselineSolverSettings:
    k_grid_min_mul: float = 0.7
    k_grid_max_mul: float = 1.4
    z_grid_mul: float = 4.0
    num_z_points: int = 31
    num_k_points: int = 100
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 300
    lbfgs_tolerance_grad: float = 1e-14
    lbfgs_tolerance_change: float = 1e-14
    lbfgs_history_size: int = 100
    lbfgs_max_eval: Optional[int] = 1000


@dataclass
class DataSettings:
    train_T: int = 60
    num_train_trajectories: int = 20
    num_test_trajectories: int = 50
    test_T: int = 60
    transversality_check_T: int = 200
    transversality_check_trajectories: int = 20
    state_0_k_std: float = 0.1
    state_0_z_std: float = 0.023


@dataclass
class OptimizerSettings:
    lr: float = 1.0
    pretrain_max_iter: int = 50
    max_iter: int = 20
    max_epochs: int = 10
    max_train_time: float = 180.0
    test_loss_success_threshold: float = 1e-7
    transversality_residual_threshold: float = 0.001
    num_attempts: int = 5
    early_stopping_loss_divergence: float = 10.0


def k_prime_HC(width: int, depth: int):
    # NN not especially sensitive to activation choice; Softplus enforces k' >= 0
    layers = [nn.Linear(2, width), nn.LeakyReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(width, width), nn.LeakyReLU()])
    layers.extend([nn.Linear(width, 1), nn.Softplus()])
    return nn.Sequential(*layers)


def stochastic_growth(
    beta: float = 0.99,
    alpha: float = 1 / 3,
    delta: float = 0.025,
    rho: float = 0.9,
    sigma: float = 0.01,
    z_0: float = 0.0,
    k_0_multiplier: float = 0.8,
    seed: int = 42,
    num_quad_nodes: int = 7,
    mlp_width: int = 64,
    mlp_depth: int = 4,
    data_set: DataSettings = DataSettings(),
    opt_set: OptimizerSettings = OptimizerSettings(),
    base_solver_set: BaselineSolverSettings = BaselineSolverSettings(),
    verbose: bool = True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert sigma > 0 and abs(rho) < 1
    # Gaussian quadrature nodes/weights for nu ~ N(0,1)
    nu_nodes_np, nu_weights_np = np.polynomial.hermite.hermgauss(num_quad_nodes)
    nu_weights = torch.tensor(nu_weights_np / np.sqrt(np.pi)).float()
    nu_nodes = torch.tensor(nu_nodes_np * np.sqrt(2)).float()

    # Resource constraint: c(z,k; k') = exp(z)^{1-a} k^a + (1-d)k - k'
    def c(state, k_prime):
        k, z = state[..., 0], state[..., 1]
        kp = k_prime(state).squeeze(-1)
        return torch.exp(z) ** (1 - alpha) * k**alpha + (1 - delta) * k - kp

    # Euler: 1 = E[ beta * (c_t / c_{t+1}) * (1-d + a * exp(z_{t+1})^{1-a} k_{t+1}^{a-1}) ]
    def euler_residuals(state, k_prime):
        c_t = c(state, k_prime).unsqueeze(-1)
        k_tp1 = k_prime(state)
        z_t = state[:, 1].unsqueeze(-1)
        z_tp1 = rho * z_t + sigma * nu_nodes
        k_tp1_b = k_tp1.expand(-1, len(nu_nodes))
        states_tp1 = torch.stack([k_tp1_b, z_tp1], dim=-1)
        c_tp1 = c(states_tp1, k_prime)
        term_val = (c_t / c_tp1) * (
            1
            - delta
            + alpha * (torch.exp(z_tp1) ** (1 - alpha)) * (k_tp1_b ** (alpha - 1))
        )
        # Gaussian quadrature weights over nu ~ N(0,1)
        exp_val = torch.sum(nu_weights * term_val, dim=-1)
        return 1 - beta * exp_val

    # Baseline solved on grid with LBFGS using shared Euler definition
    k_ss = (alpha / (1 / beta - 1 + delta)) ** (1 / (1 - alpha))
    c_ss = k_ss**alpha - delta * k_ss
    s_ss = delta * k_ss ** (1 - alpha)
    k_max_val = (1 / delta) ** (1 / (1 - alpha))
    z_ergodic_sd = np.sqrt(sigma**2 / (1 - rho**2)) if rho != 1 else np.inf

    k_grid_min = base_solver_set.k_grid_min_mul * k_ss
    k_grid_max = min(base_solver_set.k_grid_max_mul * k_ss, k_max_val - 1e-6)
    k_grid = torch.linspace(k_grid_min, k_grid_max, base_solver_set.num_k_points)
    z_grid_sd = z_ergodic_sd
    z_grid = torch.linspace(
        -base_solver_set.z_grid_mul * z_grid_sd,
        base_solver_set.z_grid_mul * z_grid_sd,
        base_solver_set.num_z_points,
    )

    k_long, z_long = torch.meshgrid(k_grid, z_grid, indexing="ij")
    train_states_baseline = torch.stack([k_long.flatten(), z_long.flatten()], dim=-1)

    def k_prime_solow(state):
        k, z = state[..., 0], state[..., 1]
        return (
            s_ss * (torch.exp(z) ** (1 - alpha)) * k**alpha + (1 - delta) * k
        ).unsqueeze(-1)

    # Baseline interpolates initialized to the Solow policy
    k_prime_baseline = RegularGridInterpolator(
        (k_grid, z_grid),
        k_prime_solow(torch.stack([k_long, z_long], dim=-1)).squeeze(-1),
    )

    # Optimize the interpolation by minimizing euler residuals using LBFGS
    optimizer_baseline = optim.LBFGS(
        k_prime_baseline.parameters(),
        lr=base_solver_set.lbfgs_lr,
        max_iter=base_solver_set.lbfgs_max_iter,
        max_eval=base_solver_set.lbfgs_max_eval,
        tolerance_grad=base_solver_set.lbfgs_tolerance_grad,
        tolerance_change=base_solver_set.lbfgs_tolerance_change,
        history_size=base_solver_set.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )

    # LBFGS uses a closure and updates the k_prime_baseline in-place
    def loss_baseline_closure():
        optimizer_baseline.zero_grad()
        resid = euler_residuals(train_states_baseline, k_prime_baseline)
        loss_val = torch.mean(resid**2)
        loss_val.backward()
        return loss_val

    # Run the optimizer
    optimizer_baseline.step(loss_baseline_closure)
    baseline_lbfgs_n_iter = optimizer_baseline.state[optimizer_baseline._params[0]].get(
        "n_iter", 0
    )

    # Checking the results from the baseline
    with torch.no_grad():
        baseline_resid = euler_residuals(train_states_baseline, k_prime_baseline)
        baseline_train_loss = torch.mean(baseline_resid**2).item()
        baseline_abs_euler_residual_mean = torch.mean(torch.abs(baseline_resid)).item()
        baseline_abs_euler_residual_max = torch.max(torch.abs(baseline_resid)).item()

    # Generate initial conditions for trajectories
    k_0 = torch.tensor(k_0_multiplier * k_ss)
    k_ss_tensor = torch.tensor([[k_ss, 0.0]])

    # Draw initial states with small random perturbations around (k0, z0)
    def draw_state_0(k0, z0, num_trajectories):
        noise = torch.randn(num_trajectories, 2) * torch.tensor(
            [data_set.state_0_k_std, data_set.state_0_z_std],
        )
        init = noise + torch.tensor([k0, z0])
        return init

    # Simulate trajectories given a policy function k_prime
    def simulate_trajectories(k_prime, state_0, shocks):
        # State dynamics: z_{t+1} = rho z_t + sigma nu_t, nu_t ~ N(0,1); k_{t+1} = k'(k_t, z_t)
        with torch.no_grad():
            N, T = shocks.shape
            traj = torch.zeros(N, T, 2)
            X = state_0.clone()
            for t in range(T):
                kp = k_prime(X).squeeze(-1)
                X_next = torch.stack([kp, rho * X[:, 1] + sigma * shocks[:, t]], dim=-1)
                traj[:, t, :] = X
                X = X_next
            return traj

    # Utility function to calculate results and errors on trajectories given a k_prime
    def gen_results(trajectories, k_prime):
        with torch.no_grad():
            states = trajectories.reshape(-1, 2)
            kp = k_prime(states).squeeze(-1)
            c_val = c(states, k_prime)
            resid = euler_residuals(states, k_prime)
            k = states[:, 0]
            z = states[:, 1]
        with torch.no_grad():
            k_prime_baseline_values = (
                k_prime_baseline(states).squeeze(-1).detach().cpu().numpy()
            )
        rel_error_values = (
            kp.cpu().numpy() - k_prime_baseline_values
        ) / k_prime_baseline_values
        flat_indices = [
            (i, t)
            for i in range(trajectories.shape[0])
            for t in range(trajectories.shape[1])
        ]
        df = pd.DataFrame(
            {
                "trajectory": [i for i, t in flat_indices],
                "t": [t for i, t in flat_indices],
                "k": k.cpu().numpy(),
                "z": z.cpu().numpy(),
                "k_prime": kp.cpu().numpy(),
                "c": c_val.cpu().numpy(),
                "euler_residual": resid.cpu().numpy(),
                "k_prime_baseline": k_prime_baseline_values.flatten(),
                "rel_error": rel_error_values.flatten(),
                "abs_rel_error": np.abs(rel_error_values).flatten(),
            }
        )
        loss_value = torch.mean(resid**2).cpu().numpy()
        return df, loss_value

    # Main algorithm for fitting a NN policy on simulated data
    # Checks convergence criteria and retries if not met
    for attempt in range(1, opt_set.num_attempts + 1):
        # Use the NN policy for k_prime instead of linear interpolation
        k_prime = k_prime_HC(width=mlp_width, depth=mlp_depth)

        # Initial trajectories from Solow policy provide a sensible starting dataset
        train_shocks = torch.randn(data_set.num_train_trajectories, data_set.train_T)
        train_state_0 = draw_state_0(
            k_0,
            z_0,
            data_set.num_train_trajectories,
        )
        train_trajectories = simulate_trajectories(
            k_prime_solow, train_state_0, train_shocks
        )
        train_data = train_trajectories.reshape(-1, 2)

        df_train_initial, _ = gen_results(train_trajectories, k_prime_solow)

        # Calculate the solow policy on the training data
        with torch.no_grad():
            k_solow_train = k_prime_solow(train_data)

        # "Pretrain" the NN to fit the Solow policy on the training data with LBFGS
        pretrain_optimizer = optim.LBFGS(
            k_prime.parameters(),
            lr=opt_set.lr,
            max_iter=opt_set.pretrain_max_iter,
            line_search_fn="strong_wolfe",
        )

        def pretrain_loss_closure():
            pretrain_optimizer.zero_grad()
            pred = k_prime(train_data)
            loss_val = torch.mean((pred - k_solow_train) ** 2)
            loss_val.backward()
            return loss_val

        # Run optimizer for pretraining
        pretrain_optimizer.step(pretrain_loss_closure)
        pretrain_n_iter = pretrain_optimizer.state[pretrain_optimizer._params[0]].get(
            "n_iter", 0
        )

        # Now setup an optimizer to fit the Euler equation residuals on the training data
        optimizer = optim.LBFGS(
            k_prime.parameters(),
            lr=opt_set.lr,
            max_iter=opt_set.max_iter,
            line_search_fn="strong_wolfe",
        )

        start_time = time.time()
        stopping_reason = "max_epochs"
        progress_bar = range(opt_set.max_epochs)

        for epoch in progress_bar:

            def loss_closure():
                optimizer.zero_grad()
                resid = euler_residuals(train_data, k_prime)
                loss = torch.mean(resid**2)
                loss.backward()
                return loss

            loss = optimizer.step(loss_closure)

            epoch_n_iter = optimizer.state[optimizer._params[0]].get("n_iter", 0)

            last_loss = loss.detach().cpu().numpy()
            elapsed_time = time.time() - start_time
            with torch.no_grad():
                k_prime_ss_ratio = k_prime(k_ss_tensor).item() / k_ss

            if verbose:
                print(
                    f"Attempt {attempt}, epoch {epoch}, loss={last_loss:.6e}, "
                    f"k'(k_ss)/k_ss={k_prime_ss_ratio:.3f}, n_iter={epoch_n_iter}"
                )

            if elapsed_time > opt_set.max_train_time:
                stopping_reason = "max_time_reached"
                break
            if last_loss > opt_set.early_stopping_loss_divergence or np.isnan(
                last_loss
            ):
                stopping_reason = "loss_divergence"
                break

            # Refresh simulated data using the updated policy each epoch
            train_shocks = torch.randn(
                data_set.num_train_trajectories, data_set.train_T
            )
            train_trajectories = simulate_trajectories(
                k_prime, train_state_0, train_shocks
            )
            train_data = train_trajectories.reshape(-1, 2)

        # Build transversality condition check
        transversality_shocks = torch.randn(
            data_set.transversality_check_trajectories,
            data_set.transversality_check_T,
        )
        transversality_state_0 = draw_state_0(
            k_0,
            z_0,
            data_set.transversality_check_trajectories,
        )
        transversality_traj = simulate_trajectories(
            k_prime, transversality_state_0, transversality_shocks
        )

        state_T = transversality_traj[:, -1, :]
        CS_ss = k_ss / c_ss
        with torch.no_grad():
            c_vals = c(state_T, k_prime)
            kp_vals = k_prime(state_T).squeeze(-1)
            # Normalized deviation from steady-state ratio, discounted by beta^T
            tv_values = (
                (
                    (beta ** (data_set.transversality_check_T - 1))
                    * ((kp_vals / c_vals - CS_ss) / CS_ss)
                )
                .cpu()
                .numpy()
            )
        transversality_residual = np.mean(tv_values)

        # Hold-out trajectories to gauge generalization vs. baseline interpolant
        test_shocks = torch.randn(data_set.num_test_trajectories, data_set.test_T)
        test_state_0 = draw_state_0(
            k_0,
            z_0,
            data_set.num_test_trajectories,
        )
        test_trajectories = simulate_trajectories(k_prime, test_state_0, test_shocks)

        # Evaluate NN and baseline on both train and test trajectories
        df_train_final, train_loss_final = gen_results(train_trajectories, k_prime)
        df_test, test_loss = gen_results(test_trajectories, k_prime)
        df_test_baseline, test_loss_baseline = gen_results(
            test_trajectories, k_prime_baseline
        )

        # Convergence: low test Euler residuals AND transversality condition satisfied
        solution_converged = (
            test_loss < opt_set.test_loss_success_threshold
            and abs(transversality_residual)
            <= opt_set.transversality_residual_threshold
        )

        results = {
            "test_loss": test_loss,
            "train_loss": train_loss_final,
            "transversality_residual": transversality_residual,
            "stopping_reason": stopping_reason,
            "train_abs_euler_residual": df_train_final["euler_residual"].abs().mean(),
            "test_abs_euler_residual": df_test["euler_residual"].abs().mean(),
            "test_baseline_abs_euler_residual": df_test_baseline["euler_residual"]
            .abs()
            .mean(),
            "baseline_abs_euler_residual_mean": baseline_abs_euler_residual_mean,
            "k_ss_ratio": (k_prime(k_ss_tensor)).item() / k_ss,
            "k_ss": k_ss,
            "c_ss": c_ss,
            "s_ss": s_ss,
            "k_max": k_max_val,
            "z_ergodic_sd": z_ergodic_sd,
            "total_data": train_data.shape[0],
            "mlp_width": mlp_width,
            "mlp_depth": mlp_depth,
            "activation": k_prime[1].__class__.__name__,
            "total_params": sum(p.numel() for p in k_prime.parameters()),
            "trainable_params": sum(
                p.numel() for p in k_prime.parameters() if p.requires_grad
            ),
            "pretrain_n_iter": pretrain_n_iter,
            "attempt": attempt,
            "elapsed_time": elapsed_time,
            "solution_converged": solution_converged,
            "test_abs_rel_error": df_test["abs_rel_error"].mean(),
            "train_abs_rel_error": df_train_final["abs_rel_error"].mean(),
            "baseline_train_loss": baseline_train_loss,
            "baseline_abs_euler_residual_max": baseline_abs_euler_residual_max,
            "baseline_lbfgs_n_iter": baseline_lbfgs_n_iter,
        }

        if solution_converged:
            break

    if verbose:
        print("\nFinal Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

    return {
        "results": results,
        "df_test": df_test,
        "df_train_final": df_train_final,
        "df_train_initial": df_train_initial,
    }


if __name__ == "__main__":
    jsonargparse.CLI(stochastic_growth)
