import time
from dataclasses import dataclass

import jsonargparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from stochastic_growth_baseline_pytorch import stochastic_growth_baseline


# NN not especially sensitive to ReLU vs. LeakyReLU vs. SiLU
# nn.Softplus enforces k' >= 0, but often not binding
def k_prime_HC(width: int, depth: int):
    layers = [nn.Linear(2, width), nn.LeakyReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(width, width), nn.LeakyReLU()])
    layers.extend([nn.Linear(width, 1), nn.Softplus()])
    return nn.Sequential(*layers)


@dataclass
class BaselineSolverSettings:
    # multipliers of the k_ss or non-stochastic steady state
    k_grid_min_mul: float = 0.7
    k_grid_max_mul: float = 1.4
    z_grid_mul: float = 4.0  # standard deviations around zero
    num_z_points: int = 31
    num_k_points: int = 100
    method: str = "lm"


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
    pretrain_max_iter: int = 50  # LBFGS iterations for pretraining (fitting to Solow)
    max_iter: int = 20  # LBFGS iterations per epoch in main training
    max_epochs: int = 10
    max_train_time: float = 180.0
    # Thresholds to determine if we should retry optimization with a new initialization
    test_loss_success_threshold: float = 1e-7
    transversality_residual_threshold: float = 0.001
    num_attempts: int = 5
    early_stopping_loss_divergence: float = 10.0


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
    freeze_backbone: bool = False,
    data_set: DataSettings = DataSettings(),
    opt_set: OptimizerSettings = OptimizerSettings(),
    base_solver_set: BaselineSolverSettings = BaselineSolverSettings(),
    use_cpu: bool = True,
    verbose: bool = True,
):
    # Allow other devices - but CPU is much faster for this problem size
    if use_cpu:
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS for macOS
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU if available
    else:
        device = torch.device("cpu")  # Fallback to CPU

    if verbose:
        print(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Find the baseline results and closed form for non-stochastic steady state
    baseline_results = stochastic_growth_baseline(
        beta=beta,
        alpha=alpha,
        delta=delta,
        rho=rho,
        sigma=sigma,
        num_quad_nodes=num_quad_nodes,
        k_grid_min_mul=base_solver_set.k_grid_min_mul,
        k_grid_max_mul=base_solver_set.k_grid_max_mul,
        z_grid_mul=base_solver_set.z_grid_mul,
        num_z_points=base_solver_set.num_z_points,
        num_k_points=base_solver_set.num_k_points,
        z_sd_grid=None,
        verbose=verbose,
        method=base_solver_set.method,
    )
    k_0 = torch.tensor(k_0_multiplier * baseline_results["k_ss"], device=device)
    k_ss = torch.tensor(baseline_results["k_ss"], device=device)
    c_ss = torch.tensor(baseline_results["c_ss"], device=device)
    k_prime_baseline = baseline_results["k_prime"]

    # Grid bounds for clamping state_0 draws
    k_min = baseline_results["k_grid"].min()
    k_max = baseline_results["k_grid"].max()
    z_min = baseline_results["z_grid"].min()
    z_max = baseline_results["z_grid"].max()

    # Weights for Gaussian quadrature with nu ~ N(0,1)
    nu_nodes_np, nu_weights_np = np.polynomial.hermite.hermgauss(num_quad_nodes)
    nu_weights = torch.tensor(
        nu_weights_np / np.sqrt(np.pi), dtype=torch.float32, device=device
    )
    nu_nodes = torch.tensor(
        nu_nodes_np * np.sqrt(2), dtype=torch.float32, device=device
    )

    # Use the solow policy at the stationary solution as an initial condition for the policy
    # and for generating the initial trajectories
    def k_prime_solow(state):
        k, z = state[..., 0], state[..., 1]
        return (
            baseline_results["s_ss"] * (torch.exp(z) ** (1 - alpha)) * k**alpha
            + (1 - delta) * k
        ).unsqueeze(-1)

    def draw_state_0(k0, z0, num_trajectories):
        noise = torch.randn(num_trajectories, 2, device=device) * torch.tensor(
            [data_set.state_0_k_std, data_set.state_0_z_std],
            device=device,
        )
        init = noise + torch.tensor([k0, z0], device=device)
        # Clamp k and z to stay within the baseline grid bounds or else baseline
        # interpolation is extrapolating.  Not an issue for the NN
        init[:, 0] = torch.clamp(init[:, 0], min=k_min, max=k_max)
        init[:, 1] = torch.clamp(init[:, 1], min=z_min, max=z_max)
        return init

    # z' = \rho z + \sigma \nu$ where $\nu \sim \mathcal{N}(0,1), and k'(k,z)
    def simulate_trajectories(k_prime, state_0, shocks, sigma):
        with torch.no_grad():
            N, T = shocks.shape
            traj = torch.zeros(N, T, 2, device=device)
            X = state_0.clone()
            for t in range(T):
                kp = k_prime(X).squeeze(-1)
                X_next = torch.stack([kp, rho * X[:, 1] + sigma * shocks[:, t]], dim=-1)
                # Clamp to ensure that the baseline results are valid. Irrelevant for the NN itself
                X_next[:, 0] = torch.clamp(X_next[:, 0], min=k_min, max=k_max)
                X_next[:, 1] = torch.clamp(X_next[:, 1], min=z_min, max=z_max)
                traj[:, t, :] = X
                X = X_next
            return traj

    # From resource constraint: c(z,k; k') \equiv \exp(z)^{1 - \alpha} k^{\alpha} + (1-\delta) k - k'(z,k)
    def c(state, k_prime):
        k, z = state[..., 0], state[..., 1]
        kp = k_prime(state).squeeze(-1)
        return torch.exp(z) ** (1 - alpha) * k**alpha + (1 - delta) * k - kp

    # Euler: expectation is taken over z' = \rho z + \sigma \nu for \nu \sim N(0,1)
    #    1  = \mathbb{E}\left[ \beta \frac{c(z,k)}{c(z', k'(z,k))}\left(1 - \delta + \alpha \exp(z')^{1 - \alpha} k'(z,k)^{\alpha-1}\right)\right]
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
        # Uses Gaussian quadrature weights for the nu shock
        exp_val = torch.sum(nu_weights * term_val, dim=-1)
        return 1 - beta * exp_val

    # Checks against the baseline for a given k_prime policy
    def gen_results(trajectories, k_prime):
        with torch.no_grad():
            N, T, _ = trajectories.shape
            states = trajectories.reshape(-1, 2)
            kp = k_prime(states).squeeze(-1)
            c_val = c(states, k_prime)
            resid = euler_residuals(states, k_prime)
            k = states[:, 0]
            z = states[:, 1]
        # Use baseline to calculate relative errors
        k_prime_baseline_values = k_prime_baseline(states).squeeze(-1).cpu().numpy()
        rel_error_values = (
            kp.cpu().numpy() - k_prime_baseline_values
        ) / k_prime_baseline_values

        # Create flat indices for trajectory and time
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

    for attempt in range(1, opt_set.num_attempts + 1):
        # Create policy function approximation
        k_prime = k_prime_HC(width=mlp_width, depth=mlp_depth).to(device)

        train_shocks = torch.randn(
            data_set.num_train_trajectories, data_set.train_T, device=device
        )
        train_state_0 = draw_state_0(
            k_0,
            z_0,
            data_set.num_train_trajectories,
        )

        # Generate initial training trajectories using Solow policy
        train_trajectories = simulate_trajectories(
            k_prime_solow, train_state_0, train_shocks, sigma
        )
        train_data = train_trajectories.reshape(-1, 2)

        # Save initial training trajectories for visualization
        df_train_initial, _ = gen_results(train_trajectories, k_prime_solow)

        # Pretraining to fit k' to the solow policy
        with torch.no_grad():
            k_solow_train = k_prime_solow(train_data)

        optimizer = optim.LBFGS(
            k_prime.parameters(),
            lr=opt_set.lr,
            max_iter=opt_set.pretrain_max_iter,
            line_search_fn="strong_wolfe",
        )

        # LBFGS uses a closure to evaluate the loss at each iteration
        # Parameters are implicitly updated within the optimizer step
        def pretrain_loss_closure():
            optimizer.zero_grad()
            pred = k_prime(train_data)
            loss_val = torch.mean((pred - k_solow_train) ** 2)
            loss_val.backward()
            return loss_val

        # LBFGS will iterate to its own convergence
        optimizer.step(pretrain_loss_closure)
        pretrain_n_iter = optimizer.state[optimizer._params[0]].get("n_iter", 0)

        # Optionally: freeze all layers except the last linear layer after pretraining
        if freeze_backbone:
            for i, layer in enumerate(k_prime):
                # Fix all parameters except the 2nd last linear layer
                if isinstance(layer, nn.Linear) and i < len(k_prime) - 2:
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False

        # Main training with LBFGS
        optimizer = optim.LBFGS(
            # Only optimize over trainable parameters
            filter(lambda p: p.requires_grad, k_prime.parameters()),
            lr=opt_set.lr,
            max_iter=opt_set.max_iter,
            line_search_fn="strong_wolfe",
        )

        start_time = time.time()
        stopping_reason = "max_epochs"
        progress_bar = tqdm(
            range(opt_set.max_epochs), desc=f"Attempt {attempt}, Processing"
        )

        # An "epoch" here is a complete LBFGS optimization
        # Between each epoch regenerate the "data" with the latest policy function
        for epoch in progress_bar:

            def euler_loss_closure():
                optimizer.zero_grad()
                resid = euler_residuals(train_data, k_prime)
                loss = torch.mean(resid**2)
                loss.backward()
                return loss

            # Run Optimization (LBFGS will iterate to its own convergence)
            # Pytorch updates parameters implicitly
            loss = optimizer.step(euler_loss_closure)

            # Get LBFGS iterations for this epoch
            epoch_n_iter = optimizer.state[optimizer._params[0]].get("n_iter", 0)

            last_loss = loss.detach().cpu().numpy()
            elapsed_time = time.time() - start_time
            with torch.no_grad():
                k_prime_ss_ratio = (
                    k_prime(torch.tensor([[k_ss, 0.0]], device=device)).item() / k_ss
                )

            progress_bar.set_description(
                f"Attempt {attempt}, loss={last_loss:.6e}, k'(k_ss)/k_ss={k_prime_ss_ratio:.3f}, n_iter={epoch_n_iter}"
            )

            if elapsed_time > opt_set.max_train_time:
                stopping_reason = "max_time_reached"
                break
            if last_loss > opt_set.early_stopping_loss_divergence or np.isnan(
                last_loss
            ):
                stopping_reason = "loss_divergence"
                break

            # Regenerate trajectories every epoch to adapt to the distribution induced by current policy
            train_shocks = torch.randn(
                data_set.num_train_trajectories, data_set.train_T, device=device
            )
            train_trajectories = simulate_trajectories(
                k_prime, train_state_0, train_shocks, sigma
            )
            train_data = train_trajectories.reshape(-1, 2)

        # For a candidate solution verify the transversality condition for a single k_0, z_0
        transversality_shocks = torch.randn(
            data_set.transversality_check_trajectories,
            data_set.transversality_check_T,
            device=device,
        )
        transversality_state_0 = draw_state_0(
            k_0,
            z_0,
            data_set.transversality_check_trajectories,
        )
        # Simulate trajectories to time T using k' policy
        transversality_traj = simulate_trajectories(
            k_prime, transversality_state_0, transversality_shocks, sigma
        )

        # Calculate approximate transversality residuals with simulations
        state_T = transversality_traj[:, -1, :]  # This is (k_{T-1}, z_{T-1})
        CS_ss = k_ss / c_ss
        with torch.no_grad():
            c_vals = c(state_T, k_prime)  # c_{T-1}
            kp_vals = k_prime(state_T).squeeze(-1)  # k_T
            # Check beta^{T-1} * k_T / c_{T-1} to match state timing
            tv_values = (
                (
                    (beta ** (data_set.transversality_check_T - 1))
                    * ((kp_vals / c_vals - CS_ss) / CS_ss)
                )
                .cpu()
                .numpy()
            )
        transversality_residual = np.mean(tv_values)  # expected value

        # Check generate new"test" trajectories to see whether
        #  generalization performance is acceptable (e.g., not overfitting)
        test_shocks = torch.randn(
            data_set.num_test_trajectories, data_set.test_T, device=device
        )
        test_state_0 = draw_state_0(
            k_0,
            z_0,
            data_set.num_test_trajectories,
        )
        test_trajectories = simulate_trajectories(
            k_prime, test_state_0, test_shocks, sigma
        )

        df_train_final, train_loss_final = gen_results(train_trajectories, k_prime)
        df_test, test_loss = gen_results(test_trajectories, k_prime)
        df_test_baseline, test_loss_baseline = gen_results(
            test_trajectories, k_prime_baseline
        )

        # Retry if not converged
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
            "baseline_abs_euler_residual_mean": baseline_results[
                "baseline_abs_euler_residual_mean"
            ],
            "k_ss_ratio": (k_prime(torch.tensor([[k_ss, 0.0]], device=device))).item()
            / k_ss,
            "k_ss": baseline_results["k_ss"],
            "c_ss": baseline_results["c_ss"],
            "s_ss": baseline_results["s_ss"],
            "k_max": baseline_results["k_max"],
            "z_ergodic_sd": baseline_results["z_ergodic_sd"],
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
        }

        if solution_converged:
            break

    # Return results from the successful attempt or the last attempt if none converged
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
