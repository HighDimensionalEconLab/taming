import json
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from stochastic_growth_pytorch import (
    BaselineSolverSettings,
    DataSettings,
    OptimizerSettings,
    stochastic_growth,
)

# Matplotlib style parameters
PLOT_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "figure.figsize": (12, 6),
    "figure.dpi": 80,
    "figure.edgecolor": "k",
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
}


def plot_epoch(ax, epoch_data, y_val, y_label, title="Plot Title", t_steps=10):
    ax.clear()
    trajectories = epoch_data["trajectory"].unique()

    for traj in trajectories:
        traj_data = epoch_data[epoch_data["trajectory"] == traj]
        ax.plot(traj_data["t"], traj_data[y_val], color="blue", alpha=0.2)

    ax.set_xlabel("t")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MultipleLocator(t_steps))
    ax.grid(False)
    return ax


def main(output_dir="./.figures"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Arguments copied from defaults to ensure reproducibility if the defaults changed later
    beta = 0.99
    alpha = 1 / 3
    delta = 0.025
    rho = 0.9
    sigma = 0.01
    z_0 = 0.0
    k_0_multiplier = 0.8
    seed = 42
    num_quad_nodes = 7
    mlp_width = 64
    mlp_depth = 4
    verbose = False

    base_solver_set = BaselineSolverSettings(
        k_grid_min_mul=0.7,
        k_grid_max_mul=1.4,
        z_grid_mul=4.0,  # standard deviations around zero
        num_z_points=31,
        num_k_points=100,
    )
    data_set = DataSettings(
        train_T=60,
        num_train_trajectories=20,
        num_test_trajectories=50,
        test_T=60,
        transversality_check_T=200,
        transversality_check_trajectories=20,
        state_0_k_std=0.1,
        state_0_z_std=0.023,
    )
    opt_set = OptimizerSettings(
        pretrain_max_iter=50,
        max_iter=20,
        max_epochs=10,
        test_loss_success_threshold=1e-7,
        transversality_residual_threshold=0.001,
        num_attempts=5,
    )

    # Solve the model
    res = stochastic_growth(
        beta=beta,
        alpha=alpha,
        delta=delta,
        rho=rho,
        sigma=sigma,
        z_0=z_0,
        k_0_multiplier=k_0_multiplier,
        seed=seed,
        num_quad_nodes=num_quad_nodes,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        verbose=verbose,
        data_set=data_set,
        opt_set=opt_set,
        base_solver_set=base_solver_set,
    )

    results = res["results"]

    # Generate comparison figure for k trajectories
    plt.rcParams.update(PLOT_PARAMS)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    # Plot initial training trajectories
    plot_epoch(
        ax1,
        res["df_train_initial"],
        y_val="k",
        y_label="k(t)",
        title="Initial Training Trajectories",
    )

    # Plot solution trajectories
    plot_epoch(
        ax2, res["df_test"], y_val="k", y_label="k(t)", title="Solution Trajectories"
    )

    plt.tight_layout()

    # Save and display figure
    plot_name = "solution_k_trajectories"
    output_path = f"{output_dir}/{plot_name}.pdf"
    plt.savefig(output_path)
    print(f"Saved figure: {output_path}")

    # Plots the initial data and final policy errors (linear scale)
    plt.rcParams.update(PLOT_PARAMS)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    # Plot initial training trajectories
    plot_epoch(
        ax1,
        res["df_train_initial"],
        y_val="abs_rel_error",
        y_label="err(t)",
        title="Initial Relative Policy Error $|(k'(t) - k'_{ref}(t))/k'_{ref}(t)|$",
    )

    # Plot solution trajectories (limit to same time range as training)
    max_t_train = res["df_train_initial"]["t"].max()
    test_data_limited = res["df_test"][res["df_test"]["t"] <= max_t_train]
    plot_epoch(
        ax2,
        test_data_limited,
        y_val="abs_rel_error",
        y_label="err(t)",
        title="Solution Relative Policy Error $|(k'(t) - k'_{ref}(t))/k'_{ref}(t)|$",
    )

    plt.tight_layout()

    plot_name = "solution_err_trajectories"
    output_path = f"{output_dir}/{plot_name}.pdf"
    plt.savefig(output_path)
    print(f"Saved figure: {output_path}")

    # Now change to log scale and save again
    for ax in [ax1, ax2]:
        ax.set_yscale("log")

    plt.tight_layout()

    plot_name = "solution_log_err_trajectories"
    output_path = f"{output_dir}/{plot_name}.pdf"
    plt.savefig(output_path)
    print(f"Saved figure: {output_path}")

    # Convert numpy types to Python types to prepare for JSON serialization
    results = {
        k: float(v) if hasattr(v, "item") else v for k, v in res["results"].items()
    }

    # Add additional parameters not already in results
    results.update(
        {
            "beta": beta,
            "alpha": alpha,
            "delta": delta,
            "rho": rho,
            "sigma": sigma,
            "z_0": z_0,
            "k_0_multiplier": k_0_multiplier,
            "k_0_dist": f"~ N({k_0_multiplier}*k_ss, {data_set.state_0_k_std}^2)",
            "z_0_dist": f"~ N({z_0}, {data_set.state_0_z_std}^2)",
            "seed": seed,
            "num_quad_nodes": num_quad_nodes,
            "train_T": data_set.train_T,
            "num_train_trajectories": data_set.num_train_trajectories,
            "test_T": data_set.test_T,
            "num_test_trajectories": data_set.num_test_trajectories,
            "total_test_data": data_set.test_T * data_set.num_test_trajectories,
            "state_0_k_std": data_set.state_0_k_std,
            "state_0_z_std": data_set.state_0_z_std,
            "transversality_check_T": data_set.transversality_check_T,
            "transversality_check_trajectories": data_set.transversality_check_trajectories,
            "max_pretrain_iter": opt_set.pretrain_max_iter,
            "max_iter_per_epoch": opt_set.max_iter,
            "max_epochs": opt_set.max_epochs,
            "test_loss_success_threshold": opt_set.test_loss_success_threshold,
            "transversality_residual_threshold": opt_set.transversality_residual_threshold,
            "num_attempts": opt_set.num_attempts,
            "k_grid_min_mul": base_solver_set.k_grid_min_mul,
            "k_grid_max_mul": base_solver_set.k_grid_max_mul,
            "z_grid_mul": base_solver_set.z_grid_mul,
            "num_z_points": base_solver_set.num_z_points,
            "num_k_points": base_solver_set.num_k_points,
        }
    )

    results_path = f"{output_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
