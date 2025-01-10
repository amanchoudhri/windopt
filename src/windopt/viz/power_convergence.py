"""
Plot power convergence over time of wind farm simulations.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from windopt.constants import PROJECT_ROOT, DT
from windopt.winc3d.io import turbine_results

DEFAULT_N_TIMESTEPS_PER_FILE = 9000
DEFAULT_SPINUP_TIMESTEPS = 9000

def power_trajectory(
        job_dir: Path,
        n_timesteps_per_file: int,
        spinup_timesteps: int,
        dt: float = DT
    ) -> pd.DataFrame:
    """
    Get the power trajectory (in MW) of a LES simulation from its job directory.
    """
    power_df = turbine_results(job_dir)

    # Throw out spinup timesteps
    spinup_filenumbers = int(spinup_timesteps / n_timesteps_per_file)
    power_df = power_df[power_df['filenumber'] > spinup_filenumbers]
    power_trajectory = power_df.groupby("filenumber")[["Power", "Power_ave"]].sum().reset_index()

    # Calculate average power per timestep
    power_trajectory['Power_ave'] = power_trajectory['Power_ave'] / (
        (1 + power_trajectory['filenumber'] * n_timesteps_per_file) - spinup_timesteps)

    # convert to MW
    power_trajectory['Power_ave'] = power_trajectory['Power_ave'] / 1e6
    power_trajectory['Power'] = power_trajectory['Power'] / 1e6

    # convert the filenumber to time after spinup, in hours
    files_after_spinup = power_trajectory['filenumber'] - spinup_filenumbers
    seconds_after_spinup = files_after_spinup * n_timesteps_per_file * dt
    power_trajectory['time_hr'] = seconds_after_spinup / 3600.0

    return power_trajectory

def plot_trajectory(
        ax: plt.Axes,
        trajectory_df: pd.DataFrame,
        title: Optional[str] = None,
        scatter_alpha: float = 0.4,
        line_alpha: float = 0.8
    ):
    sns.scatterplot(
        data=trajectory_df,
        x="time_hr",
        y="Power",
        alpha=scatter_alpha,
        ax=ax
    )
    sns.lineplot(
        data=trajectory_df,
        x="time_hr",
        y="Power_ave",
        alpha=line_alpha,
        ax=ax
    )
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Power (MW)')

    if title:
        ax.set_title(title)

    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', alpha=0.3)

def validate_grid_and_random(grid_dir: Path, random_dir: Path, outfile: Path):
    layouts = ("grid", "random")

    sns.set_theme(style="whitegrid")
    sns.set_context("talk")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

    for ax, layout, les_dir in zip(axs, layouts, (grid_dir, random_dir)):
        trajectory_df = power_trajectory(
            les_dir,
            n_timesteps_per_file=300,
            spinup_timesteps=9000,
        )
        plot_trajectory(ax, trajectory_df, title=layout.capitalize())

    fig.suptitle('Simulated Power Output Throughout LES')
    plt.tight_layout()

    # # save the plot
    plt.savefig(outfile, dpi=300)

def validate_grid_and_random_cli():
    p = ArgumentParser()
    p.add_argument('--grid_dir', type=Path, required=True)
    p.add_argument('--random_dir', type=Path, required=True)
    p.add_argument(
        '--outfile',
        type=Path,
        default=PROJECT_ROOT / 'img' / 'power_convergence.png'
        )
    args = p.parse_args()
    validate_grid_and_random(args.grid_dir, args.random_dir, args.outfile)


if __name__ == "__main__":
    validate_grid_and_random_cli()

