"""
Visualization tools for an optimization campaign.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns

from windopt import PROJECT_ROOT

sns.set_theme(style="whitegrid")
sns.set_context("talk")


BLUE = "#2E86C1"
DARK_BLUE = "#1B4F72"
GREY = "#808080"
RED = "#D63D3D"

# Power output from a grid layout of 4 turbines,
# useful baseline for comparison
FOUR_TURBINE_GRID_POWER = 11048751.666666666 / 1e6

def plot_power_trajectory(
        trials: pd.DataFrame,
        title: str = 'Power Trajectory',
        multi_fidelity: bool = False,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        ):
    # Filter out generation_method = "Manual"
    n_manual = len(trials.loc[trials["generation_method"] == "Manual"])
    trials = trials.loc[trials["generation_method"] != "Manual"]
    trials["trial_index"] = trials.index - n_manual

    # Change power to MW
    trials["power"] = trials["power"] / 1e6

    # Plot the power as a function of trial_index
    fig = plt.figure(figsize=(10, 7))

    if not multi_fidelity:
        trials['batch_index'] = trials['trial_index'] // 4
        trials['batch_max'] = trials.groupby('batch_index')['power'].transform('max')
        trials['best_so_far'] = trials['batch_max'].cummax()
        # Main scatterplot 
        ax = sns.scatterplot(
            x="batch_index",
            y="power",
            data=trials,
            color=BLUE,
            label="Individual Trials",
            zorder=2
        )
        sns.lineplot(
            x="batch_index",
            y="best_so_far",
            data=trials,
            color=DARK_BLUE,
            linewidth=2,
            label="Best Result"
        )

    else:
        trials['batch_index'] = trials['trial_index'] // 54
        les_trials = trials.loc[trials["fidelity"] == "les"]
        les_trials['batch_max'] = les_trials.groupby('batch_index')['power'].transform('max')
        les_trials['best_so_far'] = les_trials['batch_max'].cummax()
        # scatterplot on the LES trials
        ax = sns.scatterplot(
            x="batch_index",
            y="power",
            data=les_trials,
            color=BLUE,
            label='LES Evaluations'
        )
        # Add running maximum line
        sns.lineplot(
            x="batch_index",
            y="best_so_far",
            data=les_trials,
            color=DARK_BLUE,
            linewidth=2,
            label="Best Result"
        )
        gch_trials = trials.loc[trials["fidelity"] == "gch"]
        # scatterplot on the GCH trials
        ax = sns.scatterplot(
            x="batch_index",
            y="power",
            data=gch_trials,
            color=GREY,
            alpha=0.6,
            s=20,
            label='GCH Evaluations',
            zorder=0 # put it behind the LES evaluations
        )
    ax.set_xlabel("Optimization Batch")
    ax.set_ylabel("Total Power Output (MW)")
    ax.set_title(title)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    # Customize grid
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)  # Place grid behind other elements

    # Add a horizontal line at the power of the grid layout
    ax.axhline(
        FOUR_TURBINE_GRID_POWER,
        color=RED,
        linestyle="-",
        alpha=0.6,
        label="Grid Layout",
        zorder=0
    )
    ax.legend()

    plt.tight_layout()
    fig.savefig(
        campaign_dir / "power_trajectory.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()


if __name__ == "__main__":
    YLIM = (4, 12.5)
    # Single-Fidelity run
    campaign_dir = PROJECT_ROOT / "campaigns/batch_4_mesh_20m"
    title = "Single-Fidelity Power Trajectory (LES Only)"

    trials = pd.read_csv(campaign_dir / "trials.csv")
    plot_power_trajectory(trials, title, ylim=YLIM)

    # Multi-Fidelity run
    campaign_dir = PROJECT_ROOT / "campaigns/mt50"
    title = "Multi-Fidelity Power Trajectory (50 GCH trials per LES Batch)"

    trials = pd.read_csv(campaign_dir / "trials.csv")
    plot_power_trajectory(trials, title, multi_fidelity=True, ylim=YLIM)
