"""
Visualization tools for an optimization campaign.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns

from windopt.constants import PROJECT_ROOT
from windopt.optim.config import CampaignConfig

sns.set_theme(style="whitegrid")
sns.set_context("talk")


BLUE = "#2E86C1"
DARK_BLUE = "#1B4F72"
GREY = "#808080"
RED = "#D63D3D"

# Power output from a grid layout of 4 turbines,
# useful baseline for comparison
FOUR_TURBINE_GRID_POWER = 7311492.777777778 / 1e6

def _compute_batch_index(trials: pd.DataFrame, campaign_config: CampaignConfig) -> pd.DataFrame:
    """
    Process trials into batches based on their sequence and fidelity.
    
    Args:
        trials: DataFrame containing trials
        batch_sizes: Dictionary mapping fidelity to batch size
    
    Returns:
        DataFrame with added batch_index column and power in MW
    """
    batch_sizes = {
        "les": campaign_config.trial_generation_config.les_batch_size,
        "gch": campaign_config.trial_generation_config.gch_batch_size
    }
    
    # Detect changes in fidelity to group consecutive trials of the same fidelity
    trials['fidelity_group'] = trials["fidelity"].ne(trials["fidelity"].shift()).cumsum()

    # Create batch indices
    trials['group_batch_index'] = (
        trials.groupby('fidelity_group')
        .cumcount()
        .floordiv(trials["fidelity"].map(lambda f: batch_sizes[f]))
    )
    
    batches_per_group = 1 + trials.groupby("fidelity_group")["group_batch_index"].max()

    # Calculate the cumulative number of batches in each group
    cumulative_batches = batches_per_group.cumsum().shift(1, fill_value=0)
    
    offset = cumulative_batches[trials["fidelity_group"]].reset_index(drop=True)

    # Add batch indices to trials
    trials["batch_index"] = trials["group_batch_index"] + offset
    
    return trials

def _preprocess_trials(trials: pd.DataFrame, campaign_config: CampaignConfig) -> pd.DataFrame:
    # Filter out generation_method = "Manual"
    n_manual = len(trials.loc[trials["generation_method"] == "Manual"])
    trials = trials.loc[trials["generation_method"] != "Manual"]
    trials["trial_index"] = trials.index - n_manual
    
    # Reset index
    trials = trials.reset_index(drop=True)

    # Change power to MW
    trials["power"] = trials["power"] / 1e6
    
    # Compute batch index
    trials = _compute_batch_index(trials, campaign_config)
    
    return trials

def plot_power_trajectory(
        trials: pd.DataFrame,
        campaign_config: CampaignConfig,
        title: Optional[str] = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        save_path: Optional[Path] = None,
        ) -> plt.Figure:
    """
    Plot the power trajectory for an optimization campaign.
    
    Shows all trials colored by fidelity, with a running maximum line
    tracking the best LES result.
    """
    # Process trials
    trials = _preprocess_trials(trials, campaign_config)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot trials by fidelity with consistent styling
    sns.scatterplot(
        data=trials,
        x="batch_index",
        y="power",
        hue="fidelity",
        style="fidelity",
        hue_order=["les", "gch"],
        alpha=0.6,
        s=50,
        zorder=1,
        ax=ax
    )
    
    # If there are LES trials, add a running maximum line for LES trials
    if "les" in trials["fidelity"].unique():
        les_trials = trials.loc[trials["fidelity"] == "les"]
        sns.lineplot(
            data=les_trials,
            x="batch_index",
            y=les_trials["power"].cummax(),
            color=DARK_BLUE,
            linewidth=2,
            label="Best LES Result",
            ax=ax
        )
    
    # Add baseline reference
    ax.axhline(
        FOUR_TURBINE_GRID_POWER,
        color=RED,
        linestyle="--",
        alpha=0.6,
        label="Grid Layout",
        zorder=0
    )
    
    # Customize plot
    ax.set_xlabel("Optimization Batch")
    ax.set_ylabel("Total Power Output (MW)")
    ax.set_title(title or "Power Trajectory")
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)
    
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig

def parse_args():
    """Parse command line arguments for campaign visualization."""
    parser = argparse.ArgumentParser(
        description="Generate power trajectory plots for optimization campaigns."
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        help="Path to the campaign directory containing trials.csv and campaign_config.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the output plot"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title for the plot",
        required=False
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=[4, 16],
        help="Y-axis limits as min max (default: 4 16)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    trials = pd.read_csv(args.campaign_dir / "trials.csv")
    campaign_config = CampaignConfig.from_json(args.campaign_dir / "campaign_config.json")
    title = args.title if args.title else campaign_config.name
    plot_power_trajectory(
        trials,
        campaign_config,
        title=title,
        ylim=args.ylim,
        save_path=args.output
    )

if __name__ == "__main__":
    main()