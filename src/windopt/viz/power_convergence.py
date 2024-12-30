"""
Plot power convergence over time of wind farm simulations.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from windopt.constants import PROJECT_ROOT
from windopt.winc3d.io import turbine_results

DEFAULT_N_TIMESTEPS_PER_FILE = 9000
DEFAULT_SPINUP_TIMESTEPS = 9000

def power_trajectory(
        job_dir: Path,
        n_timesteps_per_file: int,
        spinup_timesteps: int,
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
    return power_trajectory

def validate_power_convergence():
    layouts = ("grid", "random")

    simulation_dir = Path('/moto/edu/users/ac4972/validate_les')

    power_trajectories = []

    for layout in layouts:
        for use_precursor in (True, False):
            precursor_string = "" if use_precursor else "_no_precursor"
            dirname = f"validate_les_{layout}{precursor_string}"
            trajectory_df = power_trajectory(
                simulation_dir / dirname,
                n_timesteps_per_file=300,
                spinup_timesteps=9000,
            )
            trajectory_df["layout"] = layout.capitalize()
            trajectory_df["use_precursor"] = use_precursor
            power_trajectories.append(trajectory_df)

    power_trajectories = pd.concat(power_trajectories)

    # subtract off the first 30 files representing the 30min spinup period
    # where no measurements were recorded
    power_trajectories['time_hr'] = (power_trajectories['filenumber'] - 30) / 60

    # map True to "With Precursor" and False to "Without Precursor"
    power_trajectories['precursor_display'] = power_trajectories['use_precursor'].map(
        lambda x: "With Precursor" if x else "Without Precursor"
    )

    # facet by precursor usage
    g = sns.FacetGrid(
        data=power_trajectories,
        col="precursor_display",
        row="layout",
        hue="layout"
        )
    g.map(sns.scatterplot,
        "time_hr", "Power",
        alpha=0.4,
        label='Sampled Instantaneous Power Output'
        )
    g.map(sns.lineplot,
        "time_hr",
        "Power_ave",
        alpha=0.8,
        label='Cumulative Average Power Output'
        )

    g.set_xlabels("Time (hr)")
    g.set_ylabels("Power (MW)")
    g.set_titles(
        "{row_name} Layout, {col_name}"
        )

    g.figure.suptitle('Simulated Power Output Throughout LES')
    plt.tight_layout()

    # save the plot
    plt.savefig(PROJECT_ROOT / "power_convergence.png", dpi=300)


if __name__ == "__main__":
    validate_power_convergence()

