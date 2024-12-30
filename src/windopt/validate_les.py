"""
Queue a LES job with very frequent visualization and statistics outputs.

Explores the convergence of the LES to the steady state.
"""

from pathlib import Path

import numpy as np

from windopt import PROJECT_ROOT
from windopt.constants import D, HUB_HEIGHT, SMALL_BOX_DIMS
from windopt.winc3d.slurm import LESJob, submit_job, SlurmConfig
from windopt.winc3d.les import start_les


def load_turbine_locations(layout: str):
    """
    Load the turbine locations for a given layout.
    """
    if layout not in ["random", "grid"]:
        raise ValueError(f"Invalid layout: {layout}. Must be 'random' or 'grid'.")

    generated_locations = np.load(
        PROJECT_ROOT / "data" / "initial_points" / "small_arena_les_samples.npy"
        )

    random_layout = generated_locations[0]
    grid_layout = generated_locations[-1]

    return random_layout if layout == "random" else grid_layout


def submit_frequent_viz_job(layout: str, use_precursor: bool = False) -> LESJob:
    """
    Submit a job with very frequent visualization and statistics outputs.
    """
    locations = load_turbine_locations(layout)

    INFLOW_DIR = PROJECT_ROOT / "simulations" / "small_arena_20m" / "planes"
    N_TIMESTEPS_PER_FILE = 6000

    run_name = f"validate_les_{layout}"
    if not use_precursor:
        run_name += "_no_precursor"

    job = start_les(
        run_name=run_name,
        locations=locations,
        inflow_directory=INFLOW_DIR if use_precursor else None,
        inflow_n_timesteps=N_TIMESTEPS_PER_FILE if use_precursor else None,
        debug_mode=False,
        frequent_viz=True,
    )

    return job

if __name__ == "__main__":
    for layout in ["random", "grid"]:
        for use_precursor in [True, False]:
            # temporary, runs without precursor are already going.
            if not use_precursor: break
            job = submit_frequent_viz_job(layout, use_precursor)

