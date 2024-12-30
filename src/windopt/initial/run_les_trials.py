"""
Queue initial trials for LES.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from windopt.constants import PROJECT_ROOT
from windopt.gch import gch
from windopt.winc3d.les import start_les

if __name__ == "__main__":
    # read in the initial layouts
    initial_points_dir = Path(PROJECT_ROOT) / "data" / "initial_points"

    # read in the initial layouts
    les_small: np.ndarray = np.load(initial_points_dir / "small_arena_les_samples.npy")

    # create an output directory
    output_dir = PROJECT_ROOT / "data" / "initial_trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    # run 12 LES trials, using the small arena precursor planes
    precursor_dir = PROJECT_ROOT / "simulations" / "small_arena_20m" / "planes"
    n_timesteps = 6000

    for i, layout in enumerate(les_small):
        job = start_les(
            run_name=f"initial_les_trial_{i}",
            inflow_directory=precursor_dir,
            inflow_n_timesteps=n_timesteps,
            locations=layout,
        )
        print(f"Queued LES, SLURM job id: {job.slurm_job_id}")
