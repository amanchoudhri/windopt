"""
Queue initial trials for GCH and LES.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from windopt.winc3d.les import start_les
from windopt.gch import gch

if __name__ == "__main__":
    # read in the initial layouts
    project_root = Path(__file__).parent.parent.parent
    initial_points_dir = project_root / "data" / "initial_points"

    # read in the initial layouts
    gch_small: np.ndarray = np.load(initial_points_dir / "small_arena_gch_samples.npy")
    les_small: np.ndarray = np.load(initial_points_dir / "small_arena_les_samples.npy")

    # create an output directory
    output_dir = project_root / "data" / "initial_trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    # create a dataframe in the trials format that Ax expects
    # to store the GCH results
    # power,x1,z1,x2,z2,x3,z3,fidelity
    n_turbines = gch_small.shape[1]
    layout_cols = [f"x{i}" for i in range(n_turbines)] + [f"z{i}" for i in range(n_turbines)]

    gch_trials = []

    for layout in gch_small:
        # run 100 GCH trials
        yaws = np.zeros((1,layout.shape[0]))
        powers = gch(locations=layout, yaws=yaws)
        power = powers.sum()

        # flatten the layout in column-major order
        # so we get x coords followed by z coords
        print(layout.flatten('F'))

        trial = pd.DataFrame([[power] + list(layout.flatten()) + [0]], columns=gch_trials.columns)
        gch_trials = pd.concat([gch_trials, trial], ignore_index=True)

    # run 12 LES trials
    for i, layout in enumerate(les_small):
        job = start_les(
            run_name=f"initial_les_trial_{i}",
            locations=layout,
            debug_mode=True,
        )
        print(job.slurm_job_id)
