from pathlib import Path

import numpy as np
import pandas as pd
from windopt.winc3d.les import start_les
from windopt.gch import gch

if __name__ == "__main__":
    # read in the initial layouts
    project_root = Path(__file__).parent.parent.parent.parent
    initial_points_dir = project_root / "data" / "initial_points"

    # read in the initial layouts
    gch_small: np.ndarray = np.load(initial_points_dir / "small_arena_gch_samples.npy")

    n_turbines = gch_small.shape[1]

    powers = np.zeros(gch_small.shape[0])

    for i, layout in enumerate(gch_small):
        # run 100 GCH trials
        yaws = np.zeros((1,layout.shape[0]))
        power = gch(locations=layout, yaws=yaws).sum()
        powers[i] = power
        print(f'{i}: {(power / 1e6):0.5f} mW')

    out_dir = Path('/moto/home/ac4972/windopt/data/initial_trials/')
    out_path = out_dir / 'small_arena_gch_trials.npy'

    np.save(out_path, powers)
