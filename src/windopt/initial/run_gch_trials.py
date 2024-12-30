from pathlib import Path

import numpy as np

from windopt import PROJECT_ROOT
from windopt.gch import gch

if __name__ == "__main__":
    # read in the initial layouts
    initial_points_dir = PROJECT_ROOT / "data" / "initial_points"

    # read in the initial layouts
    gch_small: np.ndarray = np.load(initial_points_dir / "small_arena_gch_samples.npy")

    n_turbines = gch_small.shape[1]

    powers = np.zeros(gch_small.shape[0])

    for i, layout in enumerate(gch_small):
        # run 100 GCH trials
        power = gch(locations=layout).sum()
        powers[i] = power
        print(f'{i}: {(power / 1e6):0.5f} mW')

    out_dir = Path('/moto/home/ac4972/windopt/data/initial_trials/')
    out_path = out_dir / 'small_arena_gch_trials.npy'

    np.save(out_path, powers)
