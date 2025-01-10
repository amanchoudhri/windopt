from pathlib import Path

import numpy as np

from windopt.constants import PROJECT_ROOT
from windopt.gch import gch
from windopt.layout import load_layout_batch

if __name__ == "__main__":
    initial_points_dir = PROJECT_ROOT / "data" / "initial_points"
    layouts = load_layout_batch(initial_points_dir / "small_arena_gch_samples.npz")

    output_dir = PROJECT_ROOT / "data" / "initial_trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    powers = np.zeros(len(layouts))

    for i, layout in enumerate(layouts):
        power = gch(layout).sum()
        powers[i] = power
        print(f'{i}: {(power / 1e6):0.5f} mW')

    out_dir = Path('/moto/home/ac4972/windopt/data/initial_trials/')
    out_path = out_dir / 'small_arena_gch_trials.npy'

    np.save(out_path, powers)
