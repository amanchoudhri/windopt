from pathlib import Path

import numpy as np

from windopt.constants import PROJECT_ROOT
from windopt.layout import load_layout_batch
from windopt.winc3d.io import turbine_results
from windopt.winc3d.les import average_farm_power

simulation_dir = Path('/moto/home/ac4972/windopt/simulations')

layouts = load_layout_batch(PROJECT_ROOT / 'data' / 'initial_points' / 'small_arena_les_samples.npz')

powers = np.zeros(len(layouts))

for i in range(len(layouts)):
    datetime_str = '20250108_131234'
    job_dir = simulation_dir / f'initial_les_trial_{i}_{datetime_str}'

    power_output = average_farm_power(turbine_results(job_dir))
    print(f'{i}: {(power_output / 1e6):0.5f} mW')

    powers[i] = power_output

out_dir = Path('/moto/home/ac4972/windopt/data/initial_trials/')
out_path = out_dir / 'small_arena_les_trials.npy'

np.save(out_path, powers)