from pathlib import Path

import numpy as np

from windopt.winc3d.slurm import LESJob
from windopt.winc3d.les import process_results

JOB_IDS = [
    19680306, 19680307, 19680308, 19680309, 19680310, 19680311, 19680312,
    19680313, 19680314, 19680315, 19680316, 19680317, 19680318
    ]

simulation_dir = Path('/moto/home/ac4972/windopt/simulations')

powers = np.zeros(len(JOB_IDS))

for i in range(len(JOB_IDS)):
    datetime_str = '20241212_032210'
    job_dir = simulation_dir / f'initial_les_trial_{i}_{datetime_str}' / 'out'

    job = LESJob(JOB_IDS[i], job_dir)
    power_output = process_results(job)
    print(f'{i}: {(power_output / 1e6):0.5f} mW')

    powers[i] = power_output

out_dir = Path('/moto/home/ac4972/windopt/data/initial_trials/')
out_path = out_dir / 'small_arena_les_trials.npy'

np.save(out_path, powers)

