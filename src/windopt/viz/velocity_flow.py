"""
Plot the instantaneous and mean velocity fields computed during a given simulation.
"""

import argparse

from math import ceil
from pathlib import Path

from windopt.viz.vector_field import inst_and_mean_velocity_field
from windopt.winc3d.config import LESConfig

def get_args():
    p = argparse.ArgumentParser(
        description="Compare instantaneous and mean velocity fields"
        )
    p.add_argument(
        "run_dir",
        type=Path,
        help="Path to the simulation run directory"
    )
    p.add_argument(
        "time_after_spinup",
        type=float,
        help="Time (in hours) after spinup at which to display the instantaneous flow field"
    )
    p.add_argument(
        "--save_path",
        type=Path,
        help="Path to save the plot"
    )
    return p.parse_args()

def make_plot(run_dir: Path, time_after_spinup: float, save_path: Path):
    # load in the config
    config = LESConfig.from_json(run_dir / 'les_config.json')
    
    # calculate the nearest timestep based on the time after spinup
    seconds_after_spinup = time_after_spinup * 3600
    timestep = int((
        seconds_after_spinup - config.output.spinup_time) // config.numerical.dt
        )
    nearest_filenumber = ceil(timestep / config.output.viz_interval)
    print(f'Loading instantaneous velocity field from file ux{nearest_filenumber:04d}')
    inst_and_mean_velocity_field(run_dir, nearest_filenumber, save_path)

def main():
    args = get_args()
    if args.save_path is None:
        save_path = args.run_dir / f'ux_and_umean_{args.time_after_spinup}h.png'
    else:
        save_path = args.save_path
    make_plot(args.run_dir, args.time_after_spinup, save_path)

if __name__ == '__main__':
    main()