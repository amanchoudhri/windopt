"""
Queue initial trials for LES.
"""

import argparse
from pathlib import Path

from windopt.constants import PROJECT_ROOT, INFLOW_20M, INFLOW_20M_N_TIMESTEPS
from windopt.winc3d.les import start_les
from windopt.layout import load_layout_batch


def main(layouts_file: Path, debug_mode: bool = False, frequent_viz: bool = False):
    layouts = load_layout_batch(layouts_file)

    # create an output directory
    output_dir = PROJECT_ROOT / "data" / "initial_trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    # run 12 LES trials, using the small arena precursor planes
    for i, layout in enumerate(layouts):
        job = start_les(
            run_name=f"initial_les_trial_{i}",
            layout=layout,
            inflow_directory=INFLOW_20M,
            inflow_n_timesteps=INFLOW_20M_N_TIMESTEPS,
            debug_mode=debug_mode,
            frequent_viz=frequent_viz
        )
        print(f"Queued LES, SLURM job id: {job.slurm_job_id}")


if __name__ == "__main__":
    initial_points_dir = Path(PROJECT_ROOT) / "data" / "initial_points"
    layouts_file = initial_points_dir / "small_arena_les_samples.npz"

    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts_file", type=Path, default=layouts_file)
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--frequent_viz", action="store_true")
    args = parser.parse_args()

    main(args.layouts_file, debug_mode=args.debug_mode, frequent_viz=args.frequent_viz)
