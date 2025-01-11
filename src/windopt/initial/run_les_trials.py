"""
Queue initial trials for LES.
"""

import argparse
from pathlib import Path

from windopt.constants import (
    N_STEPS_PRODUCTION, PROJECT_ROOT, INFLOW_20M, INFLOW_20M_N_TIMESTEPS, 
    SMALL_BOX_DIMS, N_STEPS_DEBUG, VIZ_INTERVAL_DEFAULT, VIZ_INTERVAL_FREQUENT, D, HUB_HEIGHT
)
from windopt.winc3d.les import start_les
from windopt.winc3d.config import (
    LESConfig, FlowConfig, NumericalConfig, 
    InflowConfig, OutputConfig, TurbineConfig
)
from windopt.layout import load_layout_batch


def main(layouts_file: Path, debug_mode: bool = False, frequent_viz: bool = False):
    layouts = load_layout_batch(layouts_file)

    # create an output directory
    output_dir = PROJECT_ROOT / "data" / "initial_trials"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure simulation parameters
    n_steps = N_STEPS_DEBUG if debug_mode else N_STEPS_PRODUCTION
    viz_interval = VIZ_INTERVAL_FREQUENT if frequent_viz else VIZ_INTERVAL_DEFAULT 

    # run LES trials, using the small arena precursor planes
    for i, layout in enumerate(layouts):
        config = LESConfig(
            box_dims=SMALL_BOX_DIMS,
            flow=FlowConfig(flow_type="precursor"),
            numerical=NumericalConfig(n_steps=n_steps),
            turbines=TurbineConfig(
                layout=layout,
                diameter=D,
                hub_height=HUB_HEIGHT
            ),
            inflow=InflowConfig(
                directory=INFLOW_20M,
                n_timesteps=INFLOW_20M_N_TIMESTEPS
            ),
            output=OutputConfig(viz_interval=viz_interval)
        )

        job = start_les(
            run_name=f"initial_les_trial_{i}",
            config=config
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
