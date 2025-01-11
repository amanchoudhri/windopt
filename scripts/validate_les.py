"""
Queue a LES job with very frequent visualization and statistics outputs.

Explores the convergence of the LES to the steady state.
"""

from windopt.constants import (
    INFLOW_20M, INFLOW_20M_N_TIMESTEPS, PROJECT_ROOT, SMALL_BOX_DIMS,
    VIZ_INTERVAL_FREQUENT
)
from windopt.winc3d.config import InflowConfig, LESConfig, OutputConfig, TurbineConfig
from windopt.winc3d.les import start_les
from windopt.layout import load_layout_batch, Layout


def load_layout(layout_name: str) -> Layout:
    """
    Load the turbine locations for a given layout.
    """
    if layout_name not in ["random", "grid"]:
        raise ValueError(f"Invalid layout: {layout}. Must be 'random' or 'grid'.")
    
    layouts_file = PROJECT_ROOT / "data" / "initial_points" / "small_arena_les_samples.npz"
    layouts = load_layout_batch(layouts_file)

    random_layout = layouts[0]
    grid_layout = layouts[-1]

    return random_layout if layout_name == "random" else grid_layout


if __name__ == "__main__":
    for layout_name in ["random", "grid"]:
        layout = load_layout(layout_name)
        config = LESConfig(
            box_dims=SMALL_BOX_DIMS,
            turbines=TurbineConfig(layout=layout),
            inflow=InflowConfig(directory=INFLOW_20M, n_timesteps=INFLOW_20M_N_TIMESTEPS),
            output=OutputConfig(viz_interval=VIZ_INTERVAL_FREQUENT)
        )
        job = start_les(
            run_name=f"validate_les_{layout_name}",
            config=config
        )
        print(f"Queued LES, SLURM job id: {job.slurm_job_id}")