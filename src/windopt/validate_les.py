"""
Queue a LES job with very frequent visualization and statistics outputs.

Explores the convergence of the LES to the steady state.
"""

from pathlib import Path

import numpy as np

from windopt.constants import PROJECT_ROOT, INFLOW_20M, INFLOW_20M_N_TIMESTEPS
from windopt.winc3d.slurm import LESJob
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
        job = start_les(
            run_name=f"validate_les_{layout_name}",
            layout=layout,
            inflow_directory=INFLOW_20M,
            inflow_n_timesteps=INFLOW_20M_N_TIMESTEPS,
            debug_mode=False,
            frequent_viz=True,
        )
