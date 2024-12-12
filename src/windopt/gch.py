"""
Run a Gauss-Curl Hybrid wake model with FLORIS.
"""

from importlib import resources

import numpy as np

from floris import FlorisModel

import windopt

def gch(locations: np.ndarray, yaws: np.ndarray):
    """
    Run the Gauss-Curl Hybrid wake model with FLORIS.

    Args:
        locations: (N, 2) array of (x, z) locations
        orientations: (N, ) array of yaw angles

    Returns:
        powers: (N, ) array of turbine powers in Watts
    """

    cfg_ptr = resources.files('windopt').joinpath('config/gch_base.yaml')
    with resources.as_file(cfg_ptr) as base_cfg_path:
        model = FlorisModel(base_cfg_path)

    model.set(
        layout_x=locations[:, 0],
        layout_y=locations[:, 1],
        yaw_angles=yaws,
    )

    model.run()

    return model.get_turbine_powers()