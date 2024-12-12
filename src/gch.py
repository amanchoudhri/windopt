"""
Run a Gauss-Curl Hybrid wake model with FLORIS.
"""

import numpy as np

from floris import FlorisModel

def gch(locations: np.ndarray, orientations: np.ndarray):
    """
    Run the Gauss-Curl Hybrid wake model with FLORIS.

    Args:
        locations: (N, 2) array of (x, z) locations
        orientations: (N, ) array of yaw angles

    Returns:
        powers: (N, ) array of turbine powers in Watts
    """

    model = FlorisModel('../config/gch_base.yaml')

    model.set(
        layout_x=locations[:, 0],
        layout_y=locations[:, 1],
        yaw_angles=orientations,
    )

    model.run()

    return model.get_turbine_powers()
