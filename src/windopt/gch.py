"""
Run a Gauss-Curl Hybrid wake model with FLORIS.
"""

from importlib import resources

import numpy as np
import numpy.typing as npt
from floris import FlorisModel

from windopt.layout import Layout


def gch(layout: Layout) -> npt.NDArray[np.float64]:
    """
    Run the Gauss-Curl Hybrid wake model with FLORIS.

    Args:
        layout: Layout object

    Returns:
        powers: (N, ) array of turbine powers in Watts
    """

    cfg_ptr = resources.files('windopt').joinpath('config/gch_base.yaml')
    with resources.as_file(cfg_ptr) as base_cfg_path:
        model = FlorisModel(base_cfg_path)

    model.set(
        layout_x=layout.arena_coords[:, 0],
        layout_y=layout.arena_coords[:, 1]
    )

    model.run()

    return model.get_turbine_powers()
