import logging

from datetime import datetime
from typing import Any

import numpy as np

from ax.core.types import TParameterization

from windopt.constants import PROJECT_ROOT
from windopt.layout import Layout
from windopt.optim.config import CampaignConfig


def setup_experiment_logging(experiment_name: str, debug_mode: bool) -> logging.Logger:
    """
    Set up logging for the experiment.
    """
    # Set up logging
    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler()

    for handler in [fh, sh]:
        handler.setLevel(logging.INFO if not debug_mode else logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logging.basicConfig(
        level=logging.INFO if not debug_mode else logging.DEBUG,
        handlers=[fh, sh]
    )
    logger = logging.getLogger(__name__)

    # Add file handler to ax logger
    from ax.utils.common.logger import ROOT_LOGGER
    ROOT_LOGGER.addHandler(fh)

    return logger

def create_turbine_parameters(
        N_turbines: int,
        arena_dims: tuple[float, float],
        optimize_angles: bool = False
        ) -> list[dict[str, Any]]:
    """
    Create a list of parameters to pass to ax_client.create_experiment.
    """
    params = []
    for i in range(1, N_turbines + 1):
        params.append({
            'name': f'x{i}',
            'type': 'range',
            'bounds': [0.0, arena_dims[0]],
        })
        params.append({
            'name': f'z{i}',
            'type': 'range',
            'bounds': [0.0, arena_dims[1]],
        })
    if optimize_angles:
        raise NotImplementedError("Optimizing angles is not supported yet!")
    return params


def locations_from_ax_params(params: TParameterization, N_turbines: int) -> np.ndarray:
    """
    Extract turbine locations into a numpy array from an Ax parameter dict.
    """
    locations = np.array([
        [params[f'x{i}'] for i in range(1, N_turbines + 1)],
        [params[f'z{i}'] for i in range(1, N_turbines + 1)]
    ]).T
    return locations


def layout_from_ax_params(campaign_config: CampaignConfig, params: TParameterization) -> Layout:
    """
    Create a Layout object from an Ax parameter dict.
    """
    locations = locations_from_ax_params(params, campaign_config.n_turbines)
    return Layout(
        locations,
        'arena',
        campaign_config.arena_dims,
        campaign_config.box_dims
    )


def layout_to_params(layout: Layout) -> dict[str, Any]:
    """
    Convert a Layout object to a dictionary of parameters,
    in the format created by create_turbine_parameters.
    """
    params = {}
    for i in range(layout.n_turbines):
        params[f'x{i+1}'] = layout.arena_coords[i, 0]
        params[f'z{i+1}'] = layout.arena_coords[i, 1]
    return params
