from typing import Any

import numpy as np

from ax.core.types import TParameterization

from windopt.layout import Layout
from windopt.optim.config import CampaignConfig


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
            'value_type': 'float',
            'bounds': [0.0, arena_dims[0]],
        })
        params.append({
            'name': f'z{i}',
            'type': 'range',
            'value_type': 'float',
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


def layout_from_ax_params(campaign_config: CampaignConfig, params: dict[str, Any]) -> Layout:
    """
    Create a Layout object from an Ax parameter dict.
    """
    locations = locations_from_ax_params(params, campaign_config.n_turbines)
    return Layout(
        locations,
        campaign_config.arena_dims,
    )


def layout_to_params(layout: Layout) -> dict[str, Any]:
    """
    Convert a Layout object to a dictionary of parameters,
    in the format created by create_turbine_parameters.
    """
    params = {}
    for i in range(layout.n_turbines):
        params[f'x{i+1}'] = layout.coords[i, 0]
        params[f'z{i+1}'] = layout.coords[i, 1]
    return params
