"""
Ax client setup and configuration.
Handles the interface between our domain and the Ax optimization library.
"""
import json
from pathlib import Path

import numpy as np
import torch

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.modelbridge.transforms.task_encode import TaskChoiceToIntTaskChoice
from ax.modelbridge.transforms.unit_x import UnitX
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.service.utils.report_utils import exp_to_df
from ax.storage.json_store.save import save_experiment

from windopt.constants import PROJECT_ROOT

from windopt.layout import load_layout_batch
from windopt.optim.config import CampaignConfig, TrialGenerationStrategy
from windopt.optim.constants import (
    AX_CLIENT_FILENAME, TRIALS_FILENAME, EXPERIMENT_FILENAME
)
from windopt.optim.trial import complete_noiseless_trial
from windopt.optim.utils import create_turbine_parameters, layout_to_params
from windopt.optim.log import logger

# Model configuration
MULTI_TYPE_TRANSFORMS = [
    TaskChoiceToIntTaskChoice,  # Since we're using a string valued task parameter
    UnitX, 
    StandardizeY
]

def _generation_strategy(
        campaign_config: CampaignConfig,
        use_cuda: bool = True
        ) -> GenerationStrategy:
    """
    Generate a generation strategy for the Ax client.
    """
    # Setup a generation strategy without initial Sobol steps since
    # we will preload initial trial data
    model_kwargs = {"transforms": MULTI_TYPE_TRANSFORMS}
    
    logger.info("Setting up Ax client")
    
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model_kwargs["torch_device"] = device
            logger.info(f"CUDA available, using device {device} for GP model")
        else:
            logger.warning("CUDA not available, using CPU for GP model")

    gs = GenerationStrategy([
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            model_kwargs=model_kwargs,
        )
    ])
    return gs

def setup_ax_client(
        campaign_config: CampaignConfig,
        use_cuda: bool = True
        ) -> AxClient:
    """
    Setup and configure the Ax client for our optimization.
    
    Args:
        campaign_config: Configuration for the optimization campaign
        
    Returns:
        Configured AxClient ready for optimization
    """
    # Set up search space
    param_names = create_turbine_parameters(
        campaign_config.n_turbines,
        campaign_config.arena_dims
        )

    # Add the task/fidelity parameter
    param_names.append({
        "name": "fidelity",
        "type": "choice",
        "values": ["les", "gch"],
        "is_task": True,
        "target_value": "les"
    })


    # Create Ax client
    gs = _generation_strategy(campaign_config, use_cuda=use_cuda)
    ax_client = AxClient(generation_strategy=gs)
    ax_client.create_experiment(
        name=campaign_config.name,
        parameters=param_names,
        objectives={"power": ObjectiveProperties(minimize=False)},
    )

    # Load initial data based on strategy
    strategy = campaign_config.trial_generation_config.strategy
    is_multi_fidelity = strategy in [
        TrialGenerationStrategy.MULTI_ALTERNATING,
        TrialGenerationStrategy.MULTI_ADAPTIVE
    ]

    if is_multi_fidelity or strategy == TrialGenerationStrategy.GCH_ONLY:
        load_initial_data(ax_client, fidelity='gch')

    if is_multi_fidelity or strategy == TrialGenerationStrategy.LES_ONLY:
        load_initial_data(ax_client, fidelity='les')

    return ax_client

def load_initial_data(ax_client: AxClient, fidelity: str = 'gch'):
    """
    Load initial trial data into the AxClient.
    
    Args:
        ax_client: Client to load data into
        fidelity: Which fidelity data to load ('gch' or 'les')
    """
    if fidelity not in ['gch', 'les']:
        raise ValueError(f"Fidelity must be 'gch' or 'les', not {fidelity}")

    layouts = load_layout_batch(
        PROJECT_ROOT / 'data' / 'initial_points' / f'small_arena_{fidelity}_samples.npz')
    powers = np.load(
        PROJECT_ROOT / 'data' / 'initial_trials' / f'small_arena_{fidelity}_trials.npy')

    for (layout, power) in zip(layouts, powers):
        params = layout_to_params(layout)
        params['fidelity'] = fidelity

        _, trial_index = ax_client.attach_trial(parameters=params)
        complete_noiseless_trial(ax_client, trial_index, power)

def save_ax_client_state(ax_client: AxClient, campaign_dir: Path):
    """
    Save Ax client state to disk.
    
    Args:
        ax_client: Client to save
        campaign_dir: Directory to save state in
    """
    exp_to_df(ax_client.experiment).to_csv(str(campaign_dir / TRIALS_FILENAME))
    save_experiment(ax_client.experiment, str(campaign_dir / EXPERIMENT_FILENAME))
    ax_client.save_to_json_file(str(campaign_dir / AX_CLIENT_FILENAME)) 

def load_ax_client(
        campaign_dir: Path,
        use_cuda: bool = True
        ) -> AxClient:
    """
    Load an Ax client from saved state.
    
    Args:
        campaign_dir: Directory containing saved client state
        
    Returns:
        Loaded AxClient
    """
    with open(campaign_dir / AX_CLIENT_FILENAME, 'r') as f:
        ax_client_json = json.load(f)
        
    ax_client_json = _update_client_cuda_state(ax_client_json, use_cuda)
    
    return AxClient.from_json_snapshot(ax_client_json)

def _update_client_cuda_state(ax_client_json: dict, use_cuda: bool):
    """
    Update the Ax client state to use the correct device.
    """
    # TODO: hacky. relies on the specific generation strategy defined
    # by _generation_strategy()
    model_kwargs = ax_client_json['generation_strategy']['steps'][0]['model_kwargs']
    if use_cuda and torch.cuda.is_available():
        logger.info(f"CUDA available, using GPU for GP model")
        model_kwargs['torch_device'] = {'__type': 'torch_device', 'value': 'cuda'}
    else:
        logger.info(f"CUDA not available, using CPU for GP model")
        model_kwargs.pop('torch_device', None)
    ax_client_json['generation_strategy']['steps'][0]['model_kwargs'] = model_kwargs
    return ax_client_json
