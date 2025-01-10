"""
Configuration for the optimization campaign.
"""

import json
import logging
import time

from dataclasses import asdict
from pathlib import Path

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.modelbridge.transforms.task_encode import TaskChoiceToIntTaskChoice
from ax.modelbridge.transforms.unit_x import UnitX
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from ax.storage.json_store.save import save_experiment
import numpy as np

from windopt.constants import PROJECT_ROOT
from windopt.layout import load_layout_batch
from windopt.optim.trial import complete_noiseless_trial, process_completed_les_jobs, run_gch_batch, start_les_batch
from windopt.optim.utils import create_turbine_parameters, layout_to_params, setup_experiment_logging
from windopt.optim.config import CampaignConfig, TrialGenerationStrategy

LES_POLLING_INTERVAL = 30

multi_type_transforms = [
    TaskChoiceToIntTaskChoice,  # Since we're using a string valued task parameter.
    UnitX, 
    StandardizeY
]


def load_initial_data(ax_client: AxClient, fidelity: str = 'gch'):
    """
    Load initial trial data into the AxClient.
    """
    if fidelity not in ['gch', 'les']:
        raise ValueError(f"Fidelity must be 'gch' or 'les', not {fidelity}")

    layouts = load_layout_batch(
        PROJECT_ROOT / 'data' / 'initial_points' / f'small_arena_{fidelity}_samples.npz')
    powers = np.load(
        PROJECT_ROOT / 'data' / 'initial_trials' / f'small_arena_{fidelity}_trials.npy')

    for (layout, power) in zip(layouts, powers):
        # convert layout to x, z params for Ax
        params = layout_to_params(layout)
        params['fidelity'] = fidelity

        _, trial_index = ax_client.attach_trial(
            parameters = params
        )
        complete_noiseless_trial(ax_client, trial_index, power)

def setup_ax_client(campaign_config: CampaignConfig) -> AxClient:
    """
    Setup the Ax client.
    """
    # Set up search space
    param_names = create_turbine_parameters(campaign_config.n_turbines, campaign_config.arena_dims)

    # Add the task/fidelity parameter
    param_names.append({
        "name": "fidelity",
        "type": "choice",
        "values": ["les", "gch"],
        "is_task": True,
        "target_value": "les"
    })

    # Setup a generation strategy without initial Sobol steps since
    # we will preload initial trial data.
    gs = GenerationStrategy([
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            model_kwargs={"transforms": multi_type_transforms},
        )
    ])

    # Create Ax client
    ax_client = AxClient(generation_strategy=gs)
    ax_client.create_experiment(
        name=campaign_config.name,
        parameters=param_names,
        objectives={"power": ObjectiveProperties(minimize=False)},
    )

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

def save_campaign_state(ax_client: AxClient, campaign_dir: Path):
    """
    Save the campaign state to a file.
    """
    exp_to_df(ax_client.experiment).to_csv(str(campaign_dir / "trials.csv"))
    save_experiment(ax_client.experiment, str(campaign_dir / "experiment.json"))

def _run_manual_fidelity_select_campaign(
        ax_client: AxClient,
        campaign_config: CampaignConfig,
        campaign_dir: Path,
        logger: logging.Logger
    ):
    """
    Run a campaign where fidelities are manually selected at each iteration.
    """
    batch_idx = 1
    continue_running = True

    trial_config = campaign_config.trial_generation_config

    while continue_running:
        if trial_config.max_les_batches is not None:
            if batch_idx >= trial_config.max_les_batches:
                continue_running = False
            logger.info(f"Running batch {batch_idx} of {trial_config.max_les_batches}")
        else:
            logger.info(f"Running batch {batch_idx}")

        if trial_config.strategy == TrialGenerationStrategy.MULTI_ALTERNATING:
            # Run the GCH trials
            for i in range(trial_config.gch_batches_per_les_batch):
                logger.info(f"Running GCH batch {i + 1} of {trial_config.gch_batches_per_les_batch}")
                run_gch_batch(ax_client, trial_config.gch_batch_size, campaign_config.n_turbines)

        if trial_config.strategy != TrialGenerationStrategy.GCH_ONLY:
            # Queue up the batch of LES trials
            active_jobs = start_les_batch(
                ax_client,
                campaign_config,
                trial_config.les_batch_size,
                campaign_config.debug_mode,
                logger
            )

            # briefly wait to let SLURM process the submission
            # before watching the jobs
            time.sleep(30)

            # Wait for all jobs in batch to complete and process results
            while active_jobs:
                active_jobs = process_completed_les_jobs(
                    ax_client,
                    active_jobs,
                    logger,
                    campaign_config.debug_mode
                )

                # Save current experiment state to file
                save_campaign_state(ax_client, campaign_dir)

                # Wait before checking again
                time.sleep(LES_POLLING_INTERVAL)

        batch_idx += 1

    return ax_client

def run_campaign(campaign_config: CampaignConfig):
    """
    Run the campaign.
    """
    logger = setup_experiment_logging(campaign_config.name, campaign_config.debug_mode)
    ax_client = setup_ax_client(campaign_config)

    # Setup campaign directory
    campaign_dir = PROJECT_ROOT / "campaigns" / campaign_config.name
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # Save the campaign config to the campaign directory
    with open(campaign_dir / "campaign_config.json", "w") as f:
        json.dump(asdict(campaign_config), f)

    strategy = campaign_config.trial_generation_config.strategy
    if strategy in [
        TrialGenerationStrategy.LES_ONLY,
        TrialGenerationStrategy.MULTI_ALTERNATING
        ]:
        _run_manual_fidelity_select_campaign(
            ax_client, campaign_config, campaign_dir, logger
        )
    elif strategy == TrialGenerationStrategy.MULTI_ADAPTIVE:
        raise NotImplementedError("Adaptive multi-fidelity strategy not yet implemented!")
    elif strategy == TrialGenerationStrategy.GCH_ONLY:
        raise NotImplementedError("GCH only strategy not yet implemented!")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")