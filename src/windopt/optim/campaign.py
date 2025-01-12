"""
Configuration for the optimization campaign.
"""

import json
import logging
import time

from dataclasses import asdict
from pathlib import Path

from ax.service.ax_client import AxClient

from windopt.constants import PROJECT_ROOT

from windopt.optim.ax_client import (
    setup_ax_client, load_ax_client, save_ax_client_state
)
from windopt.optim.constants import CAMPAIGN_CONFIG_FILENAME
from windopt.optim.trial import process_completed_les_jobs, run_gch_batch, start_les_batch
from windopt.optim.logging import logger, configure_logging
from windopt.optim.config import CampaignConfig, TrialGenerationStrategy

LES_POLLING_INTERVAL = 30


def new_campaign(campaign_config: CampaignConfig):
    """
    Run a new optimization campaign.
    """
    configure_logging(
        campaign_config.name,
        PROJECT_ROOT / "log",
        campaign_config.debug_mode
    )
    logger.info(f"Starting new campaign {campaign_config.name}")
    campaign_dir = _get_campaign_dir(campaign_config.name)
    campaign_dir.mkdir(parents=True, exist_ok=True)  # Only create directory for new campaigns
    
    ax_client = setup_ax_client(campaign_config)
    
    # Save initial campaign state
    with open(campaign_dir / CAMPAIGN_CONFIG_FILENAME, "w") as f:
        json.dump(asdict(campaign_config), f)
    save_ax_client_state(ax_client, campaign_dir)
    
    _run_campaign(ax_client, campaign_config, campaign_dir, logger)

def restart_campaign(campaign_name: str):
    """
    Restart an existing optimization campaign from its last saved state.
    """
    campaign_dir = _get_campaign_dir(campaign_name)
    
    # First load just the config to get debug_mode
    try:
        campaign_config = CampaignConfig.from_json(campaign_dir / CAMPAIGN_CONFIG_FILENAME)
    except Exception as e:
        raise ValueError(f"Failed to load campaign config: {e}")
    
    # Configure logging before any other operations
    configure_logging(
        campaign_name,
        PROJECT_ROOT / "log",
        campaign_config.debug_mode
    )
    logger.info(f"Restarting campaign {campaign_name}")
    
    # Now load the full campaign state
    try:
        ax_client = load_ax_client(campaign_dir)
    except Exception as e:
        raise ValueError(f"Failed to load campaign state: {e}")

    current_batch = _get_current_batch_index(ax_client, campaign_config)
    logger.info(f"Continuing from batch {current_batch}")
    
    _run_campaign(
        ax_client, campaign_config, campaign_dir,
        logger, start_batch=current_batch
        )

def _get_campaign_dir(campaign_name: str) -> Path:
    """
    Get the path to a campaign directory.
    """
    return PROJECT_ROOT / "campaigns" / campaign_name

def _run_campaign(
    ax_client: AxClient,
    campaign_config: CampaignConfig,
    campaign_dir: Path,
    logger: logging.Logger,
    start_batch: int = 1
) -> AxClient:
    """
    Run the optimization campaign with the specified strategy.
    """
    strategy = campaign_config.trial_generation_config.strategy
    match strategy:
        case TrialGenerationStrategy.LES_ONLY | TrialGenerationStrategy.MULTI_ALTERNATING:
            return _run_manual_fidelity_select_campaign(
                ax_client,
                campaign_config,
                campaign_dir,
                logger,
                start_batch=start_batch
            )
        case TrialGenerationStrategy.MULTI_ADAPTIVE:
            raise NotImplementedError("Adaptive multi-fidelity strategy not yet implemented!")
        case TrialGenerationStrategy.GCH_ONLY:
            raise NotImplementedError("GCH only strategy not yet implemented!")
        case _:
            raise ValueError(f"Unknown strategy: {strategy}")


def _run_manual_fidelity_select_campaign(
        ax_client: AxClient,
        campaign_config: CampaignConfig,
        campaign_dir: Path,
        logger: logging.Logger,
        start_batch: int = 1
    ):
    """
    Run a campaign where fidelities are manually selected at each iteration.
    """
    batch_idx = start_batch
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
                run_gch_batch(ax_client, campaign_config, trial_config.gch_batch_size)

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
                    logger
                )

                # Save current experiment state to file
                save_ax_client_state(ax_client, campaign_dir)

                # Wait before checking again
                time.sleep(LES_POLLING_INTERVAL)

        batch_idx += 1

    return ax_client

def _get_current_batch_index(ax_client: AxClient, campaign_config: CampaignConfig) -> int:
    """
    Determine the current batch index from completed trials.
        
    Returns:
        The next batch index to run (1-based indexing)
    """
    trials_df = ax_client.get_trials_data_frame()
    if trials_df.empty:
        return 1
    
    completed_df = trials_df[trials_df['trial_status'] == 'COMPLETED']
    if completed_df.empty:
        return 1

    # Exclude manual trials, which are standard initial trial data
    opt_df = completed_df[completed_df['generation_method'] != 'Manual']

    strategy = campaign_config.trial_generation_config.strategy
    trial_config = campaign_config.trial_generation_config
    
    n_gch_trials = len(opt_df[opt_df['fidelity'] == 'gch'])
    n_les_trials = len(opt_df[opt_df['fidelity'] == 'les'])

    match strategy:
        case TrialGenerationStrategy.GCH_ONLY:
            n_complete_batches = n_gch_trials // trial_config.gch_batch_size
        case TrialGenerationStrategy.LES_ONLY:
            n_complete_batches = n_les_trials // trial_config.les_batch_size
        case TrialGenerationStrategy.MULTI_ALTERNATING:
            n_gch_batches = n_gch_trials // trial_config.gch_batch_size
            n_complete_batches = n_gch_batches // trial_config.gch_batches_per_les_batch
        case TrialGenerationStrategy.MULTI_ADAPTIVE:
            raise NotImplementedError("Adaptive multi-fidelity strategy not yet implemented!")
        case _:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return n_complete_batches + 1