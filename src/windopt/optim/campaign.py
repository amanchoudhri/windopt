"""
Configuration for the optimization campaign.
"""

import json

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from ax.service.ax_client import AxClient

from windopt.constants import PROJECT_ROOT

from windopt.optim.ax_client import (
    setup_ax_client, load_ax_client, save_ax_client_state
)
from windopt.optim.batch_counter import BatchCounts
from windopt.optim.constants import CAMPAIGN_CONFIG_FILENAME
from windopt.optim.trial import run_gch_batch, run_les_batch
from windopt.optim.log import logger, configure_logging
from windopt.optim.config import CampaignConfig, Fidelity, TrialGenerationConfig, TrialGenerationStrategy


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
    
    _run_campaign(ax_client, campaign_config, campaign_dir)

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

    _run_campaign(ax_client, campaign_config, campaign_dir)

def _get_campaign_dir(campaign_name: str) -> Path:
    """
    Get the path to a campaign directory.
    """
    return PROJECT_ROOT / "campaigns" / campaign_name

def _run_campaign(
        ax_client: AxClient,
        campaign_config: CampaignConfig,
        campaign_dir: Path
    ):
    """
    Run the optimization campaign with the specified strategy.
    """
    strategy = campaign_config.trial_generation_config.strategy
    
    is_manual_strategy = strategy in [
        TrialGenerationStrategy.GCH_ONLY,
        TrialGenerationStrategy.LES_ONLY,
        TrialGenerationStrategy.MULTI_ALTERNATING
    ]
    if is_manual_strategy:
            _run_manual_fidelity_select_campaign(
                ax_client,
                campaign_config,
                campaign_dir
            )
    elif strategy == TrialGenerationStrategy.MULTI_ADAPTIVE:
        raise NotImplementedError("Adaptive multi-fidelity strategy not yet implemented!")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _run_manual_fidelity_select_campaign(
        ax_client: AxClient,
        campaign_config: CampaignConfig,
        campaign_dir: Path
    ):
    """
    Run a campaign where fidelities are manually selected at each iteration.
    """
    previous_trials = get_previous_trials(ax_client)
    batch_counter = BatchCounts.from_previous_trials(previous_trials)

    trial_config = campaign_config.trial_generation_config

    while _should_continue(batch_counter, trial_config):
        logger.info(f"Running batch {batch_counter.total}")
        
        fidelity = get_next_fidelity(batch_counter, trial_config)
        
        match fidelity:
            case Fidelity.LES:
                run_les_batch(ax_client, campaign_config)
            case Fidelity.GCH:
                run_gch_batch(ax_client, campaign_config)
        
        batch_counter.increment(fidelity)
        save_ax_client_state(ax_client, campaign_dir)

    return ax_client


def get_previous_trials(ax_client: AxClient, filter_manual: bool = True) -> pd.DataFrame:
    """Get the previous trials from the Ax client."""
    df = ax_client.get_trials_data_frame()
    if filter_manual:
        df = df[df['generation_method'] != 'Manual']
    return df

def get_next_fidelity(
    batch_counter: BatchCounts,
    trial_config: TrialGenerationConfig,
) -> Fidelity:
    """
    Determine the next fidelity to evaluate based on strategy and current state.
    Pure function with no side effects.
    """
    match trial_config.strategy:
        case TrialGenerationStrategy.GCH_ONLY:
            return Fidelity.GCH
        case TrialGenerationStrategy.LES_ONLY:
            return Fidelity.LES
        case TrialGenerationStrategy.MULTI_ALTERNATING:
            if trial_config.alternation_pattern is None:
                raise ValueError("Alternation pattern required for MULTI_ALTERNATING")
            return trial_config.alternation_pattern.get_next_fidelity(
                batch_counter.total
            )
        case _:
            raise ValueError(f"Unsupported strategy: {trial_config.strategy}")

def _should_continue(
    batch_counter: BatchCounts,
    config: TrialGenerationConfig,
) -> bool:
    """
    Determine if the campaign should continue based on batch counts and limits.
    """
    def _within_limit(limit: Optional[int], count: int, name: str) -> bool:
        is_within = limit is None or count < limit
        if not is_within:
            logger.info(f"Stopping campaign: Reached {name} limit of {limit}")
        return is_within
    
    return all([
        _within_limit(config.max_batches, batch_counter.total, "total batch"),
        _within_limit(config.max_les_batches, batch_counter.les, "LES batch"),
        _within_limit(config.max_gch_batches, batch_counter.gch, "GCH batch"),
    ])