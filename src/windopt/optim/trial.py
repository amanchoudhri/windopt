"""
Trial generation and management functions for both LES and GCH evaluations.
"""

import time
import warnings

from ax.core.observation import ObservationFeatures
from ax.service.ax_client import AxClient

from windopt.constants import (
    INFLOW_20M, INFLOW_20M_N_TIMESTEPS,
    N_STEPS_DEBUG, N_STEPS_PRODUCTION,
    VIZ_INTERVAL_DEFAULT, VIZ_INTERVAL_FREQUENT
)
from windopt.gch import gch
from windopt.optim.config import CampaignConfig
from windopt.optim.log import logger
from windopt.optim.utils import layout_from_ax_params
from windopt.winc3d import start_les, process_results
from windopt.winc3d.config import (
    FlowConfig, InflowConfig, LESConfig, NumericalConfig, OutputConfig, TurbineConfig
)
from windopt.winc3d.io import cleanup_viz_files
from windopt.winc3d.slurm import LESJob

# suppress pandas FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

LES_POLLING_INTERVAL = 30

def complete_noiseless_trial(ax_client: AxClient, trial_index: int, power: float):
    """
    Complete a trial without observation noise.
    """
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data={'power': (power, 0.0)}
    )

def run_gch_batch(ax_client: AxClient, campaign_config: CampaignConfig):
    """
    Run a batch of GCH trials on the provided Ax client.
    """
    trial_index_to_params, _ = ax_client.get_next_trials(
        max_trials=campaign_config.trial_generation_config.gch_batch_size,
        fixed_features=ObservationFeatures(
            {"fidelity": "gch"}
        )
    )
    for trial_index, parameters in trial_index_to_params.items():
        layout = layout_from_ax_params(campaign_config, parameters)
        power = gch(layout).sum()
        complete_noiseless_trial(ax_client, trial_index, power)

def run_les_batch(ax_client: AxClient, campaign_config: CampaignConfig):
    """
    Run a batch of LES trials on the provided Ax client.
    """
    active_jobs = _start_les_batch(ax_client, campaign_config)

    # briefly wait to let SLURM process the submission
    # before watching the jobs
    time.sleep(30)

    # Wait for all jobs in batch to complete and process results
    while active_jobs:
        active_jobs = _process_completed_les_jobs(ax_client, active_jobs)

        # Wait before checking again
        time.sleep(LES_POLLING_INTERVAL)

def _start_les_batch(
        ax_client: AxClient,
        campaign_config: CampaignConfig,
        ) -> list[tuple[LESJob, int]]:
    """
    Start a batch of LES trials on the provided Ax client.
    """
    logger.info(f"Generating LES trial batch")
    trial_index_to_params, _ = ax_client.get_next_trials(
        max_trials=campaign_config.trial_generation_config.les_batch_size,
        fixed_features=ObservationFeatures({"fidelity": "les"})
    )
    jobs = []
    for trial_index, parameters in trial_index_to_params.items():
        layout = layout_from_ax_params(campaign_config, parameters)
        config = _create_les_config(campaign_config, layout)

        logger.info(f"Submitting LES job for trial {trial_index}")
        # Start LES job
        job = start_les(
            run_name=f"{campaign_config.name}_trial_{trial_index}",
            config=config,
        )
        logger.info(f"Job submitted with ID: {job.slurm_job_id}")
        jobs.append((job, trial_index))

    return jobs

def _create_les_config(
    campaign_config: CampaignConfig,
    layout: list[tuple[float, float]],
) -> LESConfig:
    """
    Create LES configuration for a trial.
    """
    n_steps = N_STEPS_PRODUCTION if not campaign_config.debug_mode else N_STEPS_DEBUG
    viz_interval = VIZ_INTERVAL_DEFAULT if not campaign_config.debug_mode else VIZ_INTERVAL_FREQUENT
    
    return LESConfig(
        box_dims=campaign_config.box_dims,
        numerical=NumericalConfig(n_steps=n_steps),
        output=OutputConfig(viz_interval=viz_interval),
        turbines=TurbineConfig(layout=layout),
        inflow=InflowConfig(
            directory=INFLOW_20M,
            n_timesteps=INFLOW_20M_N_TIMESTEPS
        )
    )

def _process_completed_les_jobs(
        ax_client: AxClient,
        active_jobs: list[tuple[LESJob, int]],
        ) -> list[tuple[LESJob, int]]:
    """
    Process completed/failed LES jobs and return the remaining active jobs.
    """
    remaining_jobs = []
    for job, trial_index in active_jobs:
        if not (job.is_complete() or job.is_failed()):
            remaining_jobs.append((job, trial_index))
            continue

        if job.is_complete():
            try:
                power = process_results(job)
                complete_noiseless_trial(ax_client, trial_index, power)
            except Exception as e:
                logger.error(f"Error processing results for trial {trial_index}: {e}")
                ax_client.log_trial_failure(trial_index)

        elif job.is_failed():
            logger.error(f"LES job {trial_index} failed!")
            ax_client.log_trial_failure(trial_index)

        cleanup_viz_files(job.job_dir)

    return remaining_jobs