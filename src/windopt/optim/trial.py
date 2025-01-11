"""
Trial generation and management functions for both LES and GCH evaluations.
"""

import logging
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
from windopt.optim.utils import layout_from_ax_params
from windopt.winc3d import start_les, process_results
from windopt.winc3d.config import FlowConfig, InflowConfig, LESConfig, NumericalConfig, OutputConfig, TurbineConfig
from windopt.winc3d.io import cleanup_viz_files
from windopt.winc3d.slurm import LESJob

# suppress pandas FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


def complete_noiseless_trial(ax_client: AxClient, trial_index: int, power: float):
    """
    Complete a trial without observation noise.
    """
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data={'power': (power, 0.0)}
    )

def run_gch_batch(ax_client: AxClient, campaign_config: CampaignConfig, batch_size: int):
    """
    Run a batch of GCH trials on the provided Ax client.
    """
    trial_index_to_params, _ = ax_client.get_next_trials(
        max_trials=batch_size,
        fixed_features=ObservationFeatures(
            {"fidelity": "gch"}
        )
    )
    for trial_index, parameters in trial_index_to_params.items():
        layout = layout_from_ax_params(campaign_config, parameters)
        power = gch(layout).sum()
        complete_noiseless_trial(ax_client, trial_index, power)

def start_les_batch(
        ax_client: AxClient,
        campaign_config: CampaignConfig,
        batch_size: int,
        debug_mode: bool,
        logger: logging.Logger
        ) -> list[tuple[LESJob, int]]:
    """
    Start a batch of LES trials on the provided Ax client.
    """
    logger.info(f"Generating LES trial batch")
    trial_index_to_params, _ = ax_client.get_next_trials(
        max_trials=batch_size,
        fixed_features=ObservationFeatures(
            {"fidelity": "les"}
        )
    )
    jobs = []
    for trial_index, parameters in trial_index_to_params.items():
        layout = layout_from_ax_params(campaign_config, parameters)
        n_steps = N_STEPS_PRODUCTION if not debug_mode else N_STEPS_DEBUG
        viz_interval = VIZ_INTERVAL_DEFAULT if not debug_mode else VIZ_INTERVAL_FREQUENT
        config = LESConfig(
            box_dims=campaign_config.box_dims,
            numerical=NumericalConfig(n_steps=n_steps),
            output=OutputConfig(viz_interval=viz_interval),
            turbines=TurbineConfig(layout=layout),
            inflow=InflowConfig(directory=INFLOW_20M, n_timesteps=INFLOW_20M_N_TIMESTEPS)
        )
        logger.info(f"Submitting LES job for trial {trial_index}")
        # Start LES job
        job = start_les(
            run_name=f"{campaign_config.name}_trial_{trial_index}",
            config=config,
        )
        logger.info(f"Job submitted with ID: {job.slurm_job_id}")
        jobs.append((job, trial_index))

    return jobs

def process_completed_les_jobs(
        ax_client: AxClient,
        active_jobs: list[tuple[LESJob, int]],
        logger: logging.Logger,
        ):
    """
    Process completed LES jobs.
    """
    for job, trial_index in active_jobs[:]:
        if job.is_complete():
            try:
                config = LESConfig.from_json(job.job_dir / "les_config.json")
                power = process_results(job, config)
                complete_noiseless_trial(ax_client, trial_index, power)
            except Exception as e:
                logger.error(f"Error processing results for trial {trial_index}: {e}")
                ax_client.log_trial_failure(trial_index)
            active_jobs.remove((job, trial_index))

            # Clean up large visualization files
            cleanup_viz_files(job.job_dir)

        elif job.is_failed():
            logger.error(f"LES job {trial_index} failed!")
            ax_client.log_trial_failure(trial_index)
            active_jobs.remove((job, trial_index))

    return active_jobs
