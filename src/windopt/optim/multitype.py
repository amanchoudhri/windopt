"""
Multi-fidelity BO using LES and GCH, using the Ax MultiTypeExperiment class.
"""

from datetime import datetime
import logging
import time 
import warnings

from argparse import ArgumentParser

import numpy as np

from ax.core.observation import ObservationFeatures
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.modelbridge.transforms.task_encode import TaskChoiceToIntTaskChoice
from ax.modelbridge.transforms.unit_x import UnitX
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from ax.storage.json_store.save import save_experiment
from ax.utils.common.logger import ROOT_LOGGER

from windopt.constants import SMALL_BOX_DIMS, PROJECT_ROOT
from windopt.gch import gch
from windopt.main import (
    turbine_parameters,
    load_initial_data
)
from windopt.winc3d import start_les, process_results
from windopt.winc3d.io import cleanup_viz_files
from windopt.winc3d.slurm import LESJob

DEFAULT_GCH_BATCH_SIZE = 50
LES_BATCH_SIZE = 4

# each LES run takes 40 minutes, and I want a max duration of 12 hours
MAX_LES_BATCHES = 18

# how often to check if a LES has finished (in seconds)
POLLING_INTERVAL = 30

# suppress pandas FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

multi_type_transforms = [
    TaskChoiceToIntTaskChoice,  # Since we're using a string valued task parameter.
    UnitX, 
    StandardizeY
]


def run_multi_fidelity_experiment(
        experiment_name: str,
        gch_batch_size: int,
        debug_mode: bool = False,
        ) -> AxClient:
    """
    Run a multi-fidelity experiment using LES and GCH.
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
    ROOT_LOGGER.addHandler(fh)

    # hardcoded for now
    N_TURBINES = 4

    # Set up search space
    param_names = turbine_parameters(N_TURBINES, optimize_angles=False)

    # Add the task/fidelity parameter
    param_names.append({
        "name": "fidelity",
        "type": "choice",
        "values": ["les", "gch"],
        "is_task": True,
        "target_value": "les"
    })

    # Setup a generation strategy without initial Sobol steps since
    # we have preloaded initial trial data.
    gs = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=4,
                model_kwargs={"transforms": multi_type_transforms},
            )
        ]
    )

    # Create Ax client
    ax_client = AxClient(generation_strategy=gs)
    ax_client.create_experiment(
        name=experiment_name,
        parameters=param_names,
        objectives={"power": ObjectiveProperties(minimize=False)},
    )

    # Load initial data from previous LES and GCH runs
    load_initial_data(ax_client, fidelity='les', include_fidelity_parameter=True)
    load_initial_data(ax_client, fidelity='gch', include_fidelity_parameter=True)

    # Campaign directory
    campaign_dir = PROJECT_ROOT / "campaigns" / experiment_name
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # Run optimization loop
    for batch in range(MAX_LES_BATCHES):
        logger.info(f"Running batch {batch + 1} of {MAX_LES_BATCHES}")

        # Run the GCH trials
        # It says batch, but we run them sequentially to improve performance
        for i in range(gch_batch_size):
            logger.info(f"Running GCH trial {i + 1} of {gch_batch_size}")
            parameters, trial_index = ax_client.get_next_trial(
                fixed_features=ObservationFeatures(
                    {"fidelity": "gch"}
                )
            )
            locations = np.array([
                [parameters[f'x{i}'] for i in range(1, N_TURBINES + 1)],
                [parameters[f'z{i}'] for i in range(1, N_TURBINES + 1)]
            ]).T
            power = gch(locations).sum()
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={'power': (power, 0.0)}
            )

        # Generate batch of LES trials to evaluate
        logger.info(f"Generating LES trial batch")
        trial_index_to_param, _ = ax_client.get_next_trials(
            max_trials=LES_BATCH_SIZE,
            fixed_features=ObservationFeatures(
                {"fidelity": "les"}
            )
        )
        # Start all jobs in the batch
        active_jobs: list[tuple[LESJob, int]] = []
        for trial_index, parameters in trial_index_to_param.items():
            # Convert parameters to locations array
            locations = np.array([
                [parameters[f'x{i}'] for i in range(1, N_TURBINES + 1)],
                [parameters[f'z{i}'] for i in range(1, N_TURBINES + 1)]
            ]).T

            logger.info(f"Submitting LES job for trial {trial_index}")
            # Start LES job
            job = start_les(
                run_name=f"{experiment_name}_trial_{trial_index}",
                locations=locations,
                box_size=SMALL_BOX_DIMS,
                debug_mode=debug_mode,
            )
            logger.info(f"Job submitted with ID: {job.slurm_job_id}")
            active_jobs.append((job, trial_index))

        # give it 30 seconds to let SLURM process the job submission
        time.sleep(30)

        # Wait for all jobs in batch to complete
        while active_jobs:
            for job, trial_index in active_jobs[:]:
                if job.is_complete():
                    try:
                        power = process_results(job, debug_mode=debug_mode)
                        ax_client.complete_trial(
                            trial_index=trial_index,
                            raw_data={'power': (power, 0.0)}
                        )
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
            
            # Save current state
            exp_to_df(ax_client.experiment).to_csv(
                str(campaign_dir / "trials.csv")
            )
            save_experiment(
                ax_client.experiment,
                str(campaign_dir / "experiment.json")
            )

            # Wait before checking again
            time.sleep(POLLING_INTERVAL)

    return ax_client


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment_name", type=str, help="The name of the experiment")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--gch_batch_size",
        type=int,
        default=DEFAULT_GCH_BATCH_SIZE,
        help="The number of GCH trials to run in each batch"
    )
    args = parser.parse_args()

    run_multi_fidelity_experiment(
        experiment_name=args.experiment_name,
        gch_batch_size=args.gch_batch_size,
        debug_mode=args.debug
    )
