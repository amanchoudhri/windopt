"""
Run large-eddy simulations using SLURM.
"""
from datetime import datetime
from typing import Optional

import pandas as pd

from windopt.constants import PROJECT_ROOT
from windopt.winc3d.io import turbine_results
from windopt.winc3d.slurm import SlurmConfig, submit_job, LESJob
from windopt.winc3d.config import LESConfig

def start_les(
    run_name: str,
    config: LESConfig,
    slurm_config: Optional[SlurmConfig] = None,
    ) -> LESJob:
    """
    Start a large-eddy simulation run.

    Parameters
    ----------
    run_name: str
        The desired name of the run.
    config: LESConfig
        Complete configuration for the simulation.
    slurm_config: Optional[SlurmConfig]
        The SLURM configuration parameters.

    Returns
    -------
    LESJob
        The job object for the submitted job.
    """
    # create a directory in project_root/simulations for the run
    job_dir_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = PROJECT_ROOT / "simulations" / job_dir_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Write configuration files
    in_filename = "config.in"
    ad_filename = "turbines.ad"
    config.validate()
    config.write_winc3d_files(outdir, in_filename, ad_filename)
    config.to_json(outdir / "les_config.json")

    # submit the job
    if slurm_config is None:
        slurm_config = SlurmConfig()

    job = submit_job(
        input_file=outdir / in_filename,
        working_dir=outdir,
        turbines_file=(outdir / ad_filename) if config.turbines is not None else None,
        config=slurm_config
    )

    return job


def average_farm_power(power_data: pd.DataFrame, config: LESConfig) -> float:
    """
    Calculate the average farm power from simulation results.

    Parameters
    ----------
    power_data : pd.DataFrame
        Power output data from the simulation
    config : LESConfig
        Configuration used for the simulation

    Returns
    -------
    float
        Average farm power in watts
    """
    n_timesteps = config.numerical.n_steps
    spinup_timesteps = int(config.output.spinup_time / config.numerical.dt)

    n_recording_timesteps = n_timesteps - spinup_timesteps

    # Get cumulative power sums from final timestep for all turbines
    final_cumulative_power = power_data.loc[
        power_data['filenumber'] == config.n_outfiles,
        'Power_ave'
    ].sum()  # sum across all turbines

    # Average across all timesteps after spinup
    average_farm_power = final_cumulative_power / n_recording_timesteps

    return float(average_farm_power)


def process_results(job: LESJob) -> float:
    """
    Process the results of a LES job.
    """
    if not job.is_complete():
        raise ValueError("Job is not complete!")

    power_data = turbine_results(job.job_dir)

    config = LESConfig.from_json(job.job_dir / "les_config.json")

    return average_farm_power(power_data, config)