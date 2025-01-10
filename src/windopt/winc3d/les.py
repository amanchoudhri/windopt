"""
Run large-eddy simulations using SLURM.
"""
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd

from windopt.constants import D, HUB_HEIGHT, PROJECT_ROOT
from windopt.winc3d.slurm import SlurmConfig, submit_job, LESJob
from windopt.winc3d.io import make_ad_file, make_in_file, turbine_results
from windopt.layout import Layout

def start_les(
    run_name: str,
    layout: Layout,
    inflow_directory: Optional[Path] = None,
    inflow_n_timesteps: Optional[int] = None,
    rotor_diameter: float = D,
    hub_height: float = HUB_HEIGHT,
    slurm_config: Optional[SlurmConfig] = None,
    debug_mode: bool = False,
    frequent_viz: bool = False,
    ) -> LESJob:
    """
    Start a large-eddy simulation run.

    Parameters
    ----------
    run_name: str
        The desired name of the run.
    locations: np.ndarray
        The x, z coordinates of the turbines, shape (n_turbines, 2).
    rotor_diameter: float
        The diameter of the turbines, in meters.
    hub_height: float
        The hub height of the turbines, in meters.
    slurm_config: Optional[SlurmConfig]
        The SLURM configuration parameters.
    debug_mode: bool
        Whether to run in debug mode.
    frequent_viz: bool
        Whether to run WInc3D with frequent visualization output.
    Returns
    -------
    LESJob
        The job object for the submitted job.
    """
    # create a directory in project_root/simulations for the run
    # with a datetime string appended to the run name
    job_dir_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = PROJECT_ROOT / "simulations" / job_dir_name

    outdir.mkdir(parents=True, exist_ok=True)

    turbines_file = outdir / "turbines.ad"
    config_file = outdir / "config.in"

    make_ad_file(layout, rotor_diameter, hub_height, turbines_file)
    make_in_file(
        config_file,
        box_size=layout.box_dims,
        path_to_ad_file=turbines_file,
        n_turbines=layout.n_turbines,
        inflow_directory=inflow_directory,
        inflow_n_timesteps=inflow_n_timesteps,
        debug_mode=debug_mode,
        frequent_viz=frequent_viz,
        )

    # submit the job
    if slurm_config is None:
        slurm_config = SlurmConfig()

    job = submit_job(
        config_file,
        working_dir=outdir,
        turbines_file=turbines_file,
        config=slurm_config
        )

    return job

def process_results(job: LESJob, debug_mode: bool = False) -> float:
    """
    Process the results of a LES job.
    """
    if not job.is_complete():
        raise ValueError("Job is not complete!")

    power_data = job.turbine_results()

    # HARDCODED VALUES
    # TODO: make this dynamic
    N_TIMESTEPS = 45000 if not debug_mode else 1000
    SPINUP_TIMESTEPS = 9000 if not debug_mode else 0
    N_OUT_FILES = 5 if not debug_mode else 1

    # get the average farm power output after the spinup period, in watts
    average_farm_power = power_data.loc[
       (power_data['filenumber'] == N_OUT_FILES),
       ['Power_ave']
    ].sum() / (N_TIMESTEPS - SPINUP_TIMESTEPS)

    return float(average_farm_power.iloc[0])
