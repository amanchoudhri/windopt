"""
Run large-eddy simulations using SLURM.
"""
from pathlib import Path
from typing import Optional
from datetime import datetime

import f90nml
import numpy as np

from constants import D, HUB_HEIGHT, SMALL_BOX_DIMS
from slurm import SlurmConfig, submit_job

PROJECT_ROOT = Path(__file__).parent.parent

def make_ad_file(
    locations: np.ndarray,
    diameter: float,
    hub_height: float,
    outfile: Path
    ) -> str:
    """
    Make an .ad file for the given turbines.

    Parameters
    ----------
    locations: np.ndarray
        The x, z coordinates of the turbines, shape (n_turbines, 2).
    diameter: float
        The diameter of the turbines, in meters.
    hub_height: float
        The hub height of the turbines, in meters.
    outfile: Path
        The path to where the output .ad file should be written.
    """
    # Turbines are parameterized in a .ad file, where each line represents one turbine with 7 parameters:
    # X Y Z Nx Ny Nz D
    # where:
    # X Y Z: The center coordinates of the disc (250, 90, 1500 for first turbine)
    # Nx Ny Nz: The normal vector of the disc (1, 0, 0 means facing in x-direction)
    # D: The diameter of the disc (126 units)
    with open(outfile, "w+") as f:
        for location in locations:
            f.write(f"{location[0]} {hub_height} {location[1]} 1 0 0 {diameter}\n")

def make_in_file(
    outfile: Path,
    box_size: tuple[float, float, float],
    path_to_ad_file: Optional[Path] = None,
    n_turbines: Optional[int] = None,
    inflow_directory: Optional[Path] = None,
    inflow_n_timesteps: Optional[int] = None,
    debug_mode: bool = False
    ):
    """
    Create the .in configuration file for a large-eddy simulation run.
    """
    # read the base .in file
    BASE_CONFIG = PROJECT_ROOT / "config" / "les_base.in"
    config = f90nml.read(BASE_CONFIG)

    # x,y,z, box size
    config["FlowParam"]["xlx"] = box_size[0]
    config["FlowParam"]["yly"] = box_size[1]
    config["FlowParam"]["zlz"] = box_size[2]

    if path_to_ad_file is not None:
        if n_turbines is None:
            raise ValueError("n_turbines must be provided when using ADM mode")
        config["ADMParam"]["iadm"] = 1
        config["ADMParam"]["ADMcoords"] = str(path_to_ad_file)
        config["ADMParam"]["Ndiscs"] = n_turbines

    if inflow_directory is not None:
        if inflow_n_timesteps is None:
            raise ValueError("inflow_n_timesteps must be provided when using precursor planes")

        # simulation params: ensure it uses the precursor planes
        config["FlowParam"]["itype"] = 3
        config["FlowParam"]["iin"] = 3

        # file params: point to inflow directory and ensure it can read the files
        config["FileParam"]["InflowPath"] = str(inflow_directory)
        config["FileParam"]["NTimeSteps"] = inflow_n_timesteps

        # figure out how many files are in the inflow directory
        n_files = len(list(Path(inflow_directory).glob("inflow[1-9]*")))
        config["FileParam"]["NInflows"] = n_files
        
    if debug_mode:
        config["ilast"] = 1000
        print(config)

    # write the config to the file
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    f90nml.write(config, outfile)


def start_les(
    run_name: str,
    locations: np.ndarray,
    rotor_diameter: float = D,
    hub_height: float = HUB_HEIGHT,
    box_size: tuple[float, float, float] = SMALL_BOX_DIMS,
    slurm_config: Optional[SlurmConfig] = None
    ) -> int:
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
    box_size: tuple[float, float, float]
        The size of the simulation box, in meters.
    slurm_config: Optional[SlurmConfig]
        The SLURM configuration parameters.
    
    Returns
    -------
    int
        The SLURM job ID of the submitted job.
    """
    # create a directory in project_root/simulations for the run
    # with a datetime string appended to the run name
    dirname = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = PROJECT_ROOT / "simulations" / dirname

    outdir.mkdir(parents=True, exist_ok=True)

    turbines_file = outdir / "turbines.ad"
    config_file = outdir / "config.in"

    make_ad_file(locations, rotor_diameter, hub_height, turbines_file)
    make_in_file(config_file, box_size=box_size)

    # submit the job
    if slurm_config is None:
        slurm_config = SlurmConfig()

    job_id = submit_job(config_file, working_dir=outdir, slurm_config=slurm_config)

    return job_id
