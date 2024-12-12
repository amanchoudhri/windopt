"""
Read, write, and parse files for WInc3D.
"""

from importlib import resources
from pathlib import Path
from typing import Optional

import f90nml
import numpy as np
import pandas as pd

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
    cfg_ptr = resources.files('windopt').joinpath('config/les_base.in')
    with resources.as_file(cfg_ptr) as base_cfg_path:
        config = f90nml.read(base_cfg_path)

    # x,y,z, box size
    config["FlowParam"]["xlx"] = box_size[0]
    config["FlowParam"]["yly"] = box_size[1]
    config["FlowParam"]["zlz"] = box_size[2]

    if path_to_ad_file is not None:
        if n_turbines is None:
            raise ValueError("n_turbines must be provided when using ADM mode")
        config["ADMParam"]["iadm"] = 1
        # links only the filename since the fortran code has hard character
        # limits. `slurm.py` will handle creating appropriate symbolic links in
        # the working directory to the actual path
        config["ADMParam"]["ADMcoords"] = path_to_ad_file.name
        config["ADMParam"]["Ndiscs"] = n_turbines

    if inflow_directory is not None:
        if inflow_n_timesteps is None:
            raise ValueError("inflow_n_timesteps must be provided when using precursor planes")

        # simulation params: ensure it uses the precursor planes
        config["FlowParam"]["itype"] = 3
        config["FlowParam"]["iin"] = 3

        # file params: point to inflow directory and ensure it can read the files
        # the fortran code requires a trailing slash
        config["FileParam"]["InflowPath"] = f'{str(inflow_directory)}/'
        config["FileParam"]["NTimeSteps"] = inflow_n_timesteps

        # figure out how many files are in the inflow directory
        n_files = len(list(Path(inflow_directory).glob("inflow[1-9]*")))
        config["FileParam"]["NInflows"] = n_files
        
    if debug_mode:
        config["NumConfig"]["ilast"] = 1000
        print(config)

    # write the config to the file
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    f90nml.write(config, outfile)

def read_adm_file(adm_file: Path):
    """
    Read a .adm file (really just a CSV).
    """
    return pd.read_csv(
            adm_file, sep=', ', engine='python'
            )
