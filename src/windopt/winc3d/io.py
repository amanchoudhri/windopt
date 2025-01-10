"""
Read, write, and parse files for WInc3D.
"""

from importlib import resources
from pathlib import Path
from typing import Optional

import f90nml
import numpy as np
import pandas as pd

from windopt.constants import DT
from windopt.layout import Layout

def make_ad_file(
    layout: Layout,
    diameter: float,
    hub_height: float,
    outfile: Path
    ) -> str:
    """
    Make an .ad file for the given turbines.

    Parameters
    ----------
    layout: Layout
        The layout of the turbines.
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
        for location in layout.box_coords:
            f.write(f"{location[0]} {hub_height} {location[1]} 1 0 0 {diameter}\n")

def make_in_file(
    outfile: Path,
    box_size: tuple[float, float, float],
    path_to_ad_file: Optional[Path] = None,
    n_turbines: Optional[int] = None,
    inflow_directory: Optional[Path] = None,
    inflow_n_timesteps: Optional[int] = None,
    debug_mode: bool = False,
    frequent_viz: bool = False
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
        
    if frequent_viz:
        # write to the log file every generated minute
        config["FileParam"]["imodulo"] = int(60 / DT)
        # additionally, save the flow field every generated minute
        config["FileParam"]["isave"] = int(60 / DT)

    if debug_mode:
        config["NumConfig"]["ilast"] = 2000
        write_interval = 20 if frequent_viz else 1000
        config["FileParam"]["imodulo"] = write_interval
        config["StatParam"]["spinup_time"] = 0

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

def tail(filename, n=10):
    with open(filename, 'rb') as f:
        # Go to end of file
        f.seek(0, 2)
        # Get file size
        size = f.tell()
        
        lines = []
        position = size
        
        # Read backwards until we have n lines
        while len(lines) < n and position > 0:
            # Move back one byte
            position -= 1
            f.seek(position)
            
            # Read one byte
            char = f.read(1)
            
            # If we hit a newline, store the line
            if char == b'\n':
                line = f.readline()
                lines.append(line.decode())
                
        # Reverse the lines since we read backwards
        return lines[::-1]

def is_job_completed_successfully(log_file: Path) -> bool:
    """
    Get the status of the job from the log file.
    """
    tail_lines = tail(log_file, 10)

    # on successful completion, the winc3d routine outputs the string "time per
    # time_step:" we can check for this pattern to determine if the job is
    # complete
    success_pattern = "time per time_step:"

    # TODO: this is a hack, but it works for now
    if success_pattern in ''.join(tail_lines):
        return True
    else:
        return False

def cleanup_viz_files(job_dir: Path):
    """
    Clean up large visualization files generated by WInc3D.
    """
    # remove all ux, uy, uz, pp, vort, and gammadisc files from job_dir/out
    subdir = Path(job_dir) / "out"
    patterns = ["ux*", "uy*", "uz*", "pp*", "vort*", "gammadisc*"]

    for pattern in patterns:
        for f in subdir.glob(pattern):
            f.unlink()

def turbine_results(job_dir: Path) -> pd.DataFrame:
    """
    Get the power history of each turbine from the job output.
    """
    # read *.adm files from the job_dir/out subdirectory
    n_files = len(list(job_dir.glob("out/discs_time[0-9]*.adm")))
    # parse each file and stack the results
    data = []
    for i in range(1, n_files + 1):
        adm_file = job_dir / f'out/discs_time{i}.adm'
        adm_info = read_adm_file(adm_file)
        adm_info['filenumber'] = i
        data.append(adm_info)
    return pd.concat(data).reset_index(names='turbine')

def read_turbine_information(job_dir: Path) -> pd.DataFrame:
    """
    Get the information of the turbines from the job output.
    """
    turbines_ad_file = job_dir / 'turbines.ad'
    if not turbines_ad_file.exists():
        raise ValueError(
            "Cannot find turbine locations! No turbines.ad "
            "found in job directory."
            )

    # the format is 7 space-separated floats, representing:
    # x y z normal_x normal_y normal_z rotor_diameter
    # for (x, y, z) the location of the turbine hub, and
    # (normal_x, normal_y, normal_z) the components of normal vector to the
    # turbine disk.
    turbines_df = pd.read_csv(turbines_ad_file, sep=' ', engine='python', header=None)
    turbines_df.columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'D']
    return turbines_df

def read_turbine_locations(job_dir: Path) -> np.ndarray:
    """
    Get the (x, z) coordinates of the turbines from the job output.
    """
    return read_turbine_information(job_dir)[['x', 'z']].values
