"""
Command-line interface for WInc3D simulation management.
"""
import argparse
from pathlib import Path
from datetime import datetime

import f90nml

from windopt.constants import INFLOW_20M, INFLOW_20M_N_TIMESTEPS, SMALL_ARENA_DIMS
from windopt.winc3d.les import start_les
from windopt.winc3d.io import read_turbine_information
from windopt.layout import Layout

def rerun_simulation(
    job_dir: Path, 
    debug_mode: bool = False,
    frequent_viz: bool = False,
) -> None:
    """
    Rerun a simulation from an existing job directory.
    
    Parameters
    ----------
    job_dir : Path
        Path to the original job directory
    debug_mode : bool
        Whether to run in debug mode (shorter simulation)
    frequent_viz : bool
        Whether to output visualization data more frequently
    """
    config_file = job_dir / "config.in"
    turbines_file = job_dir / "turbines.ad"
    
    if not config_file.exists():
        raise FileNotFoundError(f"No config.in found in {job_dir}")
    
    if not turbines_file.exists():
        raise FileNotFoundError(f"No turbines.ad found in {job_dir}")
    
    # Read box dimensions from config file
    config = f90nml.read(config_file)
    box_size = (
        float(config['FlowParam']['xlx']),
        float(config['FlowParam']['yly']),
        float(config['FlowParam']['zlz'])
    )
    
    # Get turbine locations
    turbine_info = read_turbine_information(job_dir)
    layout = Layout(
        turbine_info[['x', 'z']].values,
        system="box",
        arena_dims=SMALL_ARENA_DIMS,
        box_dims=box_size
    )
    
    # Define new run name
    run_name = f"rerun_{job_dir.name}"
    
    # Start new simulation using the existing API
    job = start_les(
        run_name=run_name,
        layout=layout,
        rotor_diameter=float(turbine_info['D'].iloc[0]),
        hub_height=float(turbine_info['y'].iloc[0]),
        inflow_directory=INFLOW_20M,
        inflow_n_timesteps=INFLOW_20M_N_TIMESTEPS,
        debug_mode=debug_mode,
        frequent_viz=frequent_viz,
    )
    
    print(f"Submitted job {job.slurm_job_id}")

def main():
    parser = argparse.ArgumentParser(description="WInc3D simulation management")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Rerun command
    rerun_parser = subparsers.add_parser("rerun", help="Rerun an existing simulation")
    rerun_parser.add_argument(
        "job_dir",
        type=Path,
        help="Path to the job directory to rerun"
    )
    rerun_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (shorter simulation)"
    )
    rerun_parser.add_argument(
        "--frequent-viz",
        action="store_true",
        help="Output visualization data more frequently"
    )
    
    args = parser.parse_args()
    
    if args.command == "rerun":
        rerun_simulation(
            args.job_dir,
            debug_mode=args.debug,
            frequent_viz=args.frequent_viz
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 