"""
Command-line interface for WInc3D simulation management.
"""
import argparse
from pathlib import Path

from windopt.constants import N_STEPS_DEBUG, VIZ_INTERVAL_FREQUENT
from windopt.winc3d.config import LESConfig
from windopt.winc3d.les import start_les

def rerun_simulation(
    job_dir: Path, 
    debug_mode: bool = False,
    frequent_viz: bool = False,
    ) -> None:
    """
    Rerun a simulation with optional modifications.
    
    Parameters
    ----------
    job_dir : Path
        Directory containing the original simulation
    debug_mode : bool
        If True, run a shorter simulation for testing
    frequent_viz : bool
        If True, output visualization data more frequently
    """
    # Read existing configuration
    config = LESConfig.from_json(job_dir / "les_config.json")

    # Modify configuration based on flags
    if debug_mode:
        config.numerical.n_steps = N_STEPS_DEBUG
    
    if frequent_viz:
        config.output.viz_interval = VIZ_INTERVAL_FREQUENT
    
    # Start the new simulation
    job = start_les(
        run_name=f"rerun_{job_dir.name}",
        config=config
    )
    print(f"Reran simulation with job ID: {job.slurm_job_id}")

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