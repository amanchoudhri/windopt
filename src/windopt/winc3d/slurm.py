"""
Generate and submit SLURM jobs to run simulations with WInc3D.
"""
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from windopt.winc3d.io import is_job_completed_successfully

class LESJob:
    def __init__(self, slurm_job_id: int, job_dir: Path):
        self.slurm_job_id = slurm_job_id
        self.job_dir = Path(job_dir)

    def _job_info(self) -> tuple:
        # Get information about the job using the `sacct` command.
        # Example: sacct -j 19680110 -o JobID,State,ExitCode -n -P
        result = subprocess.run(
            ["sacct", "-j", str(self.slurm_job_id), "-o", "Elapsed,State,ExitCode", "-n", "-P"],
            capture_output=True,
            text=True,
            check=True
        )
        # get the first line of the output
        main_info = result.stdout.split("\n")[0].split("|")
        return main_info

    def is_running_squeue(self) -> bool:
        """
        Check if the job is currently running using `squeue`.

        Needed in case `sacct` is not working.
        """
        result = subprocess.run(
            ["squeue", "-j", str(self.slurm_job_id), "--noheader", "-O", "jobid,state"],
            capture_output=True,
            text=True,
            check=True
        )
        # check if the job has an squeue entry
        if result.stdout.strip() == "":
            return False
        # check if the job is in the running state
        state = result.stdout.split("\n")[0].split("|")[1].strip()
        is_running_state = state == "R"
        is_pending_state = state == "PD"
        return is_running_state or is_pending_state

    def is_complete(self) -> bool:
        """
        Check if the job is complete.

        First try to use `sacct` to check the job status. If that fails, fall
        back to checking the log file to determine whether it was a success or
        failure.
        """
        info = self._job_info()

        # if for some reason sacct isn't working, check the log file
        if len(info) < 2:
            # first check squeue to see if the job is active
            if self.is_running_squeue():
                return False
            # if it's not active, check the log file to determine
            # whether it was a success or failure
            log_file = list(self.job_dir.glob("winc3d_*.log"))[0]
            return is_job_completed_successfully(log_file)
        else:
            return info[1] == "COMPLETED"

    def is_failed(self) -> bool:
        info = self._job_info()
        return info[1] == "FAILED"

    def status(self) -> str:
        info = self._job_info()
        return info[1]


@dataclass
class SlurmConfig:
    account: str = "edu"
    partition: str = "short"
    job_name: str = "winc3d"
    nodes: int = 8
    ntasks_per_node: int = 24
    mem_gb: int = 128
    time_limit: str = "10:00:00"
    
    @property
    def total_tasks(self) -> int:
        return self.nodes * self.ntasks_per_node

def create_slurm_script(config: SlurmConfig) -> str:
    return f"""#!/bin/bash
#SBATCH --account={config.account}
#SBATCH --job-name={config.job_name}
#SBATCH --nodes={config.nodes}
#SBATCH --ntasks={config.total_tasks}
#SBATCH --ntasks-per-node={config.ntasks_per_node}
#SBATCH --time={config.time_limit}
#SBATCH --output={config.job_name}_%j.log
#SBATCH --error={config.job_name}_%j.err

# Get the input file from command line argument
INPUT_FILE=$1
TURBINES_FILE=$2

module load intel-parallel-studio/2020

export MLX5_SINGLE_THREADED=0
export WINC3D=/moto/home/ac4972/WInc3D/winc3d

mkdir -p out
cd out

ln -s $INPUT_FILE config.in
ln -s $TURBINES_FILE turbines.ad

ulimit -c 0

mpiexec -bootstrap slurm -print-rank-map $WINC3D config.in
"""

def submit_job(
    input_file: Path,
    working_dir: Path,
    turbines_file: Optional[Path] = None,
    config: Optional[SlurmConfig] = None
) -> LESJob:
    """
    Submit a SLURM job for the given input file.
    
    Parameters
    ----------
    input_file : Path
        The input file to run with WInc3D
    working_dir : Path
        Directory where the job should run
    config : Optional[SlurmConfig]
        SLURM configuration parameters. If None, uses defaults.
    
    Returns
    -------
    LESJob
        The job object for the submitted job.
    """
    if config is None:
        config = SlurmConfig()
    
    # Create temporary script file
    script_path = working_dir / "run.slurm"
    script_content = create_slurm_script(config)
    
    with open(script_path, "w") as f:
        f.write(script_content)

    script_path.chmod(0o755)  # Make executable
    
    # Submit the job
    result = subprocess.run(
        ["sbatch", "--parsable", str(script_path), str(input_file), str(turbines_file)],
        cwd=working_dir,
        capture_output=True,
        text=True,
        check=True
    )
    job_id = int(result.stdout.strip())
    job = LESJob(job_id, working_dir)
    return job
