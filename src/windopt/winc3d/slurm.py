"""
Generate and submit SLURM jobs to run simulations with WInc3D.
"""
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from windopt.winc3d.io import read_adm_file

class LESJob:
    def __init__(self, slurm_job_id: int, job_dir: Path):
        self.slurm_job_id = slurm_job_id
        self.job_dir = job_dir

    def _job_info(self) -> tuple:
        # Get information about the job using the `sacct` command.
        # Example: sacct -j 19680110.0 -o JobID,State,ExitCode -n -P
        job_step_0 = f'{self.slurm_job_id}.0'
        result = subprocess.run(
            ["sacct", "-j", job_step_0, "-o", "Elapsed,State,ExitCode", "-n", "-P"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split("|")

    def is_complete(self) -> bool:
        info = self._job_info()
        return info[1] == "COMPLETED"

    def turbine_results(self) -> Union[pd.DataFrame, None]:
        """
        Get the power history of each turbine from the job output.
        """
        if not self.is_complete():
            return None
        # read *.adm files from the job out subdirectory
        n_files = len(list(self.job_dir.glob("discs_time[0-9]*.adm")))
        # parse each file and stack the results
        data = []
        for i in range(1, n_files + 1):
            adm_file = self.job_dir / f'discs_time{i}.adm'
            adm_info = read_adm_file(adm_file)
            adm_info['filenumber'] = i
            data.append(adm_info)
        return pd.concat(data).reset_index(names='turbine')


@dataclass
class SlurmConfig:
    account: str = "edu"
    partition: str = "short"
    job_name: str = "winc3d"
    nodes: int = 4
    ntasks_per_node: int = 24
    mem_gb: int = 128
    time_limit: str = "12:00:00"
    
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
