"""
Generate and submit SLURM jobs to run simulations with WInc3D.
"""
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyslurm

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

module load intel-parallel-studio/2020

export MLX5_SINGLE_THREADED=0
export WINC3D=/moto/home/ac4972/WInc3D/winc3d

mkdir -p out
cd out

ulimit -c 0

mpiexec -bootstrap slurm -print-rank-map $WINC3D ${{INPUT_FILE}}
"""

def submit_job(
    input_file: Path,
    working_dir: Path,
    config: Optional[SlurmConfig] = None
) -> pyslurm.Job:
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
    pyslurm.Job
        A SLURM job object that can be watched for completion.
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
        ["sbatch", "--parsable", str(script_path), str(input_file)],
        cwd=working_dir,
        capture_output=True,
        text=True,
        check=True
    )
    job_id = int(result.stdout.strip())
    
    return pyslurm.Job(job_id)