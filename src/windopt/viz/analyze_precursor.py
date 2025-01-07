"""
Analyze a precursor simulation.

Report mean hub height velocity and turbulence intensity. Also
generate figures showing how velocity and turbulence intensity vary
by height.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from windopt.constants import (
    PROJECT_ROOT, HUB_HEIGHT, DT,
    SMALL_BOX_DIMS, LARGE_BOX_DIMS
)

ABL_HEIGHT = 504.        # m
U_STAR = 0.442           # m/s
VON_KARMAN = 0.4         # dimensionless
ROUGHNESS_LENGTH = 0.05  # m

def get_inflow_path(idx: int, inflow_dir: Path):
    return f'{inflow_dir}/inflow{idx}'

def parse_inflow(inflow_path: Path, nt: int, ny, nz):
    with open(inflow_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float64)

    # Reshape the data
    # The file should contain [u,v,w][timesteps][ny][nz]
    N_COMPONENTS = 3
    total_expected = nt * ny * nz * N_COMPONENTS

    if len(data) != total_expected:
        raise ValueError(
                "Recieved a different number of elements than expected. "
                f"Expected nt * ny * nz * components = {total_expected}, "
                f"but received {len(data)}."
                )

    # The binary file contains 3 consecutive blocks of data (ux, uy, uz)
    # Each block has size (ntimesteps * ny * nz) and is written in Fortran order
    # We read each block separately and reshape to (nt, ny, nz) to preserve the correct data layout
    plane_size = ny * nz
    component_size = nt * plane_size

    # Split into three velocity components
    ux = data[0:component_size].reshape((nt, ny, nz), order='F')
    uy = data[component_size:2*component_size].reshape((nt, ny, nz), order='F')
    uz = data[2*component_size:3*component_size].reshape((nt, ny, nz), order='F')

    data = np.stack((ux, uy, uz))

    return data

def log_law(resolution: int = 1000, y_min: float = 10, abl_height: float = ABL_HEIGHT):
    """
    Calculate and return the theoretical log law wind speed profile.
    """
    y_log = np.logspace(
        np.log10(y_min),
        np.log10(abl_height),
        resolution
        )
    u_log = (U_STAR / VON_KARMAN) * np.log(y_log / ROUGHNESS_LENGTH)
    return y_log, u_log

def plot_abl_profiles(y_coords, u_mean, y_log, u_log, TI):
    """Create plots matching Bempedelis paper style."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot (a) - Mean velocity profile
    ax1.semilogx(y_log, u_log, '-k', label='Log law')
    ax1.semilogx(y_coords, u_mean, 'ko', markerfacecolor='white', 
                 label='LES')
    
    ax1.set_xlim(10, ABL_HEIGHT)
    # ax1.set_ylim(0, 20)
    ax1.set_xlabel('y [m]')
    ax1.set_ylabel(r'$\overline{u} [m/s]$')
    # let's add a vertical line at hub height
    ax1.axvline(HUB_HEIGHT, color='grey', alpha=0.5, linestyle='--', label='Hub height')
    ax1.legend()
    
    # Plot (b) - Turbulence intensity profile
    ax2.plot(TI, y_coords, 'ko', markerfacecolor='white')
    ax2.set_ylim(0, ABL_HEIGHT)
    # ax2.set_xlim(0, 15)
    ax2.set_xlabel('TI [%]')
    ax2.set_ylabel('y [m]')
    ax2.axhline(HUB_HEIGHT, color='grey', alpha=0.5, linestyle='--', label='Hub height')
    ax2.legend()

    plt.tight_layout()
    return fig

def inflow_statistics(inflow: np.ndarray):
    # inflow is an array of shape [3, N_TIMESTEPS, ny, nz]
    print(inflow.shape)

    # Calculate mean velocity profile (v at each height)
    u_mean = np.mean(inflow[0], axis=(0, 2))  # Average over time and z
    # Calculate turbulence intensity profile (TI at each height)
    # Standard deviation at each height
    u_std = np.std(inflow[0], axis=(0, 2))
    turbulence_intensity = (u_std / u_mean) * 100

    return u_mean, turbulence_intensity

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("inflow_dir", type=Path)
    p.add_argument("--arena", type=str, required=True)
    p.add_argument("--nt", type=int, required=True)
    p.add_argument("--dt", type=float, required=False, default=DT)

    args = p.parse_args()

    if args.arena == "small_10m":
        n = (200, 51, 144)        # gridpoints
        dims = SMALL_BOX_DIMS

    elif args.arena == "small_20m":
        n = (100, 51, 72)          # gridpoints
        dims = SMALL_BOX_DIMS

    elif args.arena == "small_40m":
        n = (50, 51, 36)          # gridpoints
        dims = SMALL_BOX_DIMS  # arena dimensions

    elif args.arena == "large":
        n = (100, 51, 84)
        dims = LARGE_BOX_DIMS

    elif args.arena == "large_tall":
        n = (100, 75, 84)
        dims = LARGE_BOX_DIMS

    _, yly, _ = dims
    _, ny, nz = n

    running_means = []

    output = []
    inflow_paths = list(args.inflow_dir.glob("inflow[0-9]*"))
    n_inflows = len(inflow_paths)

    get_num = lambda path: int(path.name.replace("inflow", ""))

    velocity_profile = []
    turbulence_profile = []

    # Calculate mean velocity at hub height
    for inflow_path in sorted(inflow_paths, key=get_num):
        i = get_num(inflow_path)
        output.append(f'Inflow {i} --- {inflow_path}')
        inflow = parse_inflow(inflow_path, nt=args.nt, ny=ny, nz=nz)

        u_mean, turbulence_intensity = inflow_statistics(inflow)
        velocity_profile.append(u_mean)
        turbulence_profile.append(turbulence_intensity)

    velocity_profile = np.stack(velocity_profile).mean(axis=0)
    turbulence_profile = np.stack(turbulence_profile).mean(axis=0)

    # Pick out the mean velocity and turbulence intensity at hub height

    # Calculate grid spacing
    dy = yly / ny
    # Calculate y coordinates of cell centers
    y_coords = np.linspace(dy/2, yly - dy/2, ny)
    # Find index closest to hub height
    hub_idx = np.abs(y_coords - HUB_HEIGHT).argmin()

    hub_velocity = velocity_profile[hub_idx]
    hub_turbulence = turbulence_profile[hub_idx]

    print(f"Mean streamwise velocity at hub height: {hub_velocity:.2f} m/s")
    print(f"Turbulence intensity at hub height: {hub_turbulence:.1f}%")

    # plot mean velocity and turbulence intensity at each height
    y_values, expected_velocities = log_law()

    print(velocity_profile)
    print(turbulence_profile)
    fig = plot_abl_profiles(
        y_coords,
        velocity_profile,
        y_values,
        expected_velocities,
        turbulence_profile
    )

    fig.savefig(args.inflow_dir / 'inflow_stats.png', dpi=300)
