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

DEFAULT_DT = 0.2 # seconds

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
                f"Expected nt * ny * ny * components = {total_expected}, "
                f"but received {len(data)}."
                )

    # Reshape into components
    data = data.reshape(N_COMPONENTS, nt, ny, nz)
    return data

def calculate_abl_profiles(inflow, u_star, z0, kappa=0.4):
    """Calculate ABL profiles from inflow data."""
    # Domain setup
    # Calculate grid points
    dy = yly / ny
    y_coords = np.linspace(dy/2, yly - dy/2, ny)
    
    # Calculate mean velocity profile
    u_mean = np.mean(inflow[0], axis=(0, 2))  # Average over time and z
    u_mean_normalized = u_mean / u_star
    
    # Calculate theoretical log law profile
    y_log = np.logspace(np.log10(10), np.log10(500), 100)  # From 10m to 500m
    u_log = (1/kappa) * np.log(y_log/z0)
    
    # Calculate turbulence intensity profile
    # Standard deviation at each height
    u_std = np.std(inflow[0], axis=(0, 2))
    TI = (u_std / u_mean) * 100  # Convert to percentage

def inflow_statistics(inflow: np.ndarray, n, dims):
    output = []
    # Calculate some basic statistics
    components_name = ['u', 'v', 'w']
    for i, name in enumerate(components_name):
        output += [
            f"{name} component statistics:",
            f"Min: {inflow[i].min()}",
            f"Max: {inflow[i].max()}",
            f"Mean: {inflow[i].mean()}",
            f"Std: {inflow[i].std()}\n",
        ]

    nx, ny, nz = n
    xlx, yly, zlz = dims

    # Now calculate mean hub-height velocity and turbulence intensity

    # Calculate grid spacing
    dy = yly / ny

    # Calculate y coordinates of cell centers
    y_coords = np.linspace(dy/2, yly - dy/2, ny)

    # Find index closest to hub height (90m)
    hub_height = 90
    hub_idx = np.abs(y_coords - hub_height).argmin()

    # Extract u_x velocity component at hub height for all time and z
    # inflow[0] is u_x component, shape is (nt, ny, nz)
    u_x_hub = inflow[0, :, hub_idx, :]

    # Calculate mean velocity at hub height
    u_x_bar = np.mean(u_x_hub)

    # Calculate turbulence intensity
    # First get velocity fluctuations
    u_x_fluctuations = u_x_hub - u_x_bar

    # Calculate standard deviation and normalize by mean velocity
    sigma_u = np.std(u_x_fluctuations)
    TI = sigma_u / u_x_bar * 100  # Convert to percentage

    output += [
        f"Mean streamwise velocity at hub height: {u_x_bar:.2f} m/s",
        f"Turbulence intensity at hub height: {TI:.1f}%\n",
        ]

    output = "\n".join(output)

    # Plot convergence if matplotlib is available
    return u_x_bar, TI, u_x_hub.mean(axis=-1), output

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("inflow_dir", type=Path)
    p.add_argument("--arena", type=str, required=True)
    p.add_argument("--nt", type=int, required=True)
    p.add_argument("--dt", type=float, required=False, default=DEFAULT_DT)

    args = p.parse_args()

    if args.arena == "small":
        n = (50, 51, 36)          # gridpoints
        dims = (2004, 504, 1336)  # arena dimensions

    elif args.arena == "large":
        n = (100, 51, 84)
        dims = (4008, 504, 3340)

    elif args.arena == "large_tall":
        n = (100, 75, 84)
        dims = (4008, 750, 3340)

    _, ny, nz = n

    running_means = []

    output = []
    inflow_paths = list(args.inflow_dir.glob("inflow[0-9]*"))
    n_inflows = len(inflow_paths)

    get_num = lambda path: int(path.name.replace("inflow", ""))

    ux_bars = []
    TIs = []

    for inflow_path in sorted(inflow_paths, key=get_num):
        i = get_num(inflow_path)
        output.append(f'Inflow {i} --- {inflow_path}')
        inflow = parse_inflow(inflow_path, nt=args.nt, ny=ny, nz=nz)
        u_x_bar, TI, means, out = inflow_statistics(inflow, n, dims)
        ux_bars.append(u_x_bar)
        TIs.append(TI)
        output.append(out)
        running_means.append(means)

    print(np.mean(ux_bars))
    print(np.mean(TIs))

    # make a new directory under img
    project_root = Path(__file__).parent.parent.parent
    outdir = project_root / 'img' / f'precursor_{arena}'

    # with open(args.inflow_dir / 'inflow_stats.txt', 'w+') as f:
    #     f.write("\n".join(output))

    running_means = np.concatenate(running_means, axis=0)
    time = np.arange(0, len(running_means)) * args.dt
    plt.figure(figsize=(10, 5))
    plt.scatter(time, running_means, s = 0.1)
    plt.xlabel('Time step')
    plt.ylabel('Running mean of u_x velocity')
    plt.title('Convergence of Mean Velocity at Hub Height')
    plt.grid(True)
    plt.savefig(args.inflow_dir / 'running_means.png')
