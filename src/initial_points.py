"""
Reproducible code to sample initial points used for each experiment using Latin
Hypercube Sampling.
"""
from pathlib import Path

import numpy as np

from scipy.stats.qmc import LatinHypercube

SEED = 2024

def sample(arena_size: int, n_turbines: int, n_samples: int) -> np.ndarray:
    """
    Sample initial points for a given arena size and number of turbines.
    """
    # x, z coordinates for each turbine
    n_dimensions = 2 * n_turbines
    sampler = LatinHypercube(d=n_dimensions, seed=SEED)
    unscaled_samples = sampler.random(n=n_samples)
    # reshape to (n_samples, n_turbines, 2)
    unscaled_samples = unscaled_samples.reshape(n_samples, n_turbines, 2)
    return unscaled_samples * arena_size


if __name__ == "__main__":
    # small arena: 6D by 6D arena, 4 turbines.
    # large arena: 18D by 18D arena, 16 turbines.
    ARENA_NAMES = ["small", "large"]
    ARENA_SIZES = [6, 18]
    N_TURBINES = [4, 16]
    for arena_name, arena_size, n_turbines in zip(ARENA_NAMES, ARENA_SIZES, N_TURBINES):
        # sample 100 layouts to be evaluated using GCH
        gch_samples = sample(arena_size, n_turbines, n_samples=100)
        # 12 to be evaluated using large eddy simulations
        les_samples = sample(arena_size, n_turbines, n_samples=12)
        # to both, add a uniformly spaced grid layout
        ticks = np.linspace(0, arena_size, int(np.sqrt(n_turbines) + 1), endpoint=False)[1:]
        # Create a meshgrid of x,z coordinates
        x_coords, z_coords = np.meshgrid(ticks, ticks)
        # Stack and reshape into (1, n_turbines, 2) array
        grid_layout = np.column_stack((x_coords.flatten(), z_coords.flatten()))
        grid_layout = grid_layout.reshape(1, n_turbines, 2)
        # concatenate to the end of gch and les samples
        gch_samples = np.concatenate([gch_samples, grid_layout], axis=0)
        les_samples = np.concatenate([les_samples, grid_layout], axis=0)

        # save to file
        project_dir = Path(__file__).parent.parent
        outdir = project_dir / "data" / "initial_points"
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Save sample points
        np.save(
            outdir / f"{arena_name}_arena_gch_samples.npy",
            gch_samples
        )
        np.save(
            outdir / f"{arena_name}_arena_les_samples.npy",
            les_samples
        )
