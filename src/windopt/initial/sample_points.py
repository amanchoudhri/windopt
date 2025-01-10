"""
Reproducible code to sample initial points used for each experiment using Latin
Hypercube Sampling.
"""
from pathlib import Path

import numpy as np

from scipy.stats.qmc import LatinHypercube

from windopt.constants import (
    SMALL_ARENA_DIMS, LARGE_ARENA_DIMS, SMALL_BOX_DIMS, LARGE_BOX_DIMS, PROJECT_ROOT
    )
from windopt.layout import Layout, save_layout_batch

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

def main(arena_setups):
    for config_name, arena_size, box_size, n_turbines in arena_setups:
        arena_xdim, _ = arena_size
        # sample 100 layouts to be evaluated using GCH
        gch_samples = sample(arena_xdim, n_turbines, n_samples=100)
        # 12 to be evaluated using large eddy simulations
        les_samples = sample(arena_xdim, n_turbines, n_samples=12)

        # create and add a uniform grid layout for baseline comparison
        ticks = np.linspace(0, arena_xdim, int(np.sqrt(n_turbines) + 1), endpoint=False)[1:]
        x_coords, z_coords = np.meshgrid(ticks, ticks)
        grid_layout = np.column_stack((x_coords.flatten(), z_coords.flatten()))
        grid_layout = grid_layout.reshape(1, n_turbines, 2)

        gch_samples = np.concatenate([gch_samples, grid_layout], axis=0)
        les_samples = np.concatenate([les_samples, grid_layout], axis=0)

        # package into Layout objects
        def make_layout(coords: np.ndarray) -> Layout:
            return Layout(coords=coords, system="arena", arena_dims=arena_size, box_dims=box_size)

        gch_layouts = [make_layout(layout) for layout in gch_samples]
        les_layouts = [make_layout(layout) for layout in les_samples]

        # save to file
        outdir = PROJECT_ROOT / "data" / "initial_points"
        outdir.mkdir(parents=True, exist_ok=True)

        save_layout_batch(gch_layouts, outdir / f"{config_name}_arena_gch_samples.npz")
        save_layout_batch(les_layouts, outdir / f"{config_name}_arena_les_samples.npz")

if __name__ == "__main__":
    ARENA_SETUPS = [
        # config name, arena size, box size, n_turbines
        ("small", SMALL_ARENA_DIMS, SMALL_BOX_DIMS, 4),
        ("large", LARGE_ARENA_DIMS, LARGE_BOX_DIMS, 16)
    ]
    main(ARENA_SETUPS)