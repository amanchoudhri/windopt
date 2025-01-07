"""
Constants for the project.
"""
from pathlib import Path

PROJECT_ROOT=Path('/moto/home/ac4972/windopt/')

# NREL-5MW wind turbine diameter and hub height
D = 125.88 # meters
HUB_HEIGHT = 90.0 # meters

# arena dimensions (X, Z), configurable area where turbines can be placed
SMALL_ARENA_DIMS = (6 * D, 6 * D)
LARGE_ARENA_DIMS = (18 * D, 18 * D)

# box size (X, Y, Z) for large eddy simulations
SMALL_BOX_DIMS = (2004, 504, 1336)
LARGE_BOX_DIMS = (4008, 504, 3340)

# time step size for large eddy simulations
DT = 0.2

# mesh shape for large eddy simulations
MESH_SHAPE_20M = (100, 51, 72) # 20m horizontal resolution and small arena dims

INFLOW_20M = PROJECT_ROOT / 'simulations/small_arena_20m/planes'
INFLOW_20M_N_TIMESTEPS = 6000
