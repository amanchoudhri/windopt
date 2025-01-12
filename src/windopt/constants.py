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

# mesh shape for large eddy simulations
MESH_SHAPE_20M = (101, 51, 72) # 20m horizontal resolution and small arena dims

INFLOW_20M = PROJECT_ROOT / 'simulations/small_arena_20m/planes'
INFLOW_20M_N_TIMESTEPS = 6000

# Simulation time parameters
DT = 0.2  # Timestep in seconds
N_STEPS_PRODUCTION = 45000  # 2.5 hours of simulation time
N_STEPS_DEBUG = 2000  # Short run for testing

# Visualization parameters 
VIZ_INTERVAL_DEFAULT = 9000  # Default timesteps between outputs (30 minutes if DT = 0.2)
VIZ_INTERVAL_FREQUENT = int(60 / DT)  # Output every simulated minute