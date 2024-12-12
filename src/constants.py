"""
Constants for the project.
"""

# NREL-5MW wind turbine diameter and hub height
D = 125.88 # meters
HUB_HEIGHT = 90.0 # meters

# arena dimensions (X, Z), configurable area where turbines can be placed
SMALL_ARENA_DIMS = (6 * D, 6 * D)
LARGE_ARENA_DIMS = (18 * D, 18 * D)

# box size (X, Y, Z) for large eddy simulations
SMALL_BOX_DIMS = (2004, 504, 1336)
LARGE_BOX_DIMS = (4008, 504, 3340)