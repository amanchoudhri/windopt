import time

import numpy as np

from floris import FlorisModel

fmodel = FlorisModel('../config/gch_base.yaml')

# try with multiple wind directions and speeds
fmodel.set(
    wind_directions=[230, 250, 270, 290, 310],
    wind_speeds=[8.0, 9.0, 10.0, 11.0, 12.0],
    turbulence_intensities=[0.1, 0.1, 0.1, 0.1, 0.1],
)

# set yaw angles to 0
fmodel.set(
    yaw_angles=np.zeros((5, 3)),
)

start_time = time.time()

fmodel.run()

turbine_powers = fmodel.get_turbine_powers() / 1000.0
farm_power = fmodel.get_farm_power() / 1000.0

print(f'Ran FLORIS in {time.time() - start_time:.2f} seconds')
print(turbine_powers)
print(farm_power)

from floris.flow_visualization import visualize_cut_plane
import floris.layout_visualization as layoutviz

from matplotlib import pyplot as plt

fig, ax = plt.subplots()
horizontal_plane = fmodel.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=90.0,
    findex_for_viz=2
)
visualize_cut_plane(
    horizontal_plane,
    ax=ax,
    label_contours=False,
    title="Horizontal Flow with Turbine Rotors and labels",
)

print(fmodel.core.farm.yaw_angles)

# Plot the turbine rotors
layoutviz.plot_turbine_rotors(fmodel, wd=270, ax=ax, yaw_angles=np.zeros((5, 3)) + 20)
layoutviz.plot_turbine_labels(fmodel, ax=ax)

plt.show()
