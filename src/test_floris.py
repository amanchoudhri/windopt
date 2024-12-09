import time

from floris import FlorisModel

fmodel = FlorisModel('../config/gch_base.yaml')

# try with multiple wind directions and speeds
fmodel.set(
    wind_directions=[230, 250, 270, 290, 310],
    wind_speeds=[8.0, 9.0, 10.0, 11.0, 12.0],
    turbulence_intensities=[0.1, 0.1, 0.1, 0.1, 0.1],
)

start_time = time.time()

fmodel.run()

turbine_powers = fmodel.get_turbine_powers() / 1000.0
farm_power = fmodel.get_farm_power() / 1000.0

print(f'Ran FLORIS in {time.time() - start_time:.2f} seconds')
print(turbine_powers)
print(farm_power)
