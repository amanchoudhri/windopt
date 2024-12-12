"""
Parse the log output from a LES simulation.
"""

from pathlib import Path
from typing import Union

import numpy as np

from matplotlib import pyplot as plt

def max_cfl(log_filename: Union[str, Path]) -> dict[str, float]:
    """
    Return the maximum CFL number along each direction from the log file.
    """

    # CFL lines are formatted as
    # CFL x-direction (Adv and Diff) =   n1_x,   n2_x
    # CFL y-direction (Adv and Diff) =   n1_y,   n2_y
    # etc

    def parse_line(line: str, max_cfl: dict[str, float]):
        """
        Parse a line of the log file and return the direction and the CFL number.
        """
        direction = line.split(' ')[1].split('-')[0]
        cfl = float(line.split('=')[1].split(',')[0])

        if cfl > max_cfl[direction]:
            max_cfl[direction] = cfl

    max_cfl = {'x': 0., 'y': 0., 'z':0.}

    # log files can be long, so we'll read them line by line
    with open(log_filename, 'r') as f:
        for line in f:
            if 'CFL' in line and 'direction' in line:
                parse_line(line, max_cfl)

    return max_cfl

def wind_history(log_filename: Union[str, Path]) -> dict[str, np.ndarray]:
    """
    Return the wind history for each direction from the log file.
    """

    def process(prev_line: str, curr_line: str) -> dict[str, np.ndarray]:
        """
        Process the previous and current line to get the wind history.
        """
        # This is the continuation line for max values
        prev_line = prev_line.strip() + ' ' + curr_line.strip()
        u_value, v_value, w_value = prev_line.split('=')[1].split()
        return {'u': float(u_value), 'v': float(v_value), 'w': float(w_value)}


    with open(log_filename, 'r') as f:
        # read in the file two lines at a time
        # Initialize lists to store the max and min values
        max_values = {'u': [], 'v': [], 'w': []}
        min_values = {'u': [], 'v': [], 'w': []}

        def _update(vals: dict[str, float], min: bool):
            for direction in ['u', 'v', 'w']:
                if min:
                    min_values[direction].append(vals[direction])
                else:
                    max_values[direction].append(vals[direction])

        prev_line = ""
        for line in f:
            if 'U,V,W max=' in prev_line:
                vals = process(prev_line, line)
                _update(vals, min=False)
            elif 'U,V,W min=' in prev_line:
                vals = process(prev_line, line)
                _update(vals, min=True)
            prev_line = line
    
    # convert to numpy arrays
    arrs = {}
    for direction in ['u', 'v', 'w']:
        arrs[direction] = np.vstack((np.array(max_values[direction]), np.array(min_values[direction])))

    return arrs

def friction_velocity_history(log_filename: Union[str, Path]) -> np.ndarray:
    """
    Return the friction velocity history from the log file.
    """
    # Friction velocity lines are formatted as
    #   Friction velocity :   0.405717494235521

    friction_velocity = []
    with open(log_filename, 'r') as f:
        for line in f:
            if 'Friction velocity' in line:
                friction_velocity.append(float(line.split(':')[1].strip()))

    return np.array(friction_velocity)


if __name__ == '__main__':
    old_log = Path('/Users/amanchoudhri/Downloads/winc3d_19679617.log')
    path = Path('/Users/amanchoudhri/Downloads/winc3d_19679858.log')
    project_base = Path('/Users/amanchoudhri/aman/classes/snr-1/bayes-opt/turbine/')
    # max_cfl = max_cfl(path)
    # print(max_cfl)
    wind_history = wind_history(path)
    friction_velocity = friction_velocity_history(path)
    # create time series plots of max wind value
    for direction in ['u', 'v', 'w']:
        plt.plot(wind_history[direction][0], label=f'{direction} max')
        plt.legend()
        plt.savefig(project_base / 'img' / f'{direction}_max.png')
        plt.show()

    plt.plot(friction_velocity, label='friction velocity')
    plt.legend()
    plt.savefig(project_base / 'img' / 'friction_velocity.png')
    plt.show()
