from pathlib import Path

import numpy as np

from windopt.constants import PROJECT_ROOT

import matplotlib.pyplot as plt
import plotly.subplots
import plotly.graph_objects as go

from windopt.constants import HUB_HEIGHT, SMALL_BOX_DIMS, MESH_SHAPE_20M
from windopt.winc3d.io import read_turbine_locations


def vector_field_file(
        outdir: Path,
        filenumber: int,
        fileprefix: str
        ):
    allowable = ('ux', 'uy', 'uz', 'gammadisc', 'pp', 'vort')
    if fileprefix not in allowable:
        raise ValueError(f"Invalid file prefix: {fileprefix}. Must be one of {allowable}.")
    return outdir / f"{fileprefix}{filenumber:04d}"

def load_vector_field(
        file_path: Path,
        mesh_shape: tuple[int, int, int] = MESH_SHAPE_20M,
        ):
    with open(file_path, "rb") as f:
        uvec = np.fromfile(f, dtype=np.float64)

    return uvec.reshape(mesh_shape, order='F')


# also plot using plotly
def make_velocity_field_heatmap(
        hub_height_velocity_field: np.ndarray,
        min_speed: float,
        max_speed: float,
        grid_dims: tuple[float, float, float] = SMALL_BOX_DIMS,
        colorbar_position: float = 0.5,
        colorbar_len: float = 0.5,
        ) -> go.Figure:
    """
    Create a Heatmap displaying the velocity for a given timestep and direction.

    Expects a 2D array of shape (nz, nx)
    """

    xlx, _, zlz = grid_dims

    nz, nx = hub_height_velocity_field.shape

    x_coords = np.linspace(0, xlx, nx)
    z_coords = np.linspace(0, zlz, nz)

    return go.Heatmap(
        z=hub_height_velocity_field,
        x=x_coords,
        y=z_coords,
        colorbar=dict(title='m/s', y=colorbar_position, len=colorbar_len),  # Adjust vertical position
        zmin=min_speed,
        zmax=max_speed
        )

def plot_hub_height_vector_field(
        jobdir: Path,
        file_numbers: list[int],
        fileprefix: str,
        grid_dims: tuple[float, float, float] = SMALL_BOX_DIMS,
        mesh_shape: tuple[int, int, int] = MESH_SHAPE_20M,
        hub_height: float = HUB_HEIGHT,
        fig_width: int = 1000,
        ):
    xlx, yly, zlz = grid_dims
    nx, ny, nz = mesh_shape
    dy = yly / ny

    # hub_height_idx = int(hub_height / dy)
    hub_height_idx = 10
    # print(hub_height_idx)


    def _hub_height_vector_field(file_number: int):
        # transpose the velocity field to match the plotly heatmap
        velocity = load_vector_field(
            vector_field_file(jobdir / 'out', file_number, fileprefix),
            mesh_shape
        )[:, hub_height_idx].T
        return velocity

    fig = go.Figure()

    aspect_ratio = zlz / xlx

    fig.update_layout(
        autosize=False,
        width=fig_width,  # Width of each subplot
        height=fig_width * aspect_ratio,  # Total height for all three subplots
        margin=dict(t=50, b=50, l=50, r=50),  # Add margins to prevent cutoff
    )
    velocity_0 = _hub_height_vector_field(file_numbers[0])

    # get min and max values from the data
    min_val = velocity_0.min()
    max_val = velocity_0.max()

    for file_number in file_numbers:
        vector_field = _hub_height_vector_field(file_number)
        min_val = min(min_val, vector_field.min())
        max_val = max(max_val, vector_field.max())

    fig.add_trace(make_velocity_field_heatmap(velocity_0, min_val, max_val))

    # add x and y labels
    fig.update_xaxes(title='x [m]')
    fig.update_yaxes(title='z [m]')

    turbine_locations = read_turbine_locations(jobdir)

    fig.add_trace(go.Scatter(
        x=turbine_locations[:, 0],
        y=turbine_locations[:, 1],
        mode='markers',
        marker=dict(size=15, color='green'),
        name='Turbines'
    ))

    if len(file_numbers) > 1:
        frames = []
        for file_number in file_numbers:
            velocity = _hub_height_vector_field(file_number)
            frame = dict(
                data=[
                    make_velocity_field_heatmap(velocity, min_val, max_val)
                ],
                name=f't={file_number}'
            )
            frames.append(frame)

        fig.frames = frames

        # Add animation settings and slider
        # Display each frame for 0.1 seconds
        fig.update_layout(
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=100, redraw=True),
                            fromcurrent=True,
                            mode='immediate'
                        )],
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )],
                    )
                ],
                x=0.1,  # Position buttons on the left
                y=1.15,  # Position above the plots
                xanchor='right',
                yanchor='top',
            )],
            sliders=[dict(
                currentvalue=dict(
                    prefix='Time Step: ',
                    visible=True,
                    xanchor='right'
                ),
                pad=dict(t=0),
                len=0.9,  # Length of the slider
                x=0.1,    # Position slider to the right of buttons
                y=1.15,   # Same height as buttons
                xanchor='left',
                yanchor='top',
                steps=[dict(
                    args=[[f't={k}'],
                        dict(frame=dict(duration=0, redraw=True),
                            mode='immediate')],
                    label=str(k),
                    method='animate',
                ) for k in file_numbers]
            )],
            # Adjust top margin to make room for controls
            margin=dict(t=100, b=50, l=50, r=50),
        )
    return fig

def plot_umean(jobdir: Path):
    umean = load_vector_field(jobdir / 'out' / 'umean.dat', mesh_shape=MESH_SHAPE_20M)

    xlx, yly, zlz = SMALL_BOX_DIMS
    nx, ny, nz = MESH_SHAPE_20M

    print(umean.shape)

    x_coords = np.linspace(0, xlx, nx)
    z_coords = np.linspace(0, zlz, nz)

    dy = yly / ny
    hub_height_idx = int(HUB_HEIGHT / dy)

    return go.Heatmap(
        z=umean[:, hub_height_idx].T,
        x=x_coords,
        y=z_coords,
    )

def inst_and_mean_velocity_field(jobdir: Path, file_number: int):
    umean = load_vector_field(jobdir / 'out' / 'umean.dat', mesh_shape=MESH_SHAPE_20M) / 36000
    uinst = load_vector_field(
        vector_field_file(jobdir / 'out', file_number, 'ux'),
        mesh_shape=MESH_SHAPE_20M
        )

    min_val = min(umean.min(), uinst.min())
    max_val = max(umean.max(), uinst.max())

    xlx, yly, zlz = SMALL_BOX_DIMS
    nx, ny, nz = MESH_SHAPE_20M

    x_coords = np.linspace(0, xlx, nx)
    z_coords = np.linspace(0, zlz, nz)

    dy = yly / ny
    hub_height_idx = int(HUB_HEIGHT / dy)

    # plot the instantaneous and mean fields side by side
    aspect_ratio = zlz / xlx

    figwidth = 800
    figheight = figwidth * aspect_ratio

    fig = plotly.subplots.make_subplots(
        rows=1,
        cols=2,
        column_widths=[figwidth, figwidth],
        row_heights=[figheight],
        subplot_titles=['Instantaneous streamwise velocity', 'Mean streamwise velocity'],
        horizontal_spacing=0.07
        )

    fig.update_layout(
        width=2.1 * figwidth,
        height=figheight,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    fig.add_trace(go.Heatmap(
        z=uinst[:, hub_height_idx].T, x=x_coords, y=z_coords, zmin=min_val, zmax=max_val,
        colorbar=dict(
            title='m/s',
            y=0.5,  # Center the colorbar vertically
            len=1.0  # Make colorbar span full height
        ),
        showscale=True  # Show colorbar for first plot
        ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=umean[:, hub_height_idx].T,
        x=x_coords,
        y=z_coords,
        zmin=min_val,
        zmax=max_val,
        showscale=False
        ),
        row=1, col=2)

    # add turbine locations
    turbine_locations = read_turbine_locations(jobdir)
    def _add_turbines(fig, col: int):
        fig.add_trace(go.Scatter(
            x=turbine_locations[:, 0],
            y=turbine_locations[:, 1],
            mode='markers',
            marker=dict(size=15, color='green'),
            name='Turbines',
            showlegend=False
        ), row=1, col=col)

    _add_turbines(fig, 1)
    _add_turbines(fig, 2)

    fig.update_xaxes(title='x [m]', row=1, col=1)
    fig.update_xaxes(title='x [m]', row=1, col=2)
    fig.update_yaxes(title='z [m]', row=1, col=1)
    fig.update_yaxes(title='z [m]', row=1, col=2)

    fig.show()

    # fig.write_html(PROJECT_ROOT / 'img' / 'ux_and_umean.html')
    fig.write_image(PROJECT_ROOT / 'img' / 'ux_and_umean.png')

    return fig


if __name__ == "__main__":
    GRID = 'grid'
    RANDOM = 'random'

    use_precursor = False
    layout = RANDOM

    dirname = f'validate_les_{layout}'
    if not use_precursor:
        dirname += '_no_precursor'

    jobdir = Path(f'/moto/edu/users/ac4972/validate_les/{dirname}/')

    # jobdir = Path('/moto/home/ac4972/windopt/simulations/validate_les_grid_20241230_194106')

    # # plot a 30 minute evolution of the streamwise velocity field
    # file_numbers = list(range(0, 12))
    # fig = plot_hub_height_vector_field(jobdir, file_numbers, 'ux')
    # # title the figure, but add a good amount of space below the title since we have the slider
    # fig.update_layout(title='Streamwise velocity field', title_x=0.5, title_y=0.98)
    # fig.show()

    # # save the figure
    # fig.write_html(PROJECT_ROOT / 'img' / 'velocity_field.html')

    # fig = go.Figure()
    # fig.add_trace(plot_umean(jobdir))
    # fig.show()

    file_number = 30
    inst_and_mean_velocity_field(jobdir, file_number)