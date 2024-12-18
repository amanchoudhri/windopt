from argparse import ArgumentParser
from typing import Union
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from windopt.analyze_precursor import parse_inflow


def make_heatmap(
        inflow: np.ndarray,
        timestep: int,
        direction: int,
        arena_dims: tuple[float, float, float],
        colorbar_position: float,
        colorbar_len: float,
        ) -> go.Heatmap:
    """
    Create a Heatmap displaying the velocity for a given timestep and direction.
    """
    min_speed = inflow[direction].min()
    max_speed = inflow[direction].max()

    _, yly, zlz = arena_dims
    ny, nz = inflow.shape[2:]

    y_coords = np.linspace(0, 1, ny) * yly
    z_coords = np.linspace(0, 1, nz) * zlz

    return go.Heatmap(
        z=inflow[direction,timestep,:,:],
        x=z_coords,
        y=y_coords,
        colorbar=dict(title='m/s', y=colorbar_position, len=colorbar_len),  # Adjust vertical position
        zmin=min_speed,
        zmax=max_speed
        )

def setup_fig(
        arena_dims: tuple[float, float, float],
        base_fig_width: int = 800
        ) -> go.Figure:
    """
    Setup a figure with the correct aspect ratio and margins.
    """
    _, yly, zlz = arena_dims
    aspect_ratio = yly / zlz

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('u velocity', 'v velocity', 'w velocity'),
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )
    fig.update_layout(
        autosize=False,
        width=base_fig_width,  # Width of each subplot
        height=base_fig_width * aspect_ratio * 2.5,  # Total height for all three subplots
        margin=dict(t=50, b=50, l=50, r=50),  # Add margins to prevent cutoff
    )
    return fig

def plot_inflow(
        inflow: np.ndarray,
        timesteps: list[int],
        arena_dims: tuple[float, float, float],
        ):
    # Create figure with secondary y-axis
    fig = setup_fig(arena_dims)

    # Setup traces
    colorbar_locations = (0.883, 0.52, 0.15)
    colorbar_len = 0.33

    for direction in range(3):
        row = direction + 1
        fig.add_trace(
            make_heatmap(
                inflow, timesteps[0], direction, arena_dims,
                colorbar_locations[direction], colorbar_len
            ),
            row=row, col=1
        )
        fig.update_xaxes(
            title_text="z [m]",
            row=row,
            col=1,
            constrain='domain',
        )
        fig.update_yaxes(
            title_text="y [m]",
            row=row,
            col=1,
            constrain='domain',
            )

    # If multiple timesteps, create frames for animation
    if len(timesteps) > 1:
        frames = []
        for t in timesteps:
            frame = dict(
                data=[
                    make_heatmap(inflow, t, direction, arena_dims,
                                 colorbar_locations[direction], colorbar_len)
                    for direction in range(3)
                ],
                name=f't={t}'
            )
            frames.append(frame)

        fig.frames = frames

        # Add animation settings and slider
        fig.update_layout(
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=1, redraw=True),
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
                ) for k in range(len(frames))]
            )],
            # Adjust top margin to make room for controls
            margin=dict(t=100, b=50, l=50, r=50),
        )
    return fig

def main(
    inflow_path: str,
    arena_dims: tuple[float, float, float],
    grid_dims: tuple[int, int, int],
    n_timesteps: int,
    initial_timestep: int,
    end_timestep: Union[int, None] = None,
    output_path: Union[str, None] = None,
    ):
    inflow = parse_inflow(
        inflow_path,
        nt=n_timesteps,
        ny=grid_dims[1],
        nz=grid_dims[2],
    )
    timesteps = (
        list(range(initial_timestep, end_timestep))
        if end_timestep else [initial_timestep]
    )
    fig = plot_inflow(inflow, timesteps, arena_dims)

    if output_path:
        fig.write_html(output_path)

    fig.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('inflow_path', type=str)
    parser.add_argument('--arena', type=str)
    parser.add_argument('--n_timesteps', type=int)
    parser.add_argument('--initial_timestep', type=int)
    parser.add_argument('--end_timestep', type=int, default=None)
    parser.add_argument('--output_path', type=str, default=None)

    args = parser.parse_args()

    if args.arena == "small_10m":
        grid_dims = (200, 51, 144)
        arena_dims = (2004, 504, 1336)

    elif args.arena == "small_20m":
        grid_dims = (50, 51, 36)
        arena_dims = (2004, 504, 1336)

    elif args.arena == "small_40m":
        grid_dims = (50, 51, 36)
        arena_dims = (2004, 504, 1336)

    elif args.arena == "large":
        grid_dims = (100, 51, 84)
        arena_dims = (4008, 504, 3340)

    elif args.arena == "large_tall":
        grid_dims = (100, 75, 84)
        arena_dims = (4008, 750, 3340)

    main(
        args.inflow_path,
        arena_dims,
        grid_dims,
        args.n_timesteps,
        args.initial_timestep,
        args.end_timestep,
        args.output_path,
    )
