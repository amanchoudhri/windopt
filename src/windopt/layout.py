from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

CoordSystem = Literal["arena", "box"]


@dataclass(frozen=True)
class Layout:
    """
    Represents turbine positions in the arena coordinate system.
    
    Args:
        coords: Array of shape (n_turbines, 2) containing x,z coordinates
        arena_dims: (x,z) dimensions of the arena
    """
    coords: npt.NDArray[np.float64]
    arena_dims: tuple[float, float]
    
    def __post_init__(self) -> None:
        """Validate the layout coordinates."""
        if not isinstance(self.coords, np.ndarray):
            raise TypeError("coords must be a numpy array")
        
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError(
                f"coords must be of shape (n_turbines, 2), got {self.coords.shape}"
            )
        
        # Validate bounds
        valid = ((0 <= self.coords) & (self.coords <= self.arena_dims)).all()
        if not valid:
            raise ValueError("Coordinates must be within the arena bounds")

    @property
    def n_turbines(self) -> int:
        """Number of turbines in the layout."""
        return self.coords.shape[0]

    def get_box_coords(self, box_dims: tuple[float, float, float]) -> npt.NDArray[np.float64]:
        """Convert arena coordinates to box coordinates."""
        x_offset = (box_dims[0] - self.arena_dims[0]) / 2
        z_offset = (box_dims[2] - self.arena_dims[1]) / 2
        return self.coords + np.array([x_offset, z_offset])


def save_layout_batch(layouts: list[Layout], outpath: Path) -> None:
    """Save a batch of layouts to a numpy compressed archive."""
    coords = np.stack([layout.coords for layout in layouts])
    arena_dims = np.array([layout.arena_dims for layout in layouts])
    
    np.savez(
        outpath,
        n_layouts=len(layouts),
        coords=coords,
        arena_dims=arena_dims
    )

def load_layout_batch(path: Path) -> list[Layout]:
    """Load a batch of layouts from a numpy compressed archive."""
    data = np.load(path)
    n_layouts = int(data['n_layouts'])
    
    return [
        Layout(
            coords=data['coords'][i],
            arena_dims=tuple(data['arena_dims'][i])
        )
        for i in range(n_layouts)
    ]
