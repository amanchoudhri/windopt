from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

CoordSystem = Literal["arena", "box"]


@dataclass(frozen=True)
class Layout:
    """Represents a wind farm layout with coordinate system safety.
    
    Args:
        coords: Array of shape (n_turbines, 2) containing x,z coordinates
        system: Which coordinate system the coordinates are in
        arena_dims: (x,z) dimensions of the arena
        box_dims: (x,y,z) dimensions of the simulation box
    """
    coords: npt.NDArray[np.float64]
    system: Literal["arena", "box"]
    arena_dims: tuple[float, float]
    box_dims: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Validate the layout coordinates."""
        if not isinstance(self.coords, np.ndarray):
            raise TypeError("coords must be a numpy array")
        
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError(
                f"coords must be of shape (n_turbines, 2), got {self.coords.shape}"
            )
        
        if self.system not in ("arena", "box"):
            raise ValueError(f"Invalid coordinate system: {self.system}")
        
        self._check_bounds()

    def _check_bounds(self) -> None:
        """Check if the coordinates are within the layout bounds."""
        if self.system == "arena":
            bounds = self.arena_dims
        else:
            bounds = (self.box_dims[0], self.box_dims[2])

        valid = ((0 <= self.coords) & (self.coords <= bounds)).all()

        if not valid:
            raise ValueError("Coordinates must be within the layout bounds")

    @property
    def n_turbines(self) -> int:
        """Number of turbines in the layout."""
        return self.coords.shape[0]

    @property
    def _offsets(self) -> npt.NDArray[np.float64]:
        """Calculate the offset between box and arena coordinates."""
        x_offset = (self.box_dims[0] - self.arena_dims[0]) / 2
        z_offset = (self.box_dims[2] - self.arena_dims[1]) / 2
        return np.array([x_offset, z_offset])

    @property
    def arena_coords(self) -> npt.NDArray[np.float64]:
        """Get coordinates in arena system."""
        if self.system == "arena":
            return self.coords
        
        return self.coords - self._offsets

    @property
    def box_coords(self) -> npt.NDArray[np.float64]:
        """Get coordinates in box system."""
        if self.system == "box":
            return self.coords
        
        return self.coords + self._offsets


def save_layout_batch(layouts: list[Layout], outpath: Path) -> None:
    """Save a batch of layouts to a numpy compressed archive."""
    # Convert all fields into single arrays
    coords = np.stack([layout.coords for layout in layouts])
    systems = np.array([layout.system for layout in layouts])
    arena_dims = np.array([layout.arena_dims for layout in layouts])
    box_dims = np.array([layout.box_dims for layout in layouts])
    
    np.savez(
        outpath,
        n_layouts=len(layouts),
        coords=coords,
        systems=systems,
        arena_dims=arena_dims,
        box_dims=box_dims
    )

def load_layout_batch(path: Path) -> list[Layout]:
    """Load a batch of layouts from a numpy compressed archive."""
    data = np.load(path)
    n_layouts = int(data['n_layouts'])
    
    return [
        Layout(
            coords=data['coords'][i],
            system=str(data['systems'][i]),
            arena_dims=tuple(data['arena_dims'][i]),
            box_dims=tuple(data['box_dims'][i])
        )
        for i in range(n_layouts)
    ]
