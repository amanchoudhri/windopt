"""
Configuration management for WInc3D large-eddy simulations.

This module provides a type-safe configuration system for WInc3D simulations,
with support for JSON and YAML serialization.

Classes
-------
LESConfig : Main configuration class
    Master configuration for WInc3D simulations
FlowConfig : Flow configuration
    Configuration for flow parameters (precursor, synthetic, or custom)
NumericalConfig : Numerical parameters
    Time step and simulation duration settings
OutputConfig : Output settings
    Visualization and save intervals
InflowConfig : Inflow configuration
    Directory and timestep settings for inflow data
TurbineConfig : Turbine settings
    Layout and physical parameters for wind turbines
"""

from __future__ import annotations

import json
import yaml

from dataclasses import dataclass, field, asdict
from importlib import resources
from pathlib import Path
from typing import Literal, Optional, Any

import f90nml
import numpy as np

from yaml.representer import SafeRepresenter

from windopt.constants import D, DT, HUB_HEIGHT, N_STEPS_PRODUCTION, VIZ_INTERVAL_DEFAULT
from windopt.layout import Layout
from windopt.winc3d.io import read_turbine_information

FlowType = Literal["precursor", "synthetic", "custom"]
PathLike = Path | str

# Configure YAML serialization for Path objects
def _path_representer(dumper: SafeRepresenter, data: Path) -> yaml.ScalarNode:
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

yaml.SafeDumper.add_representer(Path, _path_representer)

@dataclass
class LESConfig:
    """Master configuration for WInc3D simulation."""
    box_dims: tuple[float, float, float]

    flow: FlowConfig = field(default_factory=lambda: FlowConfig())
    numerical: NumericalConfig = field(default_factory=lambda: NumericalConfig())
    output: OutputConfig = field(default_factory=lambda: OutputConfig())

    inflow: Optional[InflowConfig] = None
    turbines: Optional[TurbineConfig] = None

    def validate(self) -> None:
        """Ensure all configuration parameters are valid."""
        if self.flow.flow_type == "precursor" and self.inflow is None:
            raise ValueError("Inflow must be set for precursor flow type")

        if self.numerical.n_steps % self.output.viz_interval != 0:
            raise ValueError(
                f"Visualization interval ({self.output.viz_interval}) must evenly divide "
                f"number of timesteps ({self.numerical.n_steps})"
            )

    @property
    def n_outfiles(self) -> int:
        """Number of output files based on the number of timesteps and viz interval."""
        return self.numerical.n_steps // self.output.viz_interval

    def write_winc3d_files(
            self,
            output_dir: Path,
            in_filename: str = 'config.in',
            ad_filename: str = 'turbines.ad',
            ) -> None:
        """Write WInc3D input files to the specified directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._write_in_file(output_dir / in_filename, ad_filename)
        if self.turbines is not None:
            self._write_ad_file(output_dir / ad_filename)

    def _write_in_file(self, outfile: Path, ad_filename: Optional[str] = None) -> None:
        """Write the .in configuration file."""
        # Read base configuration
        cfg_ptr = resources.files('windopt').joinpath('config/les_base.in')
        with resources.as_file(cfg_ptr) as base_cfg_path:
            config = f90nml.read(base_cfg_path)
        
        # Flow parameters
        config["FlowParam"].update({
            "xlx": self.box_dims[0],
            "yly": self.box_dims[1],
            "zlz": self.box_dims[2],
            **self.flow.to_dict()
        })
        
        # Numerical parameters
        config["NumConfig"].update(self.numerical.to_dict())
        
        # Turbine configuration
        if self.turbines is not None:
            config["ADMParam"].update({
                "iadm": 1,
                "ADMcoords": ad_filename,
                "Ndiscs": self.turbines.layout.n_turbines,
            })
        
        # Inflow configuration
        if self.inflow.directory is not None:
            config["FileParam"].update({
                "InflowPath": f'{str(self.inflow.directory)}/',
                "NTimeSteps": self.inflow.n_timesteps,
                "NInflows": len(list(Path(self.inflow.directory).glob("inflow[1-9]*"))),
            })
        
        # Output configuration
        config["FileParam"]["imodulo"] = self.output.viz_interval

        # Save interval
        if self.output.save_interval is not None:
            save_interval = self.output.save_interval
        else:
            # Don't save at all, set to a value that will never be reached
            save_interval = self.numerical.n_steps + 1
        config["FileParam"]["isave"] = save_interval
        config["StatParam"]["spinup_time"] = self.output.spinup_time
        
        f90nml.write(config, outfile)

    def _write_ad_file(self, outfile: Path) -> None:
        """Write the turbine .ad file."""
        with open(outfile, "w+") as f:
            for location in self.turbines.layout.get_box_coords(self.box_dims):
                f.write(
                    f"{location[0]} {self.turbines.hub_height} {location[1]} "
                    f"1 0 0 {self.turbines.diameter}\n"
                )

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        config_dict = asdict(self)
        return _serialize_config(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LESConfig":
        """Create a LESConfig from a dictionary."""
        config_dict = _deserialize_config(config_dict.copy())
        
        return cls(
            box_dims=config_dict["box_dims"],
            flow=FlowConfig(**config_dict["flow"]),
            numerical=NumericalConfig(**config_dict["numerical"]),
            inflow=InflowConfig(**config_dict["inflow"]) if config_dict.get("inflow") else None,
            output=OutputConfig(**config_dict["output"]),
            turbines=TurbineConfig(**config_dict["turbines"]) if config_dict.get("turbines") else None
        )

    def to_json(self, json_file: PathLike) -> None:
        """Save configuration to a JSON file."""
        json_file = Path(json_file)
        config_dict = self.to_dict()

        with json_file.open("w") as f:
            json.dump(config_dict, f, indent=2, default=_serialize_config)

    @classmethod
    def from_json(cls, json_file: PathLike) -> "LESConfig":
        """Create a LESConfig from a JSON file."""
        json_file = Path(json_file)
        
        if not json_file.exists():
            raise FileNotFoundError(f"JSON configuration file not found: {json_file}")

        with json_file.open() as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

    def to_yaml(self, yaml_file: PathLike) -> None:
        """Save configuration to a YAML file."""
        yaml_file = Path(yaml_file)
        config_dict = self.to_dict()

        with yaml_file.open("w") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_file: PathLike) -> "LESConfig":
        """Create a LESConfig from a YAML file."""
        yaml_file = Path(yaml_file)

        if not yaml_file.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {yaml_file}")

        with yaml_file.open() as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

# Supporting configuration classes
@dataclass
class FlowConfig:
    """
    Configuration for flow parameters.
    
    Attributes
    ----------
    flow_type : FlowType
        Type of flow simulation (precursor, synthetic, or custom)
    reynolds_number : float
        Reynolds number for the simulation
    inflow_velocity : tuple[float, float]
        Min and max inflow velocities (u1, u2) in m/s
    turbulence_intensity : tuple[float, float]
        Turbulence intensity for (initial condition, inflow)
    """
    flow_type: FlowType = "precursor"
    reynolds_number: float = 10000
    
    inflow_velocity: tuple[float, float] = (10.0, 10.0)
    turbulence_intensity: tuple[float, float] = (0.125, 0.0)

    _itype: Optional[int] = field(default=None, repr=False)
    _iin: Optional[int] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Set itype and iin based on flow_type and validate configuration."""
        flow_types = {
            "precursor": (3, 3),
            "synthetic": (2, 2),
            "custom": (self._itype, self._iin)
        }
        
        if self.flow_type not in flow_types:
            raise ValueError(f"Unknown flow_type: {self.flow_type}")
            
        if self.flow_type == "custom" and (self._itype is None or self._iin is None):
            raise ValueError("Custom flow_type requires _itype and _iin to be set")
            
        self._itype, self._iin = flow_types[self.flow_type]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to WInc3D configuration dictionary."""
        config = {
            "itype": self._itype,
            "iin": self._iin,
            "re": self.reynolds_number,
        }
        
        if self.flow_type == "synthetic":
            config.update({
                "u1": self.inflow_velocity[0],
                "u2": self.inflow_velocity[1],
                "noise": self.turbulence_intensity[0],
                "noise1": self.turbulence_intensity[1],
            })
        
        return config

@dataclass
class NumericalConfig:
    """Configuration for numerical parameters."""
    dt: float = DT
    n_steps: int = N_STEPS_PRODUCTION
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to WInc3D configuration dictionary."""
        return {
            "dt": self.dt,
            "ilast": self.n_steps,
        }

@dataclass
class OutputConfig:
    """Configuration for simulation output."""
    viz_interval: int = VIZ_INTERVAL_DEFAULT
    save_interval: Optional[int] = None
    spinup_time: float = 1800.0  # Default 30 minutes
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to WInc3D configuration dictionary."""
        config = {
            "imodulo": self.viz_interval,
            "spinup_time": self.spinup_time
        }
        if self.save_interval is not None:
            config["isave"] = self.save_interval
        return config

@dataclass
class InflowConfig:
    """Configuration for inflow conditions."""
    directory: Path
    n_timesteps: int

@dataclass
class TurbineConfig:
    """Configuration for wind turbines."""
    layout: Layout
    diameter: float = D
    hub_height: float = HUB_HEIGHT
    
    @classmethod
    def from_ad_file(cls, ad_file: PathLike, arena_dims: tuple[float, float, float]) -> 'TurbineConfig':
        """Create a TurbineConfig from an .ad file."""
        turbines_df = read_turbine_information(Path(ad_file))
        layout = Layout(
            coords=turbines_df[['x', 'z']].values,
            arena_dims=arena_dims
        )
        return cls(
            layout=layout,
            diameter=turbines_df['D'].values[0],
            hub_height=turbines_df['y'].values[0]
        )

# Helper functions for serialization
def _serialize_config(obj: Any) -> Any:
    """Convert configuration objects to JSON/YAML-serializable types."""
    match obj:
        case Path():
            return str(obj)
        case np.ndarray():
            return obj.tolist()
        case tuple():
            return list(obj)
        case dict():
            return {k: _serialize_config(v) for k, v in obj.items()}
        case list():
            return [_serialize_config(v) for v in obj]
        case _:
            return obj

def _deserialize_config(config_dict: dict) -> dict:
    """Convert serialized data back to configuration types."""
    config_dict = config_dict.copy()
    
    # Handle inflow configuration
    if (inflow := config_dict.get("inflow")) is not None and isinstance(inflow, dict):
        if isinstance(inflow.get("directory"), str):
            inflow["directory"] = Path(inflow["directory"])
    
    # Handle turbine configuration
    if (turbines := config_dict.get("turbines")) is not None and isinstance(turbines, dict):
        if isinstance((layout := turbines.get("layout")), dict):
            if "coords" in layout:
                layout["coords"] = np.array(layout["coords"])
                turbines["layout"] = Layout(**layout)
    
    # Convert box_dims to tuple
    if "box_dims" in config_dict and isinstance(config_dict["box_dims"], (list, tuple)):
        config_dict["box_dims"] = tuple(config_dict["box_dims"])
    
    return config_dict