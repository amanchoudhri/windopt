"""
Configuration management for WInc3D large-eddy simulations.
"""
import json

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional, Any
from importlib import resources

import f90nml
import numpy as np

from windopt.constants import D, DT, HUB_HEIGHT, N_STEPS_PRODUCTION, VIZ_INTERVAL_DEFAULT
from windopt.layout import Layout
from windopt.winc3d.io import read_turbine_information

FlowType = Literal["precursor", "synthetic", "custom"]

@dataclass
class FlowConfig:
    """
    Configuration for flow parameters.
    
    Parameters
    ----------
    flow_type : str = "precursor"
        Type of flow simulation. Options:
        - "precursor": Uses precursor planes (itype=3, iin=3) - Recommended
        - "synthetic": Uses synthetic inflow (itype=2, iin=2)
        - "custom": Allows manual setting of itype and iin
    reynolds_number: float = 10000
        Reynolds number for the simulation
    
    For synthetic inflow only (flow_type="synthetic"):
    inflow_velocity: tuple[float, float] = (10.0, 10.0)
        Min and max inflow velocities (u1, u2) in m/s
    turbulence_intensity: tuple[float, float] = (0.125, 0.0)
        Turbulence intensity for (initial condition, inflow)
    """
    flow_type: FlowType = "precursor"
    reynolds_number: float = 10000
    
    # Parameters for synthetic inflow
    inflow_velocity: tuple[float, float] = (10.0, 10.0)
    turbulence_intensity: tuple[float, float] = (0.125, 0.0)
    
    # Advanced parameters for custom flow types
    _itype: Optional[int] = field(default=None, repr=False)
    _iin: Optional[int] = field(default=None, repr=False)
    
    def __post_init__(self):
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
    
    def to_dict(self) -> dict:
        """Convert to dictionary for f90nml."""
        config = {
            "itype": self._itype,
            "iin": self._iin,
            "re": self.reynolds_number,
        }
        
        # Only include synthetic inflow parameters when not using precursor planes
        if self.flow_type == "synthetic":
            config.update({
                "u1": self.inflow_velocity[0],
                "u2": self.inflow_velocity[1],
                "noise": self.turbulence_intensity[0],
                "noise1": self.turbulence_intensity[1],
            })
        
        return config

@dataclass
class TurbineConfig:
    """Configuration for wind turbines."""
    layout: Layout
    diameter: float = D
    hub_height: float = HUB_HEIGHT
    
    @classmethod
    def from_ad_file(cls, ad_file: Path, arena_dims: tuple[float, float, float]) -> "TurbineConfig":
        """
        Create a TurbineConfig from an .ad file.
        """
        turbines_df = read_turbine_information(ad_file)
        layout = Layout(
            coords=turbines_df[['x', 'z']].values,
            arena_dims=arena_dims
        )
        return cls(
            layout=layout,
            diameter=turbines_df['D'].values[0],
            hub_height=turbines_df['y'].values[0]
        )

@dataclass
class InflowConfig:
    """Configuration for inflow conditions."""
    directory: Path
    n_timesteps: int

@dataclass
class OutputConfig:
    """
    Configuration for simulation output.
    
    Parameters
    ----------
    viz_interval : int
        Number of timesteps between visualization outputs (imodulo)
    spinup_time : float
        Time in seconds to wait before collecting statistics
    save_interval : Optional[int]
        Number of timesteps between full flow field saves (isave)
        If None, flow fields are not saved
    """
    viz_interval: int = VIZ_INTERVAL_DEFAULT
    save_interval: Optional[int] = None
    spinup_time: float = 1800.0  # Default 30 minutes
    
    def to_dict(self) -> dict:
        """Convert to dictionary for f90nml."""
        config = {
            "imodulo": self.viz_interval,
            "isave": self.save_interval,
            "spinup_time": self.spinup_time
        }
        if self.save_interval is not None:
            config["isave"] = self.save_interval
        return config

@dataclass
class NumericalConfig:
    """
    Configuration for numerical parameters.
    
    Parameters
    ----------
    dt : float
        Time step size in seconds
    n_steps : int
        Number of time steps to simulate
    """
    dt: float = DT
    n_steps: int = N_STEPS_PRODUCTION
    
    def to_dict(self) -> dict:
        """Convert to dictionary for f90nml."""
        return {
            "dt": self.dt,
            "ilast": self.n_steps,
        }

@dataclass
class LESConfig:
    """Master configuration for WInc3D simulation."""
    box_dims: tuple[float, float, float]

    flow: FlowConfig = field(default_factory=FlowConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    inflow: Optional[InflowConfig] = None
    turbines: Optional[TurbineConfig] = None

    def validate(self) -> None:
        """Ensure all configuration parameters are valid."""
        # make sure that if flow_type is precursor then inflow is set
        if self.flow.flow_type == "precursor":
            if self.inflow is None:
                raise ValueError("Inflow must be set for precursor flow type")

        # make sure that the output viz interval divides the number of timesteps
        if self.numerical.n_steps % self.output.viz_interval != 0:
            raise ValueError(
                f"Visualization interval ({self.output.viz_interval}) must evenly divide "
                f"number of timesteps ({self.numerical.n_steps})"
            )

    @property
    def n_outfiles(self) -> int:
        """Number of output files based on the number of timesteps and viz interval."""
        return self.numerical.n_steps // self.output.viz_interval

    def to_json(self, json_file: Path) -> None:
        """
        Save configuration to a JSON file.
        
        Parameters
        ----------
        json_file : Path
            Path to save the JSON configuration
        """
        def _config_encoder(obj: Any) -> Any:
            """Convert special types to JSON-serializable objects."""
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        config_dict = asdict(self)

        with open(json_file, "w") as f:
            json.dump(config_dict, f, indent=2, default=_config_encoder)

    @classmethod
    def from_json(cls, json_file: Path) -> "LESConfig":
        """
        Create a LESConfig from a JSON file.
        
        Parameters
        ----------
        json_file : Path
            Path to JSON configuration file
        
        Returns
        -------
        LESConfig
            Configuration loaded from JSON
        """
        with open(json_file) as f:
            config_dict = json.load(f)
        
        # Convert paths back to Path objects
        if config_dict.get("inflow", {}).get("directory"):
            config_dict["inflow"]["directory"] = Path(config_dict["inflow"]["directory"])
        
        # Convert layout dict to Layout object
        if config_dict.get("turbines", {}).get("layout", {}).get("coords"):
            config_dict["turbines"]["layout"]["coords"] = np.array(
                config_dict["turbines"]["layout"]["coords"]
            )
            config_dict["turbines"]["layout"] = Layout(**config_dict["turbines"]["layout"])

        # Reconstruct nested dataclass objects
        return cls(
            box_dims=tuple(config_dict["box_dims"]),
            flow=FlowConfig(**config_dict["flow"]),
            numerical=NumericalConfig(**config_dict["numerical"]),
            inflow=InflowConfig(**config_dict["inflow"]),
            output=OutputConfig(**config_dict["output"]),
            turbines=TurbineConfig(**config_dict["turbines"]) if config_dict.get("turbines") else None
        ) 

    def write_winc3d_files(
            self,
            output_dir: Path,
            in_filename: str = 'config.in',
            ad_filename: str = 'turbines.ad',
            ) -> None:
        """
        Write WInc3D input files to the specified directory.
        
        Parameters
        ----------
        output_dir : Path
            The directory to write the configuration files to.
        in_filename : str
            The name of the .in configuration file.
        ad_filename : str
            The name of the .ad turbine file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write WInc3D config files
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