import json
import yaml
import numpy as np
from pathlib import Path
import pytest
import copy

from windopt.winc3d.config import (
    LESConfig,
    FlowConfig,
    TurbineConfig,
    InflowConfig,
    OutputConfig,
    NumericalConfig,
)
from windopt.layout import Layout

@pytest.fixture
def sample_config() -> LESConfig:
    """Create a sample configuration for testing."""
    layout = Layout(
        coords=np.array([[100.0, 100.0], [300.0, 300.0]]),
        arena_dims=(500.0, 500.0)
    )
    
    return LESConfig(
        box_dims=(500.0, 500.0, 500.0),
        flow=FlowConfig(
            flow_type="precursor",
            reynolds_number=12000,
        ),
        numerical=NumericalConfig(
            dt=0.5,
            n_steps=1000,
        ),
        output=OutputConfig(
            viz_interval=100,
            spinup_time=1800.0,
        ),
        inflow=InflowConfig(
            directory=Path("/path/to/inflow"),
            n_timesteps=1000,
        ),
        turbines=TurbineConfig(
            layout=layout,
            diameter=126.0,
            hub_height=90.0,
        ),
    )

def test_config_validation(sample_config: LESConfig):
    """Test configuration validation."""
    # Valid configuration should not raise
    sample_config.validate()
    
    # Test precursor flow type without inflow
    invalid_config = copy.deepcopy(sample_config)
    invalid_config.inflow = None
    with pytest.raises(ValueError, match="Inflow must be set for precursor flow type"):
        invalid_config.validate()
    
    # Test invalid viz_interval
    invalid_config = copy.deepcopy(sample_config)
    invalid_config.flow.flow_type = "synthetic"  # Change to synthetic to avoid inflow requirement
    invalid_config.inflow = None
    invalid_config.output.viz_interval = 77  # Doesn't divide n_steps (1000)
    with pytest.raises(ValueError, match="Visualization interval .* must evenly divide"):
        invalid_config.validate()

def test_json_serialization(sample_config: LESConfig, tmp_path: Path):
    """Test JSON serialization and deserialization."""
    json_file = tmp_path / "config.json"
    
    # Test serialization
    sample_config.to_json(json_file)
    assert json_file.exists()
    
    # Verify JSON content
    with json_file.open() as f:
        json_data = json.load(f)
    assert json_data["box_dims"] == list(sample_config.box_dims)
    assert json_data["flow"]["reynolds_number"] == sample_config.flow.reynolds_number
    assert json_data["inflow"]["directory"] == str(sample_config.inflow.directory)
    
    # Test deserialization
    loaded_config = LESConfig.from_json(json_file)
    assert loaded_config.box_dims == sample_config.box_dims
    assert loaded_config.flow.reynolds_number == sample_config.flow.reynolds_number
    assert loaded_config.inflow.directory == sample_config.inflow.directory
    np.testing.assert_array_equal(
        loaded_config.turbines.layout.coords,
        sample_config.turbines.layout.coords
    )

def test_yaml_serialization(sample_config: LESConfig, tmp_path: Path):
    """Test YAML serialization and deserialization."""
    yaml_file = tmp_path / "config.yaml"
    
    # Test serialization
    sample_config.to_yaml(yaml_file)
    assert yaml_file.exists()
    
    # Verify YAML content
    with yaml_file.open() as f:
        yaml_data = yaml.safe_load(f)
    assert yaml_data["box_dims"] == list(sample_config.box_dims)
    assert yaml_data["flow"]["reynolds_number"] == sample_config.flow.reynolds_number
    assert yaml_data["inflow"]["directory"] == str(sample_config.inflow.directory)
    
    # Test deserialization
    loaded_config = LESConfig.from_yaml(yaml_file)
    assert loaded_config.box_dims == sample_config.box_dims
    assert loaded_config.flow.reynolds_number == sample_config.flow.reynolds_number
    assert loaded_config.inflow.directory == sample_config.inflow.directory
    np.testing.assert_array_equal(
        loaded_config.turbines.layout.coords,
        sample_config.turbines.layout.coords
    )

def test_file_not_found():
    """Test handling of missing files."""
    with pytest.raises(FileNotFoundError):
        LESConfig.from_json(Path("nonexistent.json"))
    
    with pytest.raises(FileNotFoundError):
        LESConfig.from_yaml(Path("nonexistent.yaml"))

def test_dict_conversion(sample_config: LESConfig):
    """Test dictionary conversion methods."""
    # Convert to dict
    config_dict = sample_config.to_dict()
    assert isinstance(config_dict, dict)
    
    # Check that box_dims is converted to list
    assert isinstance(config_dict["box_dims"], list)
    assert config_dict["box_dims"] == list(sample_config.box_dims)
    
    # Convert back from dict
    loaded_config = LESConfig.from_dict(config_dict)
    assert loaded_config.box_dims == sample_config.box_dims
    assert loaded_config.flow.reynolds_number == sample_config.flow.reynolds_number
    np.testing.assert_array_equal(
        loaded_config.turbines.layout.coords,
        sample_config.turbines.layout.coords
    )

def test_optional_fields():
    """Test configuration with optional fields omitted."""
    minimal_config = LESConfig(
        box_dims=(500.0, 500.0, 500.0),
        flow=FlowConfig(flow_type="synthetic"),  # No inflow needed for synthetic
        numerical=NumericalConfig(),  # Add required fields
        output=OutputConfig(),
    )
    
    # Convert to and from dict
    config_dict = minimal_config.to_dict()
    loaded_config = LESConfig.from_dict(config_dict)
    
    assert loaded_config.inflow is None
    assert loaded_config.turbines is None
    assert loaded_config.flow.flow_type == "synthetic" 