import json

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional, Union

class Fidelity(StrEnum):
    GCH = "gch"
    LES = "les"
    
@dataclass
class AlternationPattern:
    sequence: list[Fidelity]
    
    @classmethod
    def from_counts(cls, n_gch: int, n_les: int) -> "AlternationPattern":
        """Generate an alternation pattern from the number of GCH and LES batches."""
        sequence = [Fidelity.GCH] * n_gch + [Fidelity.LES] * n_les
        return cls(sequence)
    
    def get_next_fidelity(self, batch_idx: int) -> Fidelity:
        """Get the next fidelity in the sequence."""
        return self.sequence[batch_idx % len(self.sequence)]
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "AlternationPattern":
        """Create an AlternationPattern from a dictionary configuration."""
        sequence = [Fidelity(fidelity) for fidelity in config_dict['sequence']]
        return cls(sequence)

class TrialGenerationStrategy(StrEnum):
    GCH_ONLY = "gch_only"
    LES_ONLY = "les_only"
    MULTI_ALTERNATING = "multi_alternating"
    MULTI_ADAPTIVE = "multi_adaptive"

@dataclass
class TrialGenerationConfig:
    strategy: TrialGenerationStrategy
    
    max_batches: Optional[int] = None

    les_batch_size: int = 5
    gch_batch_size: Optional[int] = None

    alternation_pattern: Optional[AlternationPattern] = None
    
    max_les_batches: Optional[int] = None
    max_gch_batches: Optional[int] = None

    def __post_init__(self):
        is_multi_fidelity = self.strategy in [
            TrialGenerationStrategy.MULTI_ALTERNATING,
            TrialGenerationStrategy.MULTI_ADAPTIVE
        ]
        if is_multi_fidelity and self.gch_batch_size is None:
            raise ValueError("GCH batch size must be specified for multi-fidelity strategies")

        if self.strategy == TrialGenerationStrategy.MULTI_ALTERNATING:
            if self.alternation_pattern is None:
                raise ValueError(
                    "Alternation pattern must be specified for multi-alternating strategy"
                )
                
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrialGenerationConfig":
        """Create a TrialGenerationConfig from a dictionary configuration."""
        if 'alternation_pattern' in config_dict and config_dict['alternation_pattern']:
            config_dict['alternation_pattern'] = AlternationPattern.from_dict(
                config_dict['alternation_pattern']
            )
        return cls(**config_dict)

@dataclass
class CampaignConfig:
    name: str
    n_turbines: int
    arena_dims: tuple[float, float]
    box_dims: tuple[float, float, float]
    trial_generation_config: TrialGenerationConfig
    debug_mode: bool = False
    les_config_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_turbines <= 0:
            raise ValueError("Number of turbines must be positive")
        
        if any(dim <= 0 for dim in self.arena_dims):
            raise ValueError("Arena dimensions must be positive")
            
        if any(dim <= 0 for dim in self.box_dims):
            raise ValueError("Box dimensions must be positive")
            
        if self.les_config_path is not None:
            path = Path(self.les_config_path)
            if not path.exists():
                raise ValueError(f"LES config file not found: {self.les_config_path}")
    
    def to_dict(self) -> dict:
        """Convert the config to a dictionary for serialization."""
        return {
            "name": self.name,
            "n_turbines": self.n_turbines,
            "arena_dims": self.arena_dims,
            "box_dims": self.box_dims,
            "trial_generation_config": self.trial_generation_config.__dict__,
            "debug_mode": self.debug_mode,
            "les_config_path": self.les_config_path
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the configuration to a JSON file."""
        path = Path(path) if isinstance(path, str) else path
        with path.open('w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'CampaignConfig':
        """Create a CampaignConfig from a dictionary configuration."""
        config_dict = config_dict.copy()  # Avoid modifying the input dict
        config_dict['trial_generation_config'] = TrialGenerationConfig.from_dict(
            config_dict['trial_generation_config']
        )
        return cls(**config_dict)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'CampaignConfig':
        """Load a CampaignConfig from a JSON file.
        
        Args:
            path: Path to the JSON configuration file
            
        Returns:
            CampaignConfig: The loaded configuration
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            JSONDecodeError: If the config file isn't valid JSON
        """
        path = Path(path) if isinstance(path, str) else path
        with path.open('r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
