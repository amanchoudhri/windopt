import json

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional

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
    def from_json(cls, path: Path) -> "AlternationPattern":
        with open(path, 'r') as f:
            config_dict = json.load(f)
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
    def from_json(cls, path: Path) -> "TrialGenerationConfig":
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config_dict['alternation_pattern'] = AlternationPattern.from_json(
            Path(config_dict['alternation_pattern'])
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
    
    @classmethod
    def from_json(cls, path: Path) -> 'CampaignConfig':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config_dict['trial_generation_config'] = TrialGenerationConfig.from_json(
            Path(config_dict['trial_generation_config'])
        )
        return cls(**config_dict)
