from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class TrialGenerationStrategy(StrEnum):
    GCH_ONLY = "gch_only"
    LES_ONLY = "les_only"
    MULTI_ALTERNATING = "multi_alternating"
    MULTI_ADAPTIVE = "multi_adaptive"

@dataclass
class TrialGenerationConfig:
    strategy: TrialGenerationStrategy
    les_batch_size: int = 5
    gch_batch_size: Optional[int] = None
    gch_batches_per_les_batch: Optional[int] = None

    max_les_batches: Optional[int] = None

    def __post_init__(self):
        is_multi_fidelity = self.strategy in [
            TrialGenerationStrategy.MULTI_ALTERNATING,
            TrialGenerationStrategy.MULTI_ADAPTIVE
        ]
        if is_multi_fidelity and self.gch_batch_size is None:
            raise ValueError("GCH batch size must be specified for multi-fidelity strategies")

        if self.strategy == TrialGenerationStrategy.MULTI_ALTERNATING:
            if self.gch_batches_per_les_batch is None:
                raise ValueError(
                    "GCH batches per LES batch must be specified for multi-alternating strategy"
                )

@dataclass
class CampaignConfig:
    name: str
    n_turbines: int
    arena_dims: tuple[float, float]
    box_dims: tuple[float, float, float]
    trial_generation_config: TrialGenerationConfig
    debug_mode: bool = False