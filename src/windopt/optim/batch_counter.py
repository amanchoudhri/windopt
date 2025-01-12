from dataclasses import dataclass

import pandas as pd

from windopt.optim.config import Fidelity

@dataclass
class BatchCounts:
    gch: int = 0
    les: int = 0
    
    @classmethod
    def from_previous_trials(cls, previous_trials: pd.DataFrame) -> "BatchCounts":
        """Initialize batch counts from any previous trials."""
        if previous_trials.empty:
            return cls()
        return cls(
            gch=len(previous_trials[previous_trials['fidelity'] == 'gch']),
            les=len(previous_trials[previous_trials['fidelity'] == 'les']),
        )
    
    @property
    def total(self) -> int:
        return self.gch + self.les
    
    def increment(self, fidelity: Fidelity):
        match fidelity:
            case Fidelity.GCH:
                self.gch += 1
            case Fidelity.LES:
                self.les += 1
            case _:
                raise ValueError(f"Unknown fidelity: {fidelity}")