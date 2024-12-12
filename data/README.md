# data

Expected directory structure:

```
data/
├── initial_points/ # initial layouts for GCH and LES
│   ├── small_arena_gch_samples.npy
│   ├── small_arena_les_samples.npy
│   ├── large_arena_gch_samples.npy
│   ├── large_arena_les_samples.npy
├── initial_trials/ # evaluated configurations from initial_points
│   ├── small_arena_gch_trials.npy
│   ├── small_arena_les_trials.npy
│   ├── large_arena_gch_trials.npy
│   ├── large_arena_les_trials.npy
├── precursor_planes/ # precursor planes for LES
│   ├── small_arena/
│   │   ├── inflow0000
│   │   ├── inflow0001
│   │   ├── ...
│   ├── large_arena/
│   │   ├── inflow0000
│   │   ├── inflow0001
│   │   ├── ...
```
