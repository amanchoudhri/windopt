# windopt: Multi-Fidelity Wind Farm Layout Optimization

A Python framework for optimizing wind farm turbine layouts using multi-fidelity Bayesian optimization. Combines inexpensive analytical wake models with high-fidelity large-eddy simulations (LES) to efficiently optimize turbine placement for maximum power output.

## Overview

Wind farm layout optimization is crucial for maximizing power output.
Gold-standard approaches using high-fidelity simulations are computationally expensive,
and approximate solutions using cheap alternatives like analytic wake models may fail to capture the
complex dynamics of the flow through wind farms.
This project implements a multi-fidelity optimization approach that:

- Uses the Gauss-curl hybrid (GCH) wake model as a fast approximation
- Combines it with accurate large-eddy simulations (LES)
- Employs multi-task Bayesian optimization to efficiently trade off between fidelities
- Supports batch parallel evaluation of layouts

Current results show the multi-fidelity approach can find near-optimal layouts with significantly fewer expensive LES evaluations compared to single-fidelity methods.

## Requirements

- Python ≥ 3.10
- Ax Platform ≥ 0.4.0 (Experiment management)
- FLORIS ≥ 4.2.1 (GCH wake model)
- WInc3D (LES simulations)
- f90nml ≥ 1.4.4 (Fortran namelist handling)
- seaborn ≥ 0.13.2 (Visualization)
- kaleido == 0.2.1 (Static plot export)

WInc3D must be installed separately following instructions at [WInc3D repository](https://github.com/imperialcollegelondon/winc3d/).

## Usage

Optimization campaigns are run through a command-line interface:

```bash
# Start a new single-fidelity campaign using only LES
python -m windopt.optim.cli new my-campaign \
    --strategy les-only \
    --n-turbines 4 \
    --les-batch-size 5 \
    --max-les-batches 20

# Start a new multi-fidelity campaign alternating between GCH and LES
python -m windopt.optim.cli new my-mf-campaign \
    --strategy multi-alternating \
    --n-turbines 4 \
    --les-batch-size 4 \
    --gch-batch-size 50 \
    --gch-batches-per-les 1 \
    --max-les-batches 20

# Resume an existing campaign
python -m windopt.optim.cli restart my-campaign
```

> **Note**: Custom LES configuration support is still under development. Currently using default WInc3D settings as described in the technical report.

The main optimization workflow consists of:

1. Running precursor simulations to generate inflow conditions
2. Evaluating candidate layouts using both GCH and LES models 
3. Fitting multi-task Gaussian processes to observations
4. Selecting new layouts via batch expected improvement

```
.
├── campaigns/     # Optimization experiment results
├── config/       # Configuration files
├── data/         # Input data and initial samples
├── doc/          # Documentation and reports
├── src/
│   └── windopt/  # Main package
│       ├── config/     # Configuration handling
│       ├── initial/    # Initial sampling routines
│       ├── optim/      # Optimization implementations
│       ├── viz/        # Visualization utilities
│       └── winc3d/     # WInc3D LES interface
```

## Current Status & Limitations

- Currently supports optimization of up to 4 turbines as proof of concept
- Handles single wind direction/speed (extension to varying conditions planned)
- Uses fixed simulation durations (adaptive durations planned)
- Requires manual alternation between fidelities (automatic selection planned)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Prof. Nikos Bempedelis (Queen Mary University of London) for guidance on LES configuration
- Prof. John Cunningham (Columbia University) for Bayesian optimization guidance
- Columbia University's Terremoto computing cluster for computational resources

## Visualization Tools

The package includes several visualization utilities for analyzing wind farm simulations and optimization results:

```bash
# Plot instantaneous and mean velocity fields, where the instantanous field is taken 1 hour after spinup
python -m windopt.viz.velocity_flow run_dir 1.0 --save_path flow.png

# Create animated visualization of velocity field evolution
python -m windopt.viz.vector_field run_dir \
    --field ux \
    --start 0 \
    --n_steps 100 \
    --save anim.html
```

Key visualization features:
- Interactive velocity field animations using Plotly
- Power convergence analysis for different layouts
- Precursor simulation validation (velocity profiles, turbulence intensity)
- Optimization campaign progress tracking
- Support for both static plots and animated visualizations
- Export to PNG/HTML formats

The visualizations were used to generate all figures in the technical report and are designed to help analyze:
- Flow field development through wind farms
- Wake interactions between turbines
- Statistical convergence of simulations
- Optimization campaign performance

See the technical report for example visualizations and detailed analysis.
