# windopt: Multi-Fidelity Wind Farm Layout Optimization

This project implements a multi-fidelity Bayesian optimization framework for wind farm layout design, combining inexpensive analytical wake models with high-fidelity large-eddy simulations (LES) to optimize turbine placement for maximum power output.

## Overview

The framework uses:
- High-fidelity: Large-eddy simulations via [WInc3D](https://github.com/imperialcollegelondon/winc3d/)
- Low-fidelity: Gauss-curl hybrid (GCH) wake model via [FLORIS](https://nrel.github.io/floris/)
- Optimization: Multi-task Bayesian optimization with batch expected improvement

## Requirements

- Python 3.11+
- FLORIS (for GCH wake model)
- WInc3D (for LES)
- GPyTorch (for Gaussian process modeling)
- BoTorch (for Bayesian optimization)
- Ax (for experiment management)

## Usage

The main optimization workflow consists of:

1. Running precursor simulations to generate inflow conditions
2. Evaluating candidate layouts using both GCH and LES models 
3. Fitting multi-task Gaussian processes to observations
4. Selecting new layouts via batch expected improvement

See the documentation for detailed usage instructions.

## Project Structure
```
.
├── campaigns/ # Optimization experiment results
├── data/ # Input data and initial samples
│ └── initial_trials/ # Results from initial trial runs
├── doc/
│ ├── report.pdf # Main technical report
├── img/ # Figures and visualizations
├── src/
│ └── windopt/ # Main package
│ │ ├── config/ # Base configuration files
│ │ ├── initial/ # Initial sampling routines
│ │ ├── optim/ # Optimization implementations
│ │ │ ├── multitype.py # Multi-fidelity BO
│ │ │ └── single_les.py # Single-fidelity BO
│ │ ├── viz/ # Visualization utilities
│ │ └── winc3d/ # WInc3D LES interface
```
