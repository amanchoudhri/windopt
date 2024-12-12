"""
Main project entrypoint.
"""
from pathlib import Path

import torch
import botorch

import numpy as np
import pandas as pd

from ax import Data
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from ax.storage.json_store.save import save_experiment
from ax.utils.notebook.plotting import render

from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.models.cost import FixedCostModel
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize

from gch import gch

# Setup the cost model ---
# Assume a fixed cost per fidelity for now. In the future, this can be extended
# with multiple fidelities for the large-eddy simulation using the
# AffineFidelityCostModel class.

# A GCH simulation takes 0.01 seconds to run.
GCH_COST = 0.01
# A LES (with 30min spinup and 2hr averaging window) takes 1.25 hours to run.
LES_COST = 1.25 * 60 * 60

COSTS = torch.Tensor([GCH_COST, LES_COST])
cost_model = FixedCostModel(COSTS)

cost_aware_utility = InverseCostWeightedUtility(cost_model)

multi_fidelity_generation_strategy = GenerationStrategy(
    steps = [
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            model_kwargs={
                'surrogate': Surrogate(
                    SingleTaskMultiFidelityGP,
                    model_options={
                        'train_Yvar': 1e-6,
                    }
                ),
                'botorch_acqf_class': qMultiFidelityKnowledgeGradient,
            },
            model_gen_kwargs={
                'acqf_kwargs': {
                    'num_fantasies': 8,
                    'cost_aware_utility': cost_aware_utility,
                }
            }
        )
    ]
)

# setup the search space

D = 126

X_DIM = 18 * D
Y_DIM = 750
Z_DIM = 18 * D


THETA_MIN = -40 # degrees
THETA_MAX = 40 # degrees

def turbine_parameters(
        N_turbines: int,
        optimize_angles: bool = False
        ) -> list[dict]:
    """
    Create a list of parameters to pass to ax_client.create_experiment.
    """
    params = []
    for i in range(1, N_turbines + 1):
        params.append({
            'name': f'x{i}',
            'type': 'range',
            'bounds': [0, X_DIM],
        })
        params.append({
            'name': f'z{i}',
            'type': 'range',
            'bounds': [0, Z_DIM],
        })
        if optimize_angles:
            params.append({
                'name': f'theta{i}',
                'type': 'range',
                'bounds': [THETA_MIN, THETA_MAX],
            })
    return params

def create_multi_fidelity_experiment(
        experiment_name: str,
        N_turbines: int,
        optimize_angles: bool = False,
        ) -> AxClient:
    """
    Create a multi-fidelity experiment with the given name and number of turbines.
    """
    params = turbine_parameters(N_turbines, optimize_angles=optimize_angles)
    params.append({
        'name': 'fidelity',
        'type': 'range',
        'value_type': 'int',
        'bounds': [0, 1],
        'is_fidelity': True,
        'target_value': 1,
    })
    ax_client = AxClient(generation_strategy=multi_fidelity_generation_strategy)
    ax_client.create_experiment(
        name=experiment_name,
        parameters=params,
        objectives = {"power": ObjectiveProperties(minimize=False)},
    )
    return ax_client

def evaluate(parameterization) -> dict:
    """
    Evaluate the power output of the turbines at the given parameterization.
    """
    n_turbines = len([k for k in parameterization.keys() if k.startswith('x')])

    # locations is a 2D array of shape (n_turbines, 2)
    locations = np.array([
        [parameterization[f'x{i}'] for i in range(1, n_turbines + 1)],
        [parameterization[f'z{i}'] for i in range(1, n_turbines + 1)]
    ]).T
    # orientations is a 2D array of shape (1, n_turbines)
    orientations = np.zeros((1, n_turbines))
    if 'theta1' in parameterization:
        orientations = np.array(
            [parameterization[f'theta{i}'] for i in range(1, n_turbines + 1)]
        ).reshape(1, -1)

    powers = gch(locations, orientations)
    # add random gaussian noise to the lower fidelity power
    # power is roughly on the order of 1-3 megawatts/turbine
    # so add noise on the order of 0.1 megawatts
    fidelity = parameterization['fidelity']
    powers += (1 - fidelity) * np.random.normal(0, 0.1e6, size=powers.shape)
    return {'power': (powers.sum(), 0.0)}

def run(
    ax_client: AxClient,
    budget: int,
    experiment_directory: Path,
    ):
    """
    Run the experiment for the given budget (in seconds).
    """
    spent = 0

    # while spent < budget:
    for i in range(10):
        parameters, trial_index = ax_client.get_next_trial()
        spent += LES_COST if parameters['fidelity'] else GCH_COST

        value = evaluate(parameters)
        ax_client.complete_trial(trial_index=trial_index, raw_data=value)

        df = exp_to_df(ax_client.experiment)
        df.to_csv(experiment_directory / 'experiment.csv')
    
    save_experiment(ax_client.experiment, str(experiment_directory / 'experiment.json'))

    best_parameters, values = ax_client.get_best_parameters()
    print(best_parameters)

    ax_client.get_contour_plot(
        param_x='x1', param_y='z1'
    )


if __name__ == '__main__':
    OUTDIR = Path('/Users/amanchoudhri/aman/classes/snr-1/bayes-opt/turbine/campaigns')
    # quick test experiment, 10 second budget
    experiment_name = 'test_noiseless_mf_setup'
    experiment_directory = OUTDIR / experiment_name
    experiment_directory.mkdir(parents=True, exist_ok=True)

    ax_client = create_multi_fidelity_experiment(
        experiment_name=experiment_name,
        N_turbines=3,
        optimize_angles=False,
    )
    # load initial data so we don't need the Sobol generation step
    # df = pd.read_csv(OUTDIR / 'test_mf_setup' / 'experiment.csv')

    run(ax_client, budget=5 * 60 * 60, experiment_directory=experiment_directory)
