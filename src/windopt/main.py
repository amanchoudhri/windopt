"""
Main project entrypoint.
"""
from collections import defaultdict
from pathlib import Path

import torch
import botorch

import numpy as np
import pandas as pd

from ax import (
    Data, SearchSpace, RangeParameter, ParameterType, Metric, Runner, Trial, TrialStatus
    )
from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.core.multi_type_experiment import MultiTypeExperiment

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

from ax.models.torch.botorch_modular.surrogate import Surrogate

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df

from ax.storage.json_store.save import save_experiment

from ax.utils.common.result import Ok, Err
from ax.utils.notebook.plotting import render

from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.models.cost import FixedCostModel
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize

from windopt.gch import gch
from windopt.constants import D, SMALL_ARENA_DIMS
from windopt.winc3d import run_les, process_results

PROJECT_ROOT = Path(__file__).parent.parent.parent

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

# multi_fidelity_generation_strategy = GenerationStrategy(
#     steps = [
#         GenerationStep(
#             model=Models.BOTORCH_MODULAR,
#             num_trials=-1,
#             model_kwargs={
#                 'surrogate': Surrogate(
#                     SingleTaskMultiFidelityGP,
#                     model_options={
#                         'train_Yvar': 1e-6,
#                     }
#                 ),
#                 'botorch_acqf_class': qMultiFidelityKnowledgeGradient,
#             },
#             model_gen_kwargs={
#                 'acqf_kwargs': {
#                     'num_fantasies': 8,
#                     'cost_aware_utility': cost_aware_utility,
#                 }
#             }
#         )
#     ]
# )

# setup the search space

THETA_MIN = -40.0 # degrees
THETA_MAX = 40.0 # degrees

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
            'bounds': [0.0, SMALL_ARENA_DIMS[1]],
        })
        params.append({
            'name': f'z{i}',
            'type': 'range',
            'bounds': [0.0, SMALL_ARENA_DIMS[1]],
        })
        if optimize_angles:
            params.append({
                'name': f'theta{i}',
                'type': 'range',
                'bounds': [THETA_MIN, THETA_MAX],
            })
    return params

def setup_search_space(
        N_turbines: int,
        optimize_angles: bool = False,
        ) -> tuple[list[str], SearchSpace]:
    """
    Create a search space for the given number of turbines and optimization parameters.

    Returns:
        param_names: list[str]
        search_space: SearchSpace
    """
    params = []
    x_param_names = [f'x{i}' for i in range(1, N_turbines + 1)]
    z_param_names = [f'z{i}' for i in range(1, N_turbines + 1)]

    if optimize_angles:
        theta_param_names = [f'theta{i}' for i in range(1, N_turbines + 1)]
    else:
        theta_param_names = []
    
    param_names = x_param_names + z_param_names + theta_param_names

    for i in range(N_turbines):
        params.append(RangeParameter(
            name=x_param_names[i],
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=SMALL_ARENA_DIMS[1],
        ))
        params.append(RangeParameter(
            name=z_param_names[i],
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=SMALL_ARENA_DIMS[1],
        ))
        if optimize_angles:
            params.append(RangeParameter(
                name=theta_param_names[i],
                parameter_type=ParameterType.FLOAT,
                lower=THETA_MIN,
                upper=THETA_MAX,
            ))
    return param_names, SearchSpace(parameters=params)


class GCHMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            parameterization = arm.parameters
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

            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": float(powers.sum()),
                "sem": 0.0, # Since GCH is deterministic
            }

            )
        return Data(df=pd.DataFrame(records))

def locations_from_ax(parameterization: dict) -> np.ndarray:
    """
    Parse the locations into an array from the Ax parameterization.

    Returns:
        locations: np.ndarray of shape (n_turbines, 2)
    """
    n_turbines = len([k for k in parameterization.keys() if k.startswith('x')])
    return np.array([
        [parameterization[f'x{i}'] for i in range(1, n_turbines + 1)],
        [parameterization[f'z{i}'] for i in range(1, n_turbines + 1)]
    ]).T

class LESRunner(Runner):
    def __init__(self, campaign_name: str):
        super().__init__()
        self.campaign_name = campaign_name

    def run(self, trial) -> dict[str]:
        """
        Deploys a LES simulation for the given trial.
        """
        if not isinstance(trial, Trial):
            raise ValueError("This runner only handles `Trial`.")

        run_name = f'{self.campaign_name}_trial_{trial.index}'

        locations = locations_from_ax(trial.arm.parameters)
        job = run_les(
            run_name,
            locations,
            debug_mode=True,
        )
        return {'job': job}

    def poll_trial_status(self, trials: list[Trial]) -> dict[TrialStatus, set[int]]:
        statusdict = defaultdict(set)

        SLURM_TO_AX = {
            'PENDING': TrialStatus.RUNNING,
            'RUNNING': TrialStatus.RUNNING,
            'COMPLETED': TrialStatus.COMPLETED,
            'FAILED': TrialStatus.FAILED,
        }

        for trial in trials:
            slurm_state = trial.run_metadata['job'].status()
            ax_state = SLURM_TO_AX[slurm_state]
            statusdict[ax_state].add(trial.index)

        return statusdict


class LESMetric(Metric):
    def fetch_trial_data(self, trial):
        """
        Obtains data from a LES simulation for a given trial.
        """
        if not isinstance(trial, Trial):
            raise ValueError("This metric only handles `Trial`.")

        try:
            job = trial.run_metadata['job']
            power = process_results(job)

            df_dict = {
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": power,
                "sem": 0.0, # Choose to model LESs as deterministic for now
            }
            return Ok(Data(df=pd.DataFrame.from_records([df_dict])))

        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )

def create_multi_fidelity_experiment(
        experiment_name: str,
        N_turbines: int,
        optimize_angles: bool = False,
        ) -> AxClient:
    """
    Create a multi-fidelity experiment with the given name and number of turbines.
    """
    param_names, search_space = setup_search_space(N_turbines, optimize_angles=optimize_angles)

    gch_metric = GCHMetric(name='power')
    les_metric = LESMetric(name='power')
    optimization_config = OptimizationConfig(
        objective=ObjectiveProperties(minimize=False),
        metric=LESMetric
    )

    # ax_client = AxClient(generation_strategy=multi_fidelity_generation_strategy)
    # ax_client.create_experiment(
    #     name=experiment_name,
    #     parameters=params,
    #     objectives = {"power": ObjectiveProperties(minimize=False)},
    # )
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
    print(locations)
    print(powers)
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

def layout_to_params(layout: np.ndarray) -> dict:
    """
    Convert a layout to Ax parameters.
    Parameters:
        layout: np.ndarray of shape (N_turbines, 2) containing x,z coordinates
    Returns:
        dict: Combined dictionary of x and z parameters
    """
    return {
        f'{dim}{i+1}': val 
        for i, row in enumerate(layout)
        for dim, val in zip(['x', 'z'], row)
    }

def load_initial_data(ax_client: AxClient, fidelity: str = 'gch'):
    """
    Load initial trial data into the AxClient.
    """
    if fidelity not in ['gch', 'les']:
        raise ValueError(f"Fidelity must be 'gch' or 'les', not {fidelity}")

    trial_points = np.load(PROJECT_ROOT / 'data' / 'initial_points' / f'small_arena_{fidelity}_samples.npy')
    trial_values = np.load(PROJECT_ROOT / 'data' / 'initial_trials' / f'small_arena_{fidelity}_trials.npy')

    for (layout, power) in zip(trial_points, trial_values):
        # convert layout to x, z params for Ax
        layout_params = layout_to_params(layout)
        _, trial_index = ax_client.attach_trial(
            parameters = {
                **layout_params,
                'fidelity': 0 if fidelity == 'gch' else 1,
            }
        )
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={'power': (power, 0.0)}
        )


if __name__ == '__main__':
    outdir = PROJECT_ROOT / 'campaigns'
    # quick test experiment, 10 second budget
    experiment_name = 'test_noiseless_mf_setup'
    experiment_directory = outdir / experiment_name
    experiment_directory.mkdir(parents=True, exist_ok=True)

    ax_client = create_multi_fidelity_experiment(
        experiment_name=experiment_name,
        N_turbines=4,
        optimize_angles=False,
    )

    # load initial data
    load_initial_data(ax_client, fidelity='gch')
    load_initial_data(ax_client, fidelity='les')

    run(ax_client, budget=5 * 60 * 60, experiment_directory=experiment_directory)
