"""
Randomly select a subset of the GCH trials and assess their performance
with a large-eddy simulation.
"""

import argparse
from pathlib import Path

import pandas as pd

from windopt.optim.config import CampaignConfig, TrialGenerationStrategy
from windopt.optim.trial import layout_from_ax_params, _create_les_config
from windopt.viz.campaign import _preprocess_trials
from windopt.winc3d.les import start_les


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "campaign_dir",
        type=Path,
        help="Path to the campaign directory",
    )
    parser.add_argument(
        "--batch-modulo",
        type=int,
        default=5,
        help="Select the best trial from every Nth batch",
    )
    return parser.parse_args()

def load_campaign_config(campaign_dir: Path) -> CampaignConfig:
    """Load and validate campaign configuration."""
    config_path = campaign_dir / 'campaign_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Campaign config not found at {config_path}")
    
    campaign_config = CampaignConfig.from_json(config_path)
    
    gen_strategy = campaign_config.trial_generation_config.strategy
    if gen_strategy != TrialGenerationStrategy.GCH_ONLY:
        raise ValueError(f"Generation method is not GCH only! Encountered: {gen_strategy}")
    
    return campaign_config

def select_best_trials(trials_df: pd.DataFrame, batch_modulo: int) -> pd.DataFrame:
    """Select the best performing trial from each Nth batch."""
    filtered_trials = trials_df[trials_df["batch_index"] % batch_modulo == 0]
    return filtered_trials.groupby('batch_index').apply(
        lambda x: x.loc[x['power'].idxmax()]
    )

def main() -> None:
    args = parse_arguments()

    campaign_config = load_campaign_config(args.campaign_dir)
    
    # Load and preprocess trials
    trials_path = args.campaign_dir / 'trials.csv'
    if not trials_path.exists():
        raise FileNotFoundError(f"Trials file not found at {trials_path}")
    
    trials = _preprocess_trials(pd.read_csv(trials_path), campaign_config)
    best_trials = select_best_trials(trials, args.batch_modulo)
    
    # Queue LES jobs
    for _, trial in best_trials.iterrows():
        layout = layout_from_ax_params(campaign_config, dict(trial))
        config = _create_les_config(campaign_config, layout)
        
        job = start_les(
            run_name=f"{campaign_config.name}_batch_{trial['batch_index']}",
            config=config
        )
        print(
            f"Queued LES for batch {trial['batch_index']}, job ID: {job.slurm_job_id}"
        )

if __name__ == "__main__":
    main()
