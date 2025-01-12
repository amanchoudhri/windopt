import sys
import traceback

from argparse import ArgumentParser

from windopt.constants import SMALL_BOX_DIMS, SMALL_ARENA_DIMS
from windopt.optim.config import CampaignConfig, TrialGenerationConfig, TrialGenerationStrategy
from windopt.optim.campaign import new_campaign, restart_campaign

def add_campaign_config_args(parser: ArgumentParser):
    """Define the command line arguments for campaign configuration."""
    parser.add_argument(
        "name",
        type=str,
        help="Name of the campaign",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[s.value for s in TrialGenerationStrategy],
        default=TrialGenerationStrategy.LES_ONLY.value,
        help="Trial generation strategy"
    )
    parser.add_argument(
        "--n-turbines",
        type=int,
        default=4,
        help="Number of turbines"
    )
    parser.add_argument(
        "--les-batch-size",
        type=int,
        default=5,
        help="Number of LES trials per batch"
    )
    parser.add_argument(
        "--gch-batch-size",
        type=int,
        help="Number of GCH trials per batch (required for multi-fidelity)"
    )
    parser.add_argument(
        "--gch-batches-per-les",
        type=int,
        help="Number of GCH batches per LES batch (required for alternating)"
    )
    parser.add_argument(
        "--max-les-batches",
        type=int,
        help="Maximum number of LES batches"
    )

def parse_args():
    parser = ArgumentParser(
        description="Run wind farm layout optimization campaign",
    )
    
    # Add debug mode at top level since it applies to both commands
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # New campaign command
    new_parser = subparsers.add_parser(
        'new',
        help='Start a new optimization campaign'
    )
    add_campaign_config_args(new_parser)
    
    # Restart campaign command
    restart_parser = subparsers.add_parser(
        'restart',
        help='Restart an existing optimization campaign'
    )
    restart_parser.add_argument(
        "name",
        type=str,
        help="Name of the campaign to restart"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    try:
        if args.command == 'new':
            strategy = TrialGenerationStrategy(args.strategy)
            trial_generation_config = TrialGenerationConfig(
                strategy=strategy,
                les_batch_size=args.les_batch_size,
                gch_batch_size=args.gch_batch_size,
                gch_batches_per_les_batch=args.gch_batches_per_les,
                max_les_batches=args.max_les_batches,
            )
            campaign_config = CampaignConfig(
                name=args.name,
                n_turbines=args.n_turbines,
                arena_dims=SMALL_ARENA_DIMS,
                box_dims=SMALL_BOX_DIMS,
                trial_generation_config=trial_generation_config,
                debug_mode=args.debug,
            )
            new_campaign(campaign_config)
        else:  # restart command
            restart_campaign(args.name)
            
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()