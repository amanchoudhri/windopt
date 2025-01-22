import sys
import traceback
from pathlib import Path

from argparse import ArgumentParser

from windopt.constants import SMALL_BOX_DIMS, SMALL_ARENA_DIMS
from windopt.optim.config import (
    CampaignConfig, 
    TrialGenerationConfig, 
    TrialGenerationStrategy,
    AlternationPattern
)
from windopt.optim.campaign import new_campaign, restart_campaign

def add_campaign_config_args(parser: ArgumentParser):
    """Define the command line arguments for campaign configuration."""
    # Group related arguments
    basic_group = parser.add_argument_group('Basic Configuration')
    strategy_group = parser.add_argument_group('Strategy Configuration')
    limits_group = parser.add_argument_group('Campaign Limits')
    les_group = parser.add_argument_group('LES Configuration')

    # Basic configuration
    basic_group.add_argument(
        "name",
        type=str,
        help="Name of the campaign",
    )
    basic_group.add_argument(
        "--n-turbines",
        type=int,
        default=4,
        help="Number of turbines",
    )

    # LES configuration
    les_group.add_argument(
        "--les-config",
        type=Path,
        help="Path to LES configuration YAML file",
        required=False
    )

    # Strategy configuration
    strategy_group.add_argument(
        "--strategy",
        type=str,
        choices=[s.value for s in TrialGenerationStrategy],
        default=TrialGenerationStrategy.LES_ONLY.value,
        help="Trial generation strategy"
    )
    strategy_group.add_argument(
        "--les-batch-size",
        type=int,
        default=5,
        help="Number of LES trials per batch"
    )
    strategy_group.add_argument(
        "--gch-batch-size",
        type=int,
        help="Number of GCH trials per batch (required for multi-fidelity)"
    )
    strategy_group.add_argument(
        "--gch-batches-per-les",
        type=int,
        help="Number of GCH batches per LES batch (required for alternating)"
    )

    # Limits configuration
    limits_group.add_argument(
        "--max-batches",
        type=int,
        help="Maximum total number of batches"
    )
    limits_group.add_argument(
        "--max-les-batches",
        type=int,
        help="Maximum number of LES batches"
    )
    limits_group.add_argument(
        "--max-gch-batches",
        type=int,
        help="Maximum number of GCH batches"
    )

def parse_args():
    parser = ArgumentParser(
        description="Run wind farm layout optimization campaign",
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # New campaign command
    new_parser = subparsers.add_parser(
        'new',
        help='Start a new optimization campaign'
    )
    add_campaign_config_args(new_parser)
    new_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    
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
    restart_parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    try:
        if args.command == 'new':
            strategy = TrialGenerationStrategy(args.strategy)
            
            # Create alternation pattern if using multi-alternating strategy
            alternation_pattern = None
            if strategy == TrialGenerationStrategy.MULTI_ALTERNATING:
                if args.gch_batches_per_les is None:
                    raise ValueError("--gch-batches-per-les is required for multi-alternating strategy")
                alternation_pattern = AlternationPattern.from_counts(
                    n_gch=args.gch_batches_per_les,
                    n_les=1
                )
            
            trial_generation_config = TrialGenerationConfig(
                strategy=strategy,
                les_batch_size=args.les_batch_size,
                gch_batch_size=args.gch_batch_size,
                alternation_pattern=alternation_pattern,
                max_batches=args.max_batches,
                max_les_batches=args.max_les_batches,
                max_gch_batches=args.max_gch_batches,
            )
            campaign_config = CampaignConfig(
                name=args.name,
                n_turbines=args.n_turbines,
                arena_dims=SMALL_ARENA_DIMS,
                box_dims=SMALL_BOX_DIMS,
                trial_generation_config=trial_generation_config,
                les_config_path=args.les_config,
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