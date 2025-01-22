"""
Centralized logging configuration for the optimization module.
"""
import logging
from datetime import datetime
from pathlib import Path

from ax.utils.common.logger import ROOT_LOGGER as AX_LOGGER

# Module-level logger
logger = logging.getLogger("windopt.optim")

def configure_logging(
    campaign_name: str,
    log_dir: Path,
    debug_mode: bool = False
) -> None:
    """
    Configure logging for the optimization module.
    
    This sets up:
    - File logging for the campaign
    - Console output
    - Ax library logging integration
    - Proper log levels and formatting
    
    Args:
        campaign_name: Name of the campaign for log file naming
        log_dir: Directory to store log files
        debug_mode: Whether to enable debug logging
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{campaign_name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    level = logging.DEBUG if debug_mode else logging.INFO
    for handler in [file_handler, console_handler]:
        handler.setFormatter(formatter)
        handler.setLevel(level)
    
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Route Ax log output to file as well
    AX_LOGGER.addHandler(file_handler) 