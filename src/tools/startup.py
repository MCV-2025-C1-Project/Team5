import yaml
import logging
import sys


# Load settings
def load_parameters(params_path: str = "config/settings.yaml") -> dict:
    """
    Read parameters and load them.

    Args:
        params_path: Path of file containing parameters.

    Returns:
        Dict object with parameters.
    """
    with open(params_path, encoding="utf8") as par_file:
        params = yaml.safe_load(par_file)
    return params


def init_logger() -> logging.Logger:
    """Initialize and configure a simple logger.

    Args:
        name (str): Logger name, usually __name__ of the calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers in interactive or reloaded environments
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(filename)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


params = load_parameters()
logger = init_logger()
