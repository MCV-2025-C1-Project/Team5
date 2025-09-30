import yaml


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


params = load_parameters()
