import os
import yaml
from pathlib import Path

_config_cache = None


def get_project_root():
    """Returns the project root directory (Dynasty_Model/)."""
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(*parts):
    """Resolves a path relative to the project root."""
    return get_project_root() / Path(*parts)


def load_config(config_path=None):
    """
    Loads the YAML config file. Caches after first load.
    """
    global _config_cache
    if _config_cache is not None and config_path is None:
        return _config_cache

    if config_path is None:
        config_path = resolve_path("configs", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config_path is None or config_path == resolve_path("configs", "config.yaml"):
        _config_cache = config

    return config


def get_season_games(season, config=None):
    """Returns the number of games in a given season, accounting for shortened seasons."""
    if config is None:
        config = load_config()
    season_games = config["preprocessing"]["season_games"]
    return season_games.get(season, season_games["default"])
