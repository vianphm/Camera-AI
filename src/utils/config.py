"""Configuration loading utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Safely load a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing configuration values.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def load_config(config_name: str = "model.yaml") -> Dict[str, Any]:
    """Load configuration from the configs/ directory.

    Args:
        config_name: Name of YAML file in configs/ (e.g. 'model.yaml').

    Returns:
        Dictionary of settings.
    """
    config_path = get_project_root() / "configs" / config_name
    return load_yaml(config_path)


def get_env(key: str, default: Optional[Any] = None) -> Any:
    """Get environment variable with fallback default."""
    return os.getenv(key, default)
