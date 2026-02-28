"""Shared utilities for saving/loading model artifacts."""

import json
import os


def ensure_dir(path):
    """Create directory and parents if they don't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_metadata(artifact_dir, metadata_dict):
    """Write metadata.json to artifact directory."""
    ensure_dir(artifact_dir)
    path = os.path.join(artifact_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata_dict, f, indent=2, default=str)
    return path


def load_metadata(artifact_dir):
    """Read metadata.json from artifact directory."""
    path = os.path.join(artifact_dir, "metadata.json")
    with open(path) as f:
        return json.load(f)
