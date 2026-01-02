"""
Configuration utilities for training and evaluation.

This module provides a lightweight `Config` class for:
- Loading YAML configuration files
- Basic structure validation
- Providing a single place to evolve config handling as we add detectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    """
    Thin wrapper around a nested configuration dictionary.

    The goal is to centralize config loading/validation while keeping the
    interface familiar (dict-like) for existing scripts.
    """

    raw: Dict[str, Any]
    path: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls(raw=data, path=path)
        cfg.validate_basic()
        return cfg

    # ------------------------------------------------------------------ #
    # Dict-like helpers
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying configuration dictionary."""
        return self.raw

    # ------------------------------------------------------------------ #
    # Section accessors (convenience)
    # ------------------------------------------------------------------ #
    @property
    def experiment(self) -> Dict[str, Any]:
        return self.raw.get("experiment", {})

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.raw.get("evaluation", {})

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate_basic(self) -> None:
        """
        Perform basic structural validation common to all experiments.

        This keeps the checks intentionally light-weight so configs remain
        easy to evolve across phases.
        """
        required_top_level = ["experiment", "data", "model", "training"]
        missing = [k for k in required_top_level if k not in self.raw]
        if missing:
            path_str = str(self.path) if self.path else ""
            raise ValueError(
                f"Config file {path_str} is missing "
                f"required top-level sections: {missing}"
            )

        # Minimal sanity checks for commonly used fields
        if "name" not in self.experiment:
            raise ValueError("Config must define experiment.name")

        if "dataset_names" not in self.data:
            raise ValueError("Config must define data.dataset_names")

        # Training section defaults are applied in scripts/train_* if absent,
        # so we only ensure the section exists here.


