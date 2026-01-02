"""
Tests for Phase 3 config system.

Verifies that the Config class can load and validate YAML configs correctly.
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from training.config import Config


def test_config_loads_valid_yaml():
    """Test that Config can load a valid YAML file."""
    config_dict = {
        "experiment": {"name": "test_experiment"},
        "data": {"dataset_names": ["test_dataset"]},
        "model": {"type": "detector"},
        "training": {"max_epochs": 10},
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = Path(f.name)
    
    try:
        cfg = Config.from_yaml(config_path)
        assert cfg.experiment["name"] == "test_experiment"
        assert cfg.data["dataset_names"] == ["test_dataset"]
        assert cfg.model["type"] == "detector"
        assert cfg.training["max_epochs"] == 10
    finally:
        config_path.unlink()


def test_config_validation_missing_sections():
    """Test that Config validates required sections."""
    config_dict = {
        "experiment": {"name": "test"},
        # Missing data, model, training
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="missing required top-level sections"):
            Config.from_yaml(config_path)
    finally:
        config_path.unlink()


def test_config_validation_missing_experiment_name():
    """Test that Config validates experiment.name exists."""
    config_dict = {
        "experiment": {},  # Missing name
        "data": {"dataset_names": ["test"]},
        "model": {},
        "training": {},
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="experiment.name"):
            Config.from_yaml(config_path)
    finally:
        config_path.unlink()


def test_config_validation_missing_dataset_names():
    """Test that Config validates data.dataset_names exists."""
    config_dict = {
        "experiment": {"name": "test"},
        "data": {},  # Missing dataset_names
        "model": {},
        "training": {},
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="data.dataset_names"):
            Config.from_yaml(config_path)
    finally:
        config_path.unlink()


def test_config_dict_like_interface():
    """Test that Config provides dict-like interface."""
    config_dict = {
        "experiment": {"name": "test"},
        "data": {"dataset_names": ["test"]},
        "model": {},
        "training": {},
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = Path(f.name)
    
    try:
        cfg = Config.from_yaml(config_path)
        
        # Test __getitem__
        assert cfg["experiment"]["name"] == "test"
        
        # Test get()
        assert cfg.get("experiment")["name"] == "test"
        assert cfg.get("nonexistent", "default") == "default"
        
        # Test to_dict()
        full_dict = cfg.to_dict()
        assert isinstance(full_dict, dict)
        assert full_dict["experiment"]["name"] == "test"
    finally:
        config_path.unlink()


def test_config_property_accessors():
    """Test that Config property accessors work correctly."""
    config_dict = {
        "experiment": {"name": "test", "project_name": "test-project"},
        "data": {"dataset_names": ["test"], "train_ratio": 0.8},
        "model": {"type": "detector", "hidden_channels": 64},
        "training": {"max_epochs": 100, "lr": 1e-3},
        "evaluation": {"eob_threshold": 2.0},
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = Path(f.name)
    
    try:
        cfg = Config.from_yaml(config_path)
        
        assert cfg.experiment == config_dict["experiment"]
        assert cfg.data == config_dict["data"]
        assert cfg.model == config_dict["model"]
        assert cfg.training == config_dict["training"]
        assert cfg.evaluation == config_dict["evaluation"]
    finally:
        config_path.unlink()


def test_config_loads_detector_config():
    """Test that Config can load the detector_config.yaml (if it exists)."""
    project_root = Path(__file__).parent.parent
    detector_config_path = project_root / "configs" / "detector_config.yaml"
    
    if detector_config_path.exists():
        cfg = Config.from_yaml(detector_config_path)
        assert cfg.experiment["name"] == "deformable_detr_table_detector"
        assert cfg.model["type"] == "detector"
        # Detector config should have backbone and detr sections
        assert "backbone" in cfg.model
        assert "detr" in cfg.model
    else:
        pytest.skip("detector_config.yaml not found")

