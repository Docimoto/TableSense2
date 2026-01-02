"""
Pytest configuration and shared fixtures for tests.
"""

import pytest
from pathlib import Path
from utils.excel_utils import load_workbook


@pytest.fixture
def test_data_dir():
    """Return path to tests/data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_tables_path(test_data_dir):
    """Return path to sample_tables.xlsx test file."""
    return test_data_dir / "sample_tables.xlsx"


def get_test_data_path(filename: str) -> Path:
    """
    Get path to a test data file.
    
    Args:
        filename: Name of the test data file (e.g., "sample_tables.xlsx")
        
    Returns:
        Path to the test data file
    """
    test_data_dir = Path(__file__).parent / "data"
    return test_data_dir / filename


def load_test_excel(filename: str, sheet_name: str = None, data_only: bool = False):
    """
    Load a test Excel file.
    
    Args:
        filename: Name of the Excel file in tests/data/
        sheet_name: Optional sheet name (defaults to active sheet)
        data_only: If True, formulas are evaluated to values. If False, formulas are preserved.
                   Default False to allow formula detection in featurization tests.
        
    Returns:
        Workbook object or Worksheet if sheet_name is specified
    """
    file_path = get_test_data_path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Test data file not found: {file_path}")
    
    workbook = load_workbook(file_path, data_only=data_only)
    
    if sheet_name:
        return workbook[sheet_name]
    return workbook

