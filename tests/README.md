# Test Suite

This directory contains comprehensive tests for the TableSense2 project. The tests verify that all components work correctly according to the requirements specification (v1.2).

## Test Files

### Core Component Tests

#### `test_featurizer.py`
Tests for the cell featurization system. Verifies that all 43 features are correctly extracted from Excel cells, including:
- Basic feature extraction (has_data, has_formula, is_formatted, is_visibly_empty)
- Color bucketing (font and background colors)
- Formatting features (bold, strikethrough, borders)
- Merged cell handling
- Excel table membership detection
- Comments and protection flags

#### `test_featurizer_sample_tables_integration.py`
**Comprehensive integration tests** using `sample_tables.xlsx`. This file provides complete featurizer integration testing. Verifies that all features documented in `data/sample_tables.md` are correctly extracted by the featurizer using the **43-feature specification** (requirements v1.2). Tests cover:
- Sheet1: Bold headers, formulas, formatted cells
- Sheet2: Color buckets, borders, Excel table objects
- Sheet3: Merged cells, large fonts, color features
- Sheet4: Comments, digit proportion, length buckets, all color buckets (red, gray, green, blue, yellow)

#### `test_color_bucketing.py`
Tests for the HLS-based color bucketing system (per Appendix A of requirements). Verifies:
- Achromatic color classification (black, white, gray)
- Chromatic color classification (red, yellow, green, blue, other)
- RGB to bucket conversion
- openpyxl Color object resolution (RGB, indexed, theme colors)

#### `test_excel_utils.py`
Tests for Excel utility functions:
- Column letter ↔ number conversion (A ↔ 1, AA ↔ 27, etc.)
- Excel range parsing ("C8:H18" → coordinates)
- Sheet dimension queries
- Merged region detection
- Excel table object extraction

### Data & I/O Tests

#### `test_annotations.py`
Tests for the annotation loader system:
- JSONL annotation file parsing
- Workbook-level train/val/test splitting
- Excel range to coordinate conversion
- Multiple dataset support

### Evaluation Tests

#### `test_metrics.py`
Tests for evaluation metrics:
- EoB (Error of Boundary) computation
- IoU (Intersection over Union) calculation
- GIoU (Generalized IoU) calculation

#### `test_evaluator.py`
Tests for the table evaluator:
- Perfect match scenarios
- False positive/negative detection
- EoB threshold-based matching
- Precision, recall, and F1 score computation

### Training Tests

#### `test_training_smoke.py`
Smoke tests for the training pipeline:
- Baseline model forward pass
- Training loop execution (one epoch)
- Loss computation
- Checkpoint saving
- Integration with experiment logger

## Test Data

The `data/` subdirectory contains:
- `sample_tables.xlsx`: Comprehensive test Excel file with various table features
- `sample_tables.md`: Documentation of all features in the sample file

## Running Tests

### Prerequisites

Ensure you have the project dependencies installed:
```bash
pip install -r requirements.txt
```

### Run All Tests

Run the entire test suite:
```bash
pytest tests/
```

### Run Specific Test Files

Run a specific test file:
```bash
pytest tests/test_featurizer.py
pytest tests/test_color_bucketing.py
pytest tests/test_featurizer_sample_tables_integration.py
```

### Run Specific Test Classes or Functions

Run a specific test class:
```bash
pytest tests/test_featurizer_sample_tables_integration.py::TestSheet1
```

Run a specific test function:
```bash
pytest tests/test_featurizer.py::test_featurizer_has_data
```

### Run Tests with Verbose Output

Get detailed output showing which tests pass/fail:
```bash
pytest tests/ -v
```

### Run Tests with Coverage

Generate coverage report:
```bash
pytest tests/ --cov=. --cov-report=html
```

### Run Tests in Parallel

Run tests in parallel (faster execution):
```bash
pytest tests/ -n auto
```

Note: Requires `pytest-xdist` plugin: `pip install pytest-xdist`

### Run Tests Matching a Pattern

Run tests matching a pattern:
```bash
pytest tests/ -k "color"  # Run all tests with "color" in the name
pytest tests/ -k "sheet1"  # Run all Sheet1 tests
```

### Run Tests and Stop on First Failure

Stop immediately after first failure:
```bash
pytest tests/ -x
```

### Run Tests with Output Capture Disabled

See print statements during test execution:
```bash
pytest tests/ -s
```

## Test Organization

Tests are organized by component:
- **Featurization**: `test_featurizer*.py` - Cell feature extraction (43 features per cell)
  - `test_featurizer.py`: Unit tests for individual features
  - `test_featurizer_sample_tables_integration.py`: Integration tests with sample Excel file
- **Utilities**: `test_excel_utils.py`, `test_color_bucketing.py` - Helper functions
- **Data I/O**: `test_annotations.py` - Dataset loading
- **Evaluation**: `test_metrics.py`, `test_evaluator.py` - Metrics and evaluation
- **Training**: `test_training_smoke.py` - Training pipeline

**Note**: The old `test_sample_tables.py` file has been replaced by `test_featurizer_sample_tables_integration.py` which provides comprehensive featurizer testing with the 43-feature specification.

## Writing New Tests

When adding new tests:

1. **Follow naming conventions**: Test files should start with `test_`, test functions should start with `test_`
2. **Use fixtures**: Leverage fixtures in `conftest.py` for common setup (e.g., `sample_tables_path`)
3. **Use descriptive names**: Test function names should clearly describe what they test
4. **Add docstrings**: Include docstrings explaining what each test verifies
5. **Group related tests**: Use test classes to group related tests together

Example:
```python
def test_new_feature_extraction():
    """Test that new feature is correctly extracted."""
    featurizer = CellFeaturizer()
    # ... test code ...
    assert features[0, 0, NEW_FEATURE_IDX] == expected_value
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. All tests should:
- Be deterministic (no random behavior without seeds)
- Not depend on external services
- Clean up after themselves (use fixtures for temporary files)
- Complete in reasonable time (< 1 minute for full suite)

## Troubleshooting

### Tests Fail Due to Missing Data Files

Ensure `tests/data/sample_tables.xlsx` exists. If missing, tests will be skipped automatically.

### Import Errors

Make sure the project root is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Or run tests from the project root:
```bash
cd /path/to/TableSense2
pytest tests/
```

### Color Bucketing Tests Fail

Color bucketing tests may be sensitive to slight variations in color resolution. If tests fail, check:
1. openpyxl version compatibility
2. Excel file color encoding (RGB vs indexed vs theme)
3. Color resolution logic in `features/color_bucketing.py`

