"""
Tests for annotation loader.
"""

import pytest
import json
import tempfile
from pathlib import Path
from data_io.annotations import AnnotationLoader


def create_temp_annotations_file(content: list) -> Path:
    """Create a temporary annotations.jsonl file."""
    temp_dir = Path(tempfile.mkdtemp())
    annotations_dir = temp_dir / "training_data" / "test_dataset" / "annotations"
    annotations_dir.mkdir(parents=True)
    
    annotations_file = annotations_dir / "annotations.jsonl"
    with open(annotations_file, 'w') as f:
        for item in content:
            f.write(json.dumps(item) + '\n')
    
    return temp_dir


def test_annotation_loader_basic():
    """Test basic annotation loading."""
    annotations = [
        {
            "file_name": "test1.xlsx",
            "file_path": "training_data/test_dataset/interim",
            "sheet_name": "Sheet1",
            "split_code": "training_set",
            "table_regions": ["A1:B2"]
        },
        {
            "file_name": "test2.xlsx",
            "file_path": "training_data/test_dataset/interim",
            "sheet_name": "Sheet1",
            "split_code": "training_set",
            "table_regions": ["C3:D4", "E5:F6"]
        }
    ]
    
    temp_dir = create_temp_annotations_file(annotations)
    
    loader = AnnotationLoader(
        project_root=temp_dir,
        dataset_names=["test_dataset"]
    )
    loader.load_annotations()
    
    assert len(loader.annotations) == 2
    assert loader.annotations[0]['table_regions_numeric'] == [(1, 1, 2, 2)]
    assert loader.annotations[1]['table_regions_numeric'] == [(3, 3, 4, 4), (5, 5, 6, 6)]


def test_annotation_loader_splits():
    """Test workbook-level splitting."""
    annotations = [
        {
            "file_name": "test1.xlsx",
            "file_path": "training_data/test_dataset/interim",
            "sheet_name": "Sheet1",
            "split_code": "training_set",
            "table_regions": ["A1:B2"]
        },
        {
            "file_name": "test1.xlsx",  # Same workbook
            "file_path": "training_data/test_dataset/interim",
            "sheet_name": "Sheet2",
            "split_code": "training_set",
            "table_regions": ["C3:D4"]
        },
        {
            "file_name": "test2.xlsx",  # Different workbook
            "file_path": "training_data/test_dataset/interim",
            "sheet_name": "Sheet1",
            "split_code": "training_set",
            "table_regions": ["E5:F6"]
        }
    ]
    
    temp_dir = create_temp_annotations_file(annotations)
    
    loader = AnnotationLoader(
        project_root=temp_dir,
        dataset_names=["test_dataset"],
        split_seed=42
    )
    loader.load_annotations()
    
    # Check that all sheets from same workbook are in same split
    train_anns = loader.get_annotations_by_split('train')
    val_anns = loader.get_annotations_by_split('val')
    test_anns = loader.get_annotations_by_split('test')
    
    # All annotations should be assigned to a split
    assert len(train_anns) + len(val_anns) + len(test_anns) == 3
    
    # Check workbook consistency
    workbook_splits = {}
    for ann in loader.annotations:
        wb_id = ann['workbook_id']
        split = loader.workbook_splits[wb_id]
        if wb_id not in workbook_splits:
            workbook_splits[wb_id] = split
        else:
            assert workbook_splits[wb_id] == split  # Same workbook should be in same split


def test_annotation_loader_invalid_range():
    """Test handling of invalid Excel ranges."""
    annotations = [
        {
            "file_name": "test1.xlsx",
            "file_path": "training_data/test_dataset/interim",
            "sheet_name": "Sheet1",
            "split_code": "training_set",
            "table_regions": ["INVALID"]  # Invalid range
        }
    ]
    
    temp_dir = create_temp_annotations_file(annotations)
    
    loader = AnnotationLoader(
        project_root=temp_dir,
        dataset_names=["test_dataset"]
    )
    
    with pytest.raises(ValueError):
        loader.load_annotations()

