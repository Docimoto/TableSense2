"""
Annotation loading and dataset management.

Loads JSONL annotation files and provides dataset splitting functionality.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
from collections import defaultdict

from utils.excel_utils import parse_excel_range, get_workbook_id


class AnnotationLoader:
    """
    Loads and manages annotations from JSONL files.
    
    Supports:
    - Loading annotations from multiple datasets
    - Workbook-level train/val/test splits (80/10/10)
    - Deterministic splitting using hashing
    - Converting Excel ranges to numeric coordinates
    """
    
    def __init__(
        self,
        project_root: Path,
        dataset_names: List[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_seed: int = 42,
    ):
        """
        Initialize annotation loader.
        
        Args:
            project_root: Root directory of the project
            dataset_names: List of dataset names to load (e.g., ["tablesense"])
            train_ratio: Proportion for training set (default: 0.8)
            val_ratio: Proportion for validation set (default: 0.1)
            test_ratio: Proportion for test set (default: 0.1)
            split_seed: Seed for deterministic splitting
        """
        self.project_root = Path(project_root)
        self.dataset_names = dataset_names
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        
        # Validate ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        self.annotations: List[Dict] = []
        self.workbook_splits: Dict[str, str] = {}  # workbook_id -> split name
        
    def load_annotations(self):
        """
        Load annotations from JSONL files for all specified datasets.
        
        Annotations are loaded from:
        <project_root>/training_data/<dataset_name>/annotations/annotations.jsonl
        """
        self.annotations = []
        
        for dataset_name in self.dataset_names:
            annotations_path = (
                self.project_root / "training_data" / dataset_name / "annotations" / "annotations.jsonl"
            )
            
            if not annotations_path.exists():
                raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
            
            with open(annotations_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        annotation = json.loads(line)
                        # Validate required fields
                        required_fields = ['file_name', 'file_path', 'sheet_name', 'table_regions']
                        for field in required_fields:
                            if field not in annotation:
                                raise ValueError(f"Missing required field '{field}' in line {line_num}")
                        
                        # Convert Excel ranges to numeric tuples
                        table_regions_numeric = []
                        for range_str in annotation['table_regions']:
                            try:
                                col_left, row_top, col_right, row_bottom = parse_excel_range(range_str)
                                table_regions_numeric.append((col_left, row_top, col_right, row_bottom))
                            except ValueError as e:
                                raise ValueError(f"Invalid range '{range_str}' in line {line_num}: {e}")
                        
                        annotation['table_regions_numeric'] = table_regions_numeric
                        
                        # Generate workbook ID
                        workbook_id = get_workbook_id(annotation['file_path'], annotation['file_name'])
                        annotation['workbook_id'] = workbook_id
                        
                        self.annotations.append(annotation)
                        
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in line {line_num}: {e}")
                    except Exception as e:
                        raise ValueError(f"Error processing line {line_num}: {e}")
        
        # Compute workbook-level splits
        self._compute_workbook_splits()
    
    def _compute_workbook_splits(self):
        """
        Compute deterministic workbook-level splits using hashing.
        
        Ensures no workbook appears in more than one split.
        """
        # Group annotations by workbook
        workbooks = defaultdict(list)
        for ann in self.annotations:
            workbooks[ann['workbook_id']].append(ann)
        
        # Sort workbook IDs for deterministic ordering
        workbook_ids = sorted(workbooks.keys())
        
        # Assign splits deterministically using hash
        for workbook_id in workbook_ids:
            # Use hash of workbook_id + seed for deterministic assignment
            hash_input = f"{workbook_id}_{self.split_seed}".encode('utf-8')
            hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
            normalized_hash = (hash_value % 10000) / 10000.0  # Value in [0, 1)
            
            if normalized_hash < self.train_ratio:
                split = 'train'
            elif normalized_hash < self.train_ratio + self.val_ratio:
                split = 'val'
            else:
                split = 'test'
            
            self.workbook_splits[workbook_id] = split
    
    def get_annotations_by_split(self, split: str) -> List[Dict]:
        """
        Get all annotations for a specific split.
        
        Args:
            split: Split name ('train', 'val', or 'test')
            
        Returns:
            List of annotation dictionaries
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        result = []
        for ann in self.annotations:
            workbook_id = ann['workbook_id']
            if self.workbook_splits[workbook_id] == split:
                result.append(ann)
        
        return result
    
    def get_full_file_path(self, annotation: Dict) -> Path:
        """
        Construct the full file path for an Excel file from an annotation.
        
        Args:
            annotation: Annotation dictionary with 'file_path' and 'file_name'
            
        Returns:
            Full Path object to the Excel file
        """
        return self.project_root / annotation['file_path'] / annotation['file_name']
    
    def iter_sheets(self, split: Optional[str] = None) -> Iterator[Tuple[str, str, List[Tuple[int, int, int, int]]]]:
        """
        Iterate over sheets with their ground truth table regions.
        
        Args:
            split: Optional split filter ('train', 'val', 'test'). If None, returns all.
            
        Yields:
            Tuples of (workbook_id, sheet_name, table_regions)
            where table_regions is a list of (col_left, row_top, col_right, row_bottom) tuples
        """
        annotations = self.get_annotations_by_split(split) if split else self.annotations
        
        for ann in annotations:
            workbook_id = ann['workbook_id']
            sheet_name = ann['sheet_name']
            table_regions = ann['table_regions_numeric']
            yield (workbook_id, sheet_name, table_regions)
    
    def get_split_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics about the dataset splits.
        
        Returns:
            Dictionary with split names as keys and stats as values:
            - num_workbooks: Number of unique workbooks
            - num_sheets: Number of sheets
            - num_tables: Total number of table regions
        """
        stats = {}
        for split in ['train', 'val', 'test']:
            annotations = self.get_annotations_by_split(split)
            workbooks = set(ann['workbook_id'] for ann in annotations)
            num_tables = sum(len(ann['table_regions_numeric']) for ann in annotations)
            
            stats[split] = {
                'num_workbooks': len(workbooks),
                'num_sheets': len(annotations),
                'num_tables': num_tables,
            }
        
        return stats

