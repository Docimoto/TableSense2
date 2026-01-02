"""
Cell featurization for Excel worksheets.

Converts each cell in a worksheet into a 43-dimensional feature vector
representing data presence, formatting, colors, borders, and structural properties.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.cell.cell import Cell
from datetime import datetime

from features.color_bucketing import ColorBucketer
from utils.excel_utils import get_merged_regions, iterate_cells, get_sheet_dimensions


# Feature dimensions
NUM_FEATURES = 43

# Feature group sizes
COLOR_BUCKET_SIZE = 8  # black, white, gray, red, yellow, green, blue, other
LENGTH_BUCKET_SIZE = 4  # s, m, l, xl
FONT_SIZE_BUCKET_SIZE = 3  # s, m, l

# Feature Index Mapping (0-based indices)
# Each feature is a float32 value (typically 0.0 or 1.0 for binary features)
FEATURE_INDEX = {
    # Basic data features (0-3)
    0: "has_data",                    # 1.0 if cell has value or formula, else 0.0
    1: "has_formula",                 # 1.0 if cell.data_type == "f", else 0.0
    2: "is_formatted",                # 1.0 if number_format != 'General', else 0.0
    3: "is_visibly_empty",            # 1.0 if value is None/"" or non-top-left merged cell, else 0.0
    
    # Merge indicators (4-7)
    4: "merge_left",                  # 1.0 if cell is merged and not leftmost, else 0.0
    5: "merge_right",                 # 1.0 if cell is merged and not rightmost, else 0.0
    6: "merge_top",                   # 1.0 if cell is merged and not topmost, else 0.0
    7: "merge_bottom",                # 1.0 if cell is merged and not bottommost, else 0.0
    
    # Font color buckets (8-15) - one-hot encoding
    8: "font_color_black",            # 1.0 if font color is black, else 0.0
    9: "font_color_white",            # 1.0 if font color is white, else 0.0
    10: "font_color_gray",            # 1.0 if font color is gray, else 0.0
    11: "font_color_red",             # 1.0 if font color is red, else 0.0
    12: "font_color_yellow",           # 1.0 if font color is yellow, else 0.0
    13: "font_color_green",            # 1.0 if font color is green, else 0.0
    14: "font_color_blue",            # 1.0 if font color is blue, else 0.0
    15: "font_color_other",           # 1.0 if font color is other (not black/white/gray/red/yellow/green/blue), else 0.0
    
    # Background color buckets (16-23) - one-hot encoding
    16: "bg_color_black",              # 1.0 if background color is black, else 0.0
    17: "bg_color_white",              # 1.0 if background color is white, else 0.0
    18: "bg_color_gray",               # 1.0 if background color is gray, else 0.0
    19: "bg_color_red",                # 1.0 if background color is red, else 0.0
    20: "bg_color_yellow",             # 1.0 if background color is yellow, else 0.0
    21: "bg_color_green",              # 1.0 if background color is green, else 0.0
    22: "bg_color_blue",               # 1.0 if background color is blue, else 0.0
    23: "bg_color_other",              # 1.0 if background color is other, else 0.0
    
    # Content features (24)
    24: "digit_proportion",            # Proportion of non-whitespace chars that are digits (0-1)
    
    # Length buckets (25-28) - one-hot encoding
    25: "length_s",                    # 1.0 if length < 10, else 0.0
    26: "length_m",                    # 1.0 if 10 <= length < 50, else 0.0
    27: "length_l",                    # 1.0 if 50 <= length < 100, else 0.0
    28: "length_xl",                   # 1.0 if length >= 100, else 0.0
    
    # Font size buckets (29-31) - one-hot encoding (relative to sheet median)
    29: "font_size_s",                 # 1.0 if font_size < 0.7 * median, else 0.0
    30: "font_size_m",                 # 1.0 if 0.7 * median <= font_size <= 1.0 * median, else 0.0
    31: "font_size_l",                 # 1.0 if font_size > 1.0 * median, else 0.0
    
    # Font style flags (32-33)
    32: "bold",                        # 1.0 if font.bold is True, else 0.0
    33: "strikethrough",               # 1.0 if font.strike is True, else 0.0
    
    # Border indicators (34-37)
    34: "border_left",                 # 1.0 if left border style exists, else 0.0
    35: "border_right",                # 1.0 if right border style exists, else 0.0
    36: "border_top",                  # 1.0 if top border style exists, else 0.0
    37: "border_bottom",               # 1.0 if bottom border style exists, else 0.0
    
    # Protection flags (38-39)
    38: "is_locked",                   # 1.0 if cell.protection.locked is True, else 0.0
    39: "is_hidden",                   # 1.0 if cell.protection.hidden is True, else 0.0
    
    # Excel table membership (40-41)
    40: "is_in_table",                 # 1.0 if cell is in an Excel table, else 0.0
    41: "is_in_table_header",          # 1.0 if cell is in Excel table header row, else 0.0
    
    # Comment flag (42)
    42: "has_comment",                 # 1.0 if cell has a comment, else 0.0
}


class CellFeaturizer:
    """
    Extracts 43 features per cell from an Excel worksheet.
    
    Features include:
    - Has data (1 channel)
    - Has formula (1 channel)
    - Is formatted (1 channel)
    - Is visibly empty (1 channel)
    - Merge indicators (4 channels)
    - Font color (8 channels)
    - Background color (8 channels)
    - Digit proportion (1 channel)
    - Length of data (4 channels)
    - Font size bucket (3 channels)
    - Bold flag (1 channel)
    - Strikethrough flag (1 channel)
    - Border indicators (4 channels)
    - Lock/hidden flags (2 channels)
    - Excel table membership (2 channels)
    - Comment flags (1 channel)
    """
    
    def __init__(self):
        """Initialize the featurizer."""
        self.color_bucketer = ColorBucketer()
    
    def featurize_sheet(
        self,
        workbook: Workbook,
        sheet_name: str,
        merged_regions: Optional[List[Tuple[int, int, int, int]]] = None,
        excel_tables: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Featurize an entire worksheet.
        
        Args:
            workbook: openpyxl Workbook object
            sheet_name: Name of the sheet to featurize
            merged_regions: Optional pre-computed merged regions list
            excel_tables: Optional dict mapping table names to (col_left, row_top, col_right, row_bottom)
            
        Returns:
            Tuple of:
            - Feature tensor of shape (H, W, 43) where H=rows, W=columns
            - Metadata dict with 'median_font_size' and other sheet-level info
        """
        sheet = workbook[sheet_name]
        H, W = get_sheet_dimensions(workbook, sheet_name)
        
        if H == 0 or W == 0:
            # Empty sheet
            return np.zeros((H, W, NUM_FEATURES), dtype=np.float32), {}
        
        # Get merged regions if not provided
        if merged_regions is None:
            merged_regions = get_merged_regions(sheet)
        
        # Build merged cell lookup: (row, col) -> (row_top, col_left, row_bottom, col_right)
        merged_lookup = {}
        for row_top, col_left, row_bottom, col_right in merged_regions:
            for r in range(row_top, row_bottom + 1):
                for c in range(col_left, col_right + 1):
                    merged_lookup[(r, c)] = (row_top, col_left, row_bottom, col_right)
        
        # Get Excel table membership if provided
        table_membership = {}
        table_header_rows = set()
        if excel_tables:
            for table_name, (col_left, row_top, col_right, row_bottom) in excel_tables.items():
                for r in range(row_top, row_bottom + 1):
                    for c in range(col_left, col_right + 1):
                        table_membership[(r, c)] = True
                        if r == row_top:  # Header row
                            table_header_rows.add((r, c))
        
        # Compute sheet-level statistics
        font_sizes = []
        for row_idx, col_idx, cell in iterate_cells(sheet):
            if cell.font and cell.font.sz:
                font_sizes.append(cell.font.sz)
        median_font_size = np.median(font_sizes) if font_sizes else 11.0  # Default to 11 if no fonts
        
        # Initialize feature tensor
        features = np.zeros((H, W, NUM_FEATURES), dtype=np.float32)
        
        # Featurize each cell
        for row_idx, col_idx, cell in iterate_cells(sheet):
            if row_idx > H or col_idx > W:
                continue
            
            cell_features = self._featurize_cell(
                cell=cell,
                sheet=sheet,
                row_idx=row_idx,
                col_idx=col_idx,
                merged_lookup=merged_lookup,
                table_membership=table_membership,
                table_header_rows=table_header_rows,
                median_font_size=median_font_size,
            )
            features[row_idx - 1, col_idx - 1, :] = cell_features  # Convert to 0-based indexing
        
        metadata = {
            'median_font_size': median_font_size,
            'height': H,
            'width': W,
        }
        
        return features, metadata
    
    def _featurize_cell(
        self,
        cell: Cell,
        sheet: Worksheet,
        row_idx: int,
        col_idx: int,
        merged_lookup: Dict[Tuple[int, int], Tuple[int, int, int, int]],
        table_membership: Dict[Tuple[int, int], bool],
        table_header_rows: set,
        median_font_size: float,
    ) -> np.ndarray:
        """
        Extract features for a single cell.
        
        For merged cells, formatting features are extracted from the top-left cell
        of the merged region, while the cell value comes from the current cell.
        
        Returns:
            Feature vector of length 43
        """
        features = np.zeros(NUM_FEATURES, dtype=np.float32)
        feature_idx = 0
        
        # Get the effective (top-left) cell for formatting if this cell is merged
        # The cell value still comes from the current cell, but formatting comes from top-left
        format_cell = cell
        is_top_left = True
        if (row_idx, col_idx) in merged_lookup:
            row_top, col_left, _, _ = merged_lookup[(row_idx, col_idx)]
            # Only get top-left cell if this is not already the top-left cell
            if row_idx != row_top or col_idx != col_left:
                format_cell = sheet.cell(row=row_top, column=col_left)
                is_top_left = False
        
        # 0. Has data (1 channel)
        # Merged cells inherit from top-left cell, except is_visibly_empty
        has_data = self._has_data(format_cell)
        features[feature_idx] = has_data
        feature_idx += 1
        
        # 1. Has formula (1 channel)
        # Merged cells inherit from top-left cell
        has_formula = self._has_formula(format_cell)
        features[feature_idx] = has_formula
        feature_idx += 1
        
        # 2. Is formatted (1 channel)
        # Merged cells inherit from top-left cell
        is_formatted = 1.0 if (format_cell.number_format and format_cell.number_format != 'General') else 0.0
        features[feature_idx] = is_formatted
        feature_idx += 1
        
        # 3. Is visibly empty (1 channel)
        # For merged non-top-left cells, this is always 1
        # For other cells, check if value is None or ""
        is_visibly_empty = self._is_visibly_empty(cell, row_idx, col_idx, merged_lookup, is_top_left)
        features[feature_idx] = is_visibly_empty
        feature_idx += 1
        
        # 4-7. Merge indicators (4 channels)
        merged_left, merged_right, merged_top, merged_bottom = self._get_merge_indicators(
            row_idx, col_idx, merged_lookup
        )
        features[feature_idx] = merged_left
        features[feature_idx + 1] = merged_right
        features[feature_idx + 2] = merged_top
        features[feature_idx + 3] = merged_bottom
        feature_idx += 4
        
        # 8-15. Font color (8-channel one-hot)
        # Use format_cell for formatting
        # Default font color in Excel is black when font.color is None
        font_color_bucket = self.color_bucketer.color_to_bucket(
            format_cell.font.color if format_cell.font else None,
            default_for_none='black'
        )
        features[feature_idx + font_color_bucket] = 1.0
        feature_idx += COLOR_BUCKET_SIZE
        
        # 16-23. Background color (8-channel one-hot)
        # Use format_cell for formatting
        # Default background in Excel is white when fill is None or fgColor is None
        bg_color = format_cell.fill.fgColor if format_cell.fill else None
        bg_color_bucket = self.color_bucketer.color_to_bucket(bg_color, default_for_none='white')
        features[feature_idx + bg_color_bucket] = 1.0
        feature_idx += COLOR_BUCKET_SIZE
        
        # 24. Digit proportion (1 channel)
        # Use current cell for value
        cell_value_str = self._get_cell_value_string(cell)
        digit_prop = self._compute_digit_proportion(cell_value_str)
        features[feature_idx] = digit_prop
        feature_idx += 1
        
        # 25-28. Length of data (4-channel one-hot)
        # Use current cell for value
        length_bucket = self._get_length_bucket(cell_value_str)
        features[feature_idx + length_bucket] = 1.0
        feature_idx += LENGTH_BUCKET_SIZE
        
        # 29-31. Font size bucket (3-channel one-hot)
        # Use format_cell for formatting
        font_size_bucket = self._get_font_size_bucket(format_cell, median_font_size)
        features[feature_idx + font_size_bucket] = 1.0
        feature_idx += FONT_SIZE_BUCKET_SIZE
        
        # 32. Bold flag (1 channel)
        # Use format_cell for formatting
        is_bold = 1.0 if (format_cell.font and format_cell.font.bold) else 0.0
        features[feature_idx] = is_bold
        feature_idx += 1
        
        # 33. Strikethrough flag (1 channel)
        # Use format_cell for formatting
        has_strikethrough = 1.0 if (format_cell.font and format_cell.font.strike) else 0.0
        features[feature_idx] = has_strikethrough
        feature_idx += 1
        
        # 34-37. Border indicators (4 channels)
        # Use format_cell for formatting
        border_left = 1.0 if (format_cell.border and format_cell.border.left and format_cell.border.left.style) else 0.0
        border_right = 1.0 if (format_cell.border and format_cell.border.right and format_cell.border.right.style) else 0.0
        border_top = 1.0 if (format_cell.border and format_cell.border.top and format_cell.border.top.style) else 0.0
        border_bottom = 1.0 if (format_cell.border and format_cell.border.bottom and format_cell.border.bottom.style) else 0.0
        features[feature_idx] = border_left
        features[feature_idx + 1] = border_right
        features[feature_idx + 2] = border_top
        features[feature_idx + 3] = border_bottom
        feature_idx += 4
        
        # 38. Is locked (1 channel)
        # Use format_cell for formatting
        is_locked = 1.0 if (format_cell.protection and format_cell.protection.locked) else 0.0
        features[feature_idx] = is_locked
        feature_idx += 1
        
        # 39. Is hidden (1 channel)
        # Use format_cell for formatting
        is_hidden = 1.0 if (format_cell.protection and format_cell.protection.hidden) else 0.0
        features[feature_idx] = is_hidden
        feature_idx += 1
        
        # 40-41. Excel table membership (2 channels)
        is_in_table = 1.0 if (row_idx, col_idx) in table_membership else 0.0
        is_in_table_header = 1.0 if (row_idx, col_idx) in table_header_rows else 0.0
        features[feature_idx] = is_in_table
        features[feature_idx + 1] = is_in_table_header
        feature_idx += 2
        
        # 42. Comment flags (1 channel)
        # Comments are typically on the top-left cell, but check current cell too
        # Use format_cell for consistency with formatting
        has_comment = 1.0 if format_cell.comment else 0.0
        features[feature_idx] = has_comment
        feature_idx += 1
        
        assert feature_idx == NUM_FEATURES, f"Feature count mismatch: {feature_idx} != {NUM_FEATURES}"
        
        return features
    
    def _has_data(self, cell: Cell) -> float:
        """
        Check if cell has data (value or formula).
        
        Returns:
            1.0 if cell has value or formula, else 0.0
        """
        if cell.value is not None and cell.value != "":
            return 1.0
        if cell.data_type == 'f':  # Formula
            return 1.0
        return 0.0
    
    def _has_formula(self, cell: Cell) -> float:
        """
        Check if cell has a formula.
        
        Returns:
            1.0 if cell.data_type == "f", else 0.0
        """
        return 1.0 if cell.data_type == 'f' else 0.0
    
    def _is_visibly_empty(
        self,
        cell: Cell,
        row_idx: int,
        col_idx: int,
        merged_lookup: Dict[Tuple[int, int], Tuple[int, int, int, int]],
        is_top_left: bool,
    ) -> float:
        """
        Check if cell is visibly empty.
        
        Returns:
            1.0 if:
            - cell.value is None OR
            - cell.value == "" OR
            - Cell is non-top-left cell in merged range
            Else 0.0
        """
        # Check if it's a non-top-left cell in merged range
        if not is_top_left:
            return 1.0
        
        # Check if value is None or empty string
        if cell.value is None or cell.value == "":
            return 1.0
        
        return 0.0
    
    def _get_cell_value_string(self, cell: Cell) -> str:
        """Get string representation of cell value."""
        if cell.value is None:
            return ""
        return str(cell.value)
    
    def _compute_digit_proportion(self, value_str: str) -> float:
        """
        Compute proportion of characters that are digits.
        
        Requirements: "Total characters includes all non white space characters.
        Digits are defined as values between 0-9."
        
        Returns:
            Value in [0, 1]
        """
        if not value_str:
            return 0.0
        
        # Remove whitespace for counting total characters
        # "Total characters includes all non white space characters"
        non_whitespace = ''.join(c for c in value_str if not c.isspace())
        if not non_whitespace:
            return 0.0
        
        # Count digits (0-9 only, as per requirements)
        digit_count = sum(1 for c in non_whitespace if c in '0123456789')
        return digit_count / len(non_whitespace)
    
    def _get_length_bucket(self, value_str: str) -> int:
        """
        Get length bucket index.
        
        Returns:
            0: s (< 10), 1: m (< 50), 2: l (< 100), 3: xl (>= 100)
        """
        length = len(value_str)
        if length < 10:
            return 0  # s
        elif length < 50:
            return 1  # m
        elif length < 100:
            return 2  # l
        else:
            return 3  # xl
    
    def _get_font_size_bucket(self, cell: Cell, median_font_size: float) -> int:
        """
        Get font size bucket relative to sheet median.
        
        Requirements:
        - s: sz < 0.7 * median
        - m: 0.7 * median ≤ sz ≤ 1 * median
        - l: sz > 1 * median
        
        Returns:
            0: s, 1: m, 2: l
        """
        if not cell.font or not cell.font.sz:
            # Default to medium if no font size
            return 1
        
        font_size = cell.font.sz
        ratio = font_size / median_font_size if median_font_size > 0 else 1.0
        
        if ratio < 0.7:
            return 0  # s
        elif ratio <= 1.0:
            return 1  # m
        else:
            return 2  # l
    
    def _get_merge_indicators(
        self,
        row_idx: int,
        col_idx: int,
        merged_lookup: Dict[Tuple[int, int], Tuple[int, int, int, int]],
    ) -> Tuple[float, float, float, float]:
        """
        Get merge indicators for a cell.
        
        Returns:
            Tuple of (merged_left, merged_right, merged_top, merged_bottom) flags
        """
        if (row_idx, col_idx) not in merged_lookup:
            return (0.0, 0.0, 0.0, 0.0)
        
        row_top, col_left, row_bottom, col_right = merged_lookup[(row_idx, col_idx)]
        
        merged_left = 1.0 if col_idx > col_left else 0.0
        merged_right = 1.0 if col_idx < col_right else 0.0
        merged_top = 1.0 if row_idx > row_top else 0.0
        merged_bottom = 1.0 if row_idx < row_bottom else 0.0
        
        return (merged_left, merged_right, merged_top, merged_bottom)

