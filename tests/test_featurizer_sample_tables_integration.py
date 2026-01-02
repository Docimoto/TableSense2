"""
Comprehensive integration tests for featurizer using sample_tables.xlsx.

Tests verify that all features described in tests/data/sample_tables.md are correctly
extracted by the CellFeaturizer with the 43-feature specification.
"""

import pytest
from pathlib import Path
from utils.excel_utils import load_workbook

from features.featurizer import CellFeaturizer, NUM_FEATURES
from utils.excel_utils import get_excel_tables


# Feature indices (0-based)
HAS_DATA_IDX = 0
HAS_FORMULA_IDX = 1
IS_FORMATTED_IDX = 2
IS_VISIBLY_EMPTY_IDX = 3
MERGE_LEFT_IDX = 4
MERGE_RIGHT_IDX = 5
MERGE_TOP_IDX = 6
MERGE_BOTTOM_IDX = 7
FONT_COLOR_START = 8  # 8-15: font color buckets
BG_COLOR_START = 16  # 16-23: background color buckets
DIGIT_PROPORTION_IDX = 24
LENGTH_START = 25  # 25-28: length buckets (s, m, l, xl)
FONT_SIZE_START = 29  # 29-31: font size buckets (s, m, l)
BOLD_IDX = 32
STRIKETHROUGH_IDX = 33
BORDER_LEFT_IDX = 34
BORDER_RIGHT_IDX = 35
BORDER_TOP_IDX = 36
BORDER_BOTTOM_IDX = 37
IS_LOCKED_IDX = 38
IS_HIDDEN_IDX = 39
IS_IN_TABLE_IDX = 40
IS_IN_TABLE_HEADER_IDX = 41
HAS_COMMENT_IDX = 42

# Color bucket indices
COLOR_BLACK = 0
COLOR_WHITE = 1
COLOR_GRAY = 2
COLOR_RED = 3
COLOR_YELLOW = 4
COLOR_GREEN = 5
COLOR_BLUE = 6
COLOR_OTHER = 7

# Length bucket indices
LENGTH_S = 0
LENGTH_M = 1
LENGTH_L = 2
LENGTH_XL = 3

# Font size bucket indices
FONT_SIZE_S = 0
FONT_SIZE_M = 1
FONT_SIZE_L = 2


@pytest.fixture
def sample_tables_path():
    """Path to sample_tables.xlsx."""
    test_data_dir = Path(__file__).parent / "data"
    sample_file = test_data_dir / "sample_tables.xlsx"
    if not sample_file.exists():
        pytest.skip(f"Sample file not found: {sample_file}")
    return sample_file


@pytest.fixture
def featurizer():
    """CellFeaturizer instance."""
    return CellFeaturizer()


def excel_to_zero_based(row, col):
    """Convert Excel 1-based coordinates to 0-based array indices."""
    return row - 1, col - 1


class TestSheet1:
    """Tests for Sheet1 table1: A1:D7"""
    
    def test_sheet1_bold_header(self, sample_tables_path, featurizer):
        """Test that A1:D1 has bold font (font_b = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet1")
        
        # A1:D1 in Excel = features[0, 0:4] in 0-based
        for col in range(4):  # A=0, B=1, C=2, D=3
            assert features[0, col, BOLD_IDX] == 1.0, f"Cell {chr(65+col)}1 should be bold"
        
        wb.close()
    
    def test_sheet1_formulas(self, sample_tables_path, featurizer):
        """Test that D2:D7 has formulas (has_formula = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet1")
        
        # D2:D7 in Excel = features[1:7, 3] in 0-based
        for row in range(1, 7):  # D2:D7 (rows 2-7 in Excel, 1-6 in 0-based)
            assert features[row, 3, HAS_FORMULA_IDX] == 1.0, f"Cell D{row+1} should have formula"
        
        wb.close()
    
    def test_sheet1_formatted(self, sample_tables_path, featurizer):
        """Test that A2:A7 has format (is_formatted = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet1")
        
        # A2:A7 in Excel = features[1:7, 0] in 0-based
        for row in range(1, 7):  # A2:A7 (rows 2-7 in Excel, 1-6 in 0-based)
            assert features[row, 0, IS_FORMATTED_IDX] == 1.0, f"Cell A{row+1} should be formatted"
        
        wb.close()
    
    def test_sheet1_bg_white(self, sample_tables_path, featurizer):
        """Test that A1:D1 has white background (bg_is_white = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet1")
        
        # A1:D1 in Excel = features[0, 0:4] in 0-based
        for col in range(4):  # A=0, B=1, C=2, D=3
            assert features[0, col, BG_COLOR_START + COLOR_WHITE] == 1.0, \
                f"Cell {chr(65+col)}1 should have white background"
        
        wb.close()


class TestSheet2:
    """Tests for Sheet2 - multiple tables"""
    
    def test_sheet2_table1_bg_yellow(self, sample_tables_path, featurizer):
        """Test that A1:D1 has yellow background (bg_is_yellow = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet2")
        
        for col in range(4):  # A1:D1
            assert features[0, col, BG_COLOR_START + COLOR_YELLOW] == 1.0, \
                f"Cell {chr(65+col)}1 should have yellow background"
        
        wb.close()
    
    def test_sheet2_table1_bold(self, sample_tables_path, featurizer):
        """Test that A1:D1 has bold font (font_b = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet2")
        
        for col in range(4):  # A1:D1
            assert features[0, col, BOLD_IDX] == 1.0, f"Cell {chr(65+col)}1 should be bold"
        
        wb.close()
    
    def test_sheet2_table1_font_black(self, sample_tables_path, featurizer):
        """Test that A1:D1 has black font (font_is_black = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet2")
        
        for col in range(4):  # A1:D1
            assert features[0, col, FONT_COLOR_START + COLOR_BLACK] == 1.0, \
                f"Cell {chr(65+col)}1 should have black font"
        
        wb.close()
    
    def test_sheet2_table2_bold_header(self, sample_tables_path, featurizer):
        """Test that G5:I5 has bold font (font_b = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet2")
        
        # G5:I5 in Excel = features[4, 6:9] in 0-based
        for col in range(6, 9):  # G=6, H=7, I=8
            assert features[4, col, BOLD_IDX] == 1.0, f"Cell {chr(71+col-6)}5 should be bold"
        
        wb.close()
    
    def test_sheet2_table2_empty_cell(self, sample_tables_path, featurizer):
        """Test that H7 is visibly empty (is_visibly_empty = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet2")
        
        # H7 in Excel = features[6, 7] in 0-based
        assert features[6, 7, IS_VISIBLY_EMPTY_IDX] == 1.0, "Cell H7 should be visibly empty"
        
        wb.close()
    
    def test_sheet2_table2_borders(self, sample_tables_path, featurizer):
        """Test that G5 has borders (border_left, border_top, border_bottom = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet2")
        
        # G5 in Excel = features[4, 6] in 0-based
        assert features[4, 6, BORDER_LEFT_IDX] == 1.0, "Cell G5 should have left border"
        assert features[4, 6, BORDER_TOP_IDX] == 1.0, "Cell G5 should have top border"
        assert features[4, 6, BORDER_BOTTOM_IDX] == 1.0, "Cell G5 should have bottom border"
        
        wb.close()
    
    def test_sheet2_table3_excel_table(self, sample_tables_path, featurizer):
        """Test that C14:F19 is an Excel table (is_in_table = 1, is_in_table_header = 1 for C14:F14)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        sheet = wb["Sheet2"]
        
        # Get Excel tables
        excel_tables = get_excel_tables(sheet)
        features, _ = featurizer.featurize_sheet(wb, "Sheet2", excel_tables=excel_tables)
        
        # C14:F19 in Excel = features[13:19, 2:6] in 0-based
        # All cells in table should have is_in_table = 1
        for row in range(13, 19):  # Rows 14-19
            for col in range(2, 6):  # Columns C-F
                assert features[row, col, IS_IN_TABLE_IDX] == 1.0, \
                    f"Cell {chr(67+col-2)}{row+1} should be in Excel table"
        
        # Header row C14:F14 should have is_in_table_header = 1
        for col in range(2, 6):  # Columns C-F
            assert features[13, col, IS_IN_TABLE_HEADER_IDX] == 1.0, \
                f"Cell {chr(67+col-2)}14 should be in Excel table header"
        
        wb.close()


class TestSheet3:
    """Tests for Sheet3 - tables with merged cells"""
    
    def test_sheet3_table1_large_font(self, sample_tables_path, featurizer):
        """Test that A1:C2 has large font (font_l = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, metadata = featurizer.featurize_sheet(wb, "Sheet3")
        
        # A1:C2 in Excel = features[0:2, 0:3] in 0-based
        for row in range(2):  # Rows 1-2
            for col in range(3):  # Columns A-C
                assert features[row, col, FONT_SIZE_START + FONT_SIZE_L] == 1.0, \
                    f"Cell {chr(65+col)}{row+1} should have large font"
        
        wb.close()
    
    def test_sheet3_table1_bold(self, sample_tables_path, featurizer):
        """Test that A1:C2 has bold font (font_b = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet3")
        
        for row in range(2):  # Rows 1-2
            for col in range(3):  # Columns A-C
                assert features[row, col, BOLD_IDX] == 1.0, \
                    f"Cell {chr(65+col)}{row+1} should be bold"
        
        wb.close()
    
    def test_sheet3_table1_font_red(self, sample_tables_path, featurizer):
        """Test that A1:C2 has red font (font_is_red = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet3")
        
        for row in range(2):  # Rows 1-2
            for col in range(3):  # Columns A-C
                assert features[row, col, FONT_COLOR_START + COLOR_RED] == 1.0, \
                    f"Cell {chr(65+col)}{row+1} should have red font"
        
        wb.close()
    
    def test_sheet3_table1_merged_cells(self, sample_tables_path, featurizer):
        """Test merged cell indicators for Sheet3 table1."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet3")
        
        # A1 merged with A2: A1 should have is_merged_bottom, A2 should have is_merged_top
        assert features[0, 0, MERGE_BOTTOM_IDX] == 1.0, "Cell A1 should have merged_bottom"
        assert features[1, 0, MERGE_TOP_IDX] == 1.0, "Cell A2 should have merged_top"
        
        # A4 merged with A5: A4 should have is_merged_bottom, A5 should have is_merged_top
        assert features[3, 0, MERGE_BOTTOM_IDX] == 1.0, "Cell A4 should have merged_bottom"
        assert features[4, 0, MERGE_TOP_IDX] == 1.0, "Cell A5 should have merged_top"
        
        # B6:C7 merged: B6 should have merged_right and merged_bottom
        assert features[5, 1, MERGE_RIGHT_IDX] == 1.0, "Cell B6 should have merged_right"
        assert features[5, 1, MERGE_BOTTOM_IDX] == 1.0, "Cell B6 should have merged_bottom"
        # C6 should have merged_left and merged_bottom
        assert features[5, 2, MERGE_LEFT_IDX] == 1.0, "Cell C6 should have merged_left"
        assert features[5, 2, MERGE_BOTTOM_IDX] == 1.0, "Cell C6 should have merged_bottom"
        # B7 should have merged_right and merged_top
        assert features[6, 1, MERGE_RIGHT_IDX] == 1.0, "Cell B7 should have merged_right"
        assert features[6, 1, MERGE_TOP_IDX] == 1.0, "Cell B7 should have merged_top"
        # C7 should have merged_left and merged_top
        assert features[6, 2, MERGE_LEFT_IDX] == 1.0, "Cell C7 should have merged_left"
        assert features[6, 2, MERGE_TOP_IDX] == 1.0, "Cell C7 should have merged_top"
        
        wb.close()
    
    def test_sheet3_table2_bg_yellow(self, sample_tables_path, featurizer):
        """Test that E7:G8 has yellow background (bg_is_yellow = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet3")
        
        # E7:G8 in Excel = features[6:8, 4:7] in 0-based
        for row in range(6, 8):  # Rows 7-8
            for col in range(4, 7):  # Columns E-G
                assert features[row, col, BG_COLOR_START + COLOR_YELLOW] == 1.0, \
                    f"Cell {chr(69+col-4)}{row+1} should have yellow background"
        
        wb.close()
    
    def test_sheet3_table2_bold_header(self, sample_tables_path, featurizer):
        """Test that E7:G8 has bold font (font_b = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet3")
        
        for row in range(6, 8):  # Rows 7-8
            for col in range(4, 7):  # Columns E-G
                assert features[row, col, BOLD_IDX] == 1.0, \
                    f"Cell {chr(69+col-4)}{row+1} should be bold"
        
        wb.close()
    
    def test_sheet3_table2_merged_cells(self, sample_tables_path, featurizer):
        """Test merged cell indicators for Sheet3 table2."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet3")
        
        # E10 merged with E11: E10 should have merged_bottom, E11 should have merged_top
        assert features[9, 4, MERGE_BOTTOM_IDX] == 1.0, "Cell E10 should have merged_bottom"
        assert features[10, 4, MERGE_TOP_IDX] == 1.0, "Cell E11 should have merged_top"
        # E11 should also be visibly empty (non-top-left in merged range)
        assert features[10, 4, IS_VISIBLY_EMPTY_IDX] == 1.0, "Cell E11 should be visibly empty"
        
        # F12:G13 merged: F12 should have merged_right and merged_bottom
        assert features[11, 5, MERGE_RIGHT_IDX] == 1.0, "Cell F12 should have merged_right"
        assert features[11, 5, MERGE_BOTTOM_IDX] == 1.0, "Cell F12 should have merged_bottom"
        # G12 should have merged_left and merged_bottom
        assert features[11, 6, MERGE_LEFT_IDX] == 1.0, "Cell G12 should have merged_left"
        assert features[11, 6, MERGE_BOTTOM_IDX] == 1.0, "Cell G12 should have merged_bottom"
        # F13 should have merged_right and merged_top
        assert features[12, 5, MERGE_RIGHT_IDX] == 1.0, "Cell F13 should have merged_right"
        assert features[12, 5, MERGE_TOP_IDX] == 1.0, "Cell F13 should have merged_top"
        # G13 should have merged_left and merged_top
        assert features[12, 6, MERGE_LEFT_IDX] == 1.0, "Cell G13 should have merged_left"
        assert features[12, 6, MERGE_TOP_IDX] == 1.0, "Cell G13 should have merged_top"
        # G13 should also be visibly empty
        assert features[12, 6, IS_VISIBLY_EMPTY_IDX] == 1.0, "Cell G13 should be visibly empty"
        
        wb.close()


class TestSheet4:
    """Tests for Sheet4 - tables with comments, Excel tables, and color buckets"""
    
    def test_sheet4_table1_comment(self, sample_tables_path, featurizer):
        """Test that A1 has a comment (has_comment = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet4")
        
        # A1 in Excel = features[0, 0] in 0-based
        assert features[0, 0, HAS_COMMENT_IDX] == 1.0, "Cell A1 should have a comment"
        
        wb.close()
    
    def test_sheet4_table1_digit_proportion(self, sample_tables_path, featurizer):
        """Test that A2 has digit_proportion = 0.07692307692."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet4")
        
        # A2 in Excel = features[1, 0] in 0-based
        digit_prop = features[1, 0, DIGIT_PROPORTION_IDX]
        # Should be approximately 0.07692307692 (1 digit out of 13 non-whitespace chars)
        assert abs(digit_prop - 0.07692307692) < 0.001, \
            f"Cell A2 should have digit_proportion ≈ 0.07692307692, got {digit_prop}"
        
        wb.close()
    
    def test_sheet4_table1_length_l(self, sample_tables_path, featurizer):
        """Test that A2 has length bucket l (size_l = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet4")
        
        # A2 in Excel = features[1, 0] in 0-based
        assert features[1, 0, LENGTH_START + LENGTH_L] == 1.0, "Cell A2 should have length bucket l"
        
        wb.close()
    
    def test_sheet4_table1_locked(self, sample_tables_path, featurizer):
        """Test that A2 is locked (is_locked = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet4")
        
        # A2 in Excel = features[1, 0] in 0-based
        assert features[1, 0, IS_LOCKED_IDX] == 1.0, "Cell A2 should be locked"
        
        wb.close()
    
    def test_sheet4_table2_excel_table(self, sample_tables_path, featurizer):
        """Test that C6:H14 is an Excel table (is_in_table = 1, is_in_table_header = 1 for C6:H6)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        sheet = wb["Sheet4"]
        
        # Get Excel tables
        excel_tables = get_excel_tables(sheet)
        features, _ = featurizer.featurize_sheet(wb, "Sheet4", excel_tables=excel_tables)
        
        # C6:H14 in Excel = features[5:14, 2:8] in 0-based
        # All cells in table should have is_in_table = 1
        for row in range(5, 14):  # Rows 6-14
            for col in range(2, 8):  # Columns C-H
                assert features[row, col, IS_IN_TABLE_IDX] == 1.0, \
                    f"Cell {chr(67+col-2)}{row+1} should be in Excel table"
        
        # Header row C6:H6 should have is_in_table_header = 1
        for col in range(2, 8):  # Columns C-H
            assert features[5, col, IS_IN_TABLE_HEADER_IDX] == 1.0, \
                f"Cell {chr(67+col-2)}6 should be in Excel table header"
        
        wb.close()
    
    def test_sheet4_color_buckets_red(self, sample_tables_path, featurizer):
        """Test that A17:A19 has red background (bg_is_red = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet4")
        
        # A17:A19 in Excel = features[16:19, 0] in 0-based
        for row in range(16, 19):  # Rows 17-19
            assert features[row, 0, BG_COLOR_START + COLOR_RED] == 1.0, \
                f"Cell A{row+1} should have red background"
        
        wb.close()
    
    def test_sheet4_color_buckets_gray(self, sample_tables_path, featurizer):
        """Test that B17:B19 has gray background (bg_is_gray = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet4")
        
        # B17:B19 in Excel = features[16:19, 1] in 0-based
        for row in range(16, 19):  # Rows 17-19
            assert features[row, 1, BG_COLOR_START + COLOR_GRAY] == 1.0, \
                f"Cell B{row+1} should have gray background"
        
        wb.close()
    
    def test_sheet4_color_buckets_green(self, sample_tables_path, featurizer):
        """Test that C17:C19 has green background (bg_is_green = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet4")
        
        # C17:C19 in Excel = features[16:19, 2] in 0-based
        for row in range(16, 19):  # Rows 17-19
            assert features[row, 2, BG_COLOR_START + COLOR_GREEN] == 1.0, \
                f"Cell C{row+1} should have green background"
        
        wb.close()
    
    def test_sheet4_color_buckets_blue(self, sample_tables_path, featurizer):
        """Test that D17:D19 has blue background (bg_is_blue = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet4")
        
        # D17:D19 in Excel = features[16:19, 3] in 0-based
        for row in range(16, 19):  # Rows 17-19
            assert features[row, 3, BG_COLOR_START + COLOR_BLUE] == 1.0, \
                f"Cell D{row+1} should have blue background"
        
        wb.close()
    
    def test_sheet4_color_buckets_yellow(self, sample_tables_path, featurizer):
        """Test that E17:E19 has yellow background (bg_is_yellow = 1)."""
        wb = load_workbook(sample_tables_path, data_only=False)
        features, _ = featurizer.featurize_sheet(wb, "Sheet4")
        
        # E17:E19 in Excel = features[16:19, 4] in 0-based
        for row in range(16, 19):  # Rows 17-19
            assert features[row, 4, BG_COLOR_START + COLOR_YELLOW] == 1.0, \
                f"Cell E{row+1} should have yellow background"
        
        wb.close()


class TestFeatureCount:
    """Test that feature count is correct."""
    
    def test_feature_count(self, sample_tables_path, featurizer):
        """Verify that all sheets produce 43 features."""
        wb = load_workbook(sample_tables_path, data_only=False)
        
        for sheet_name in wb.sheetnames:
            features, _ = featurizer.featurize_sheet(wb, sheet_name)
            assert features.shape[2] == NUM_FEATURES, \
                f"Sheet {sheet_name} should have {NUM_FEATURES} features, got {features.shape[2]}"
            assert NUM_FEATURES == 43, "NUM_FEATURES should be 43"
        
        wb.close()

