# Utils Directory

This directory contains utility scripts for Excel file processing and analysis.

## Table of Contents

- [detect_excel_tables.py](#detect_excel_tablespy) - Detect Excel table objects in worksheets

---

## detect_excel_tables.py

### Purpose

This utility script scans Excel files (`.xlsx`) in a directory and detects worksheets that contain Excel table objects (formatted tables created via Excel's "Format as Table" feature). The script generates a JSONL (JSON Lines) file containing metadata about files, sheets, and table regions.

### Use Cases

- **Dataset preparation**: Identify which Excel files contain structured table objects for training data
- **Data quality assessment**: Quickly scan directories to find files with Excel table objects
- **Annotation pipeline**: Generate initial annotations for files that already have Excel table objects defined

### Requirements

- Python 3.8+
- `openpyxl>=3.1.0`
- Conda environment: `tablesense2` (see project root `environment.yml`)

### Usage

```bash
# Activate conda environment
conda activate tablesense2

# Basic usage
python utils/detect_excel_tables.py <input_dir> <output_dir>

# With file limit (for testing)
python utils/detect_excel_tables.py <input_dir> <output_dir> --limit 5
```

### Arguments

- `input_dir` (required): Directory containing Excel files (`.xlsx`) to process
- `output_dir` (required): Directory where the output JSONL file will be written
- `--limit` (optional): Limit the number of files to process (useful for testing)

### Output Format

The script creates a file named `annotations_tables.jsonl` in the output directory. Each line contains a JSON object with the following structure:

```json
{
  "file_name": "/absolute/path/to/file.xlsx",
  "sheet_name": "Sheet1",
  "table_regions": ["C8:H18", "A1:B5"]
}
```

**Fields:**
- `file_name`: Absolute file path to the Excel file
- `sheet_name`: Name of the worksheet containing table objects
- `table_regions`: Array of table region strings in Excel range format (e.g., "C8:H18")

**Note:** Only worksheets that contain at least one Excel table object are included in the output.

### Examples

#### Process all files in a directory

```bash
python utils/detect_excel_tables.py \
  training_data/tablesense/files \
  training_data/tablesense/files
```

#### Test with first 5 files (alphabetically sorted)

```bash
python utils/detect_excel_tables.py \
  training_data/tablesense/files \
  training_data/tablesense/files \
  --limit 5
```

### Behavior

- **File processing**: Processes all `.xlsx` files in the input directory, sorted alphabetically
- **Worksheet iteration**: Examines all worksheets in each Excel file
- **Table detection**: Uses `openpyxl`'s `Worksheet.tables` to detect Excel table objects
- **Error handling**: Stops immediately on the first corrupted file or error (no error recovery)
- **Output**: Creates output directory if it doesn't exist

### Technical Details

- Uses `openpyxl` in read-only mode for efficient processing
- Extracts table regions from the `ref` attribute of table objects
- Writes JSONL format (one JSON object per line) for easy streaming and processing
- Uses absolute file paths for `file_name` to ensure unambiguous file identification

### Limitations

- Only processes `.xlsx` files (not `.xls` legacy format)
- Only detects Excel table objects (not manually formatted tables)
- Stops on first error (no partial results for corrupted files)

---

## Future Utilities

Additional utility scripts will be documented here as they are added to the project.

