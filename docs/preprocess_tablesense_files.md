# Preprocess TableSense Files

## Overview

Manually download the TableSense dataset from [Hugging Face](https://huggingface.co/datasets/kl3269/tablesense/tree/main). The files referenced in the TableSense [Github](https://github.com/microsoft/TableSense/blob/main/README.md) dataset is in the .xls binary format. The HF project has converted the files to .xlsx format that is readable with openpyxl. 

Download data.tar.gz file from HF and use "tar -xvf" command to extract the data into any directory. Copy the annotations.jsonl from HF into training_data/tablesense/annotations/annotations_hf.jsonl

`preprocess_tablesense_files.py` reads the Hugging Face `annotations_hf.jsonl`, verifies each referenced Excel file can be opened with `openpyxl`, copies the readable files into `training_data/tablesense/files`, and creates `annotations.jsonl` in a TableSense2 jsonl format.

## Requirements

- Python 3.8+
- `openpyxl>=3.1.0`
- The downloaded TableSense dataset extracted into a directory that you can point to with `<data_download_dir>`.
- The TableSense2 root directory (`TABLESENSE_DIR`) that contains `training_data/tablesense/annotations`.

## Usage

```bash
# Activate the project coda/venv environment if needed
python utils/preprocess_tablesense_files.py <data_download_dir> <tablesense_root>
```

### Optional flags

- `--hf-annotations`: path to the source Hugging Face annotations file (default: `training_data/tablesense/annotations/annotations_hf.jsonl` relative to `tablesense2_root`).
- `--output-annotations`: destination for the converted annotations file (default: `training_data/tablesense/annotations/annotations.jsonl` relative to `tablesense2_root`).

### Example

```bash
# In this example HF data has been downloaded to /tmp/tablesense_hf_data and the TableSense2 project is in /Users/me/dev/TableSense2 directory
python utils/preprocess_tablesense_files.py \
  /tmp/tablesense_hf_data \
  /Users/me/dev/TableSense2
```

## Behavior

- Processes each line in `annotations_hf.jsonl`.
- Prints a warning to STDOUT if the referenced clean file is missing or cannot be opened by `openpyxl`.
- Copies the valid Excel files into `training_data/tablesense/files/<original_file>`, creating directories if needed.
- Generates `annotations.jsonl` with the TableSense2 attributes (`file_name`, `file_path`, `sheet_name`, `split_code`, `table_regions`).
- Outputs a summary of processed, converted, and skipped entries plus the locations of the generated artifacts.

## Next Steps

- Run the script before running downstream training or evaluation so the TableSense2 dataset is fully populated.

