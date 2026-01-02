#!/usr/bin/env python3
"""
Utility script that converts the original TableSense annotations into the
TableSense2 format while validating Excel files and copying them into place.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.excel_utils import load_workbook


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert the Hugging Face TableSense dataset annotations into the "
            "TableSense2 JSONL format and copy the corresponding Excel files."
        )
    )
    parser.add_argument(
        "data_download_dir",
        type=Path,
        help="Directory where the downloaded TableSense data was extracted.",
    )
    parser.add_argument(
        "tablesense_dir",
        type=Path,
        help="Root directory of the TableSense2 project.",
    )
    parser.add_argument(
        "--hf-annotations",
        type=Path,
        default=Path("training_data/tablesense/annotations/annotations_hf.jsonl"),
        help="Path to the Hugging Face annotations JSONL file (relative to table sense root).",
    )
    parser.add_argument(
        "--output-annotations",
        type=Path,
        default=Path("training_data/tablesense/annotations/annotations.jsonl"),
        help="Destination path for the converted TableSense2 annotations (relative to table sense root).",
    )
    return parser.parse_args()


def resolve_relative_path(base: Path, candidate: Path) -> Path:
    """Return an absolute path, interpreting non-absolute candidates relative to base."""
    return candidate if candidate.is_absolute() else base / candidate


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield dictionaries parsed from a JSON Lines file."""
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def verify_workbook(excel_path: Path) -> bool:
    """Return True if openpyxl can load the workbook."""
    try:
        workbook = load_workbook(excel_path, read_only=True, data_only=True)
        workbook.close()
        return True
    except Exception as exc:
        print(f"{excel_path}: {type(exc).__name__}: {exc}")
        return False


def copy_source_file(source: Path, destination: Path) -> None:
    """Copy the Excel file into the TableSense2 files directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def convert_entry(
    entry: Dict[str, Any],
    data_dir: Path,
    tablesense_files_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Validate an entry and prepare the converted TableSense2 record."""
    clean_file = entry.get("clean_file")
    original_file = entry.get("original_file")
    if not clean_file or not original_file:
        print("Skipping invalid annotation (missing file names).")
        return None

    source_path = data_dir / clean_file
    if not source_path.exists():
        print(f"Skipping {clean_file}: file not found at {source_path}")
        return None

    if not verify_workbook(source_path):
        print(f"Skipping {clean_file}: unable to open with openpyxl")
        return None

    destination_path = tablesense_files_dir / original_file
    copy_source_file(source_path, destination_path)

    return {
        "file_name": original_file,
        "file_path": "training_data/tablesense/files",
        "sheet_name": entry.get("sheet_name"),
        "split_code": entry.get("split"),
        "table_regions": entry.get("table_regions"),
    }


def write_jsonl(path: Path, entries: List[Dict[str, Any]]) -> None:
    """Overwrite the output JSONL file with the converted entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_fh:
        for entry in entries:
            json_line = json.dumps(entry, ensure_ascii=False)
            out_fh.write(json_line + "\n")


def main() -> int:
    args = parse_args()
    data_dir = args.data_download_dir.expanduser().resolve()
    tablesense_dir = args.tablesense_dir.expanduser().resolve()

    if not data_dir.is_dir():
        print(f"Error: data download directory does not exist: {data_dir}")
        return 1
    if not tablesense_dir.is_dir():
        print(f"Error: table sense directory does not exist: {tablesense_dir}")
        return 1

    hf_annotations = resolve_relative_path(tablesense_dir, args.hf_annotations)
    output_annotations = resolve_relative_path(tablesense_dir, args.output_annotations)
    files_dir = tablesense_dir / "training_data" / "tablesense" / "files"

    if not hf_annotations.exists():
        print(f"Error: Hugging Face annotations file not found: {hf_annotations}")
        return 1

    converted_entries: List[Dict[str, Any]] = []
    skipped = 0
    processed = 0

    for annotation in read_jsonl(hf_annotations):
        processed += 1
        entry = convert_entry(annotation, data_dir, files_dir)
        if entry:
            converted_entries.append(entry)
        else:
            skipped += 1

    write_jsonl(output_annotations, converted_entries)

    print("\nSummary")
    print(f"  processed entries: {processed}")
    print(f"  converted entries: {len(converted_entries)}")
    print(f"  skipped entries:   {skipped}")
    print(f"  annotations written to: {output_annotations}")
    print(f"  files copied to: {files_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


