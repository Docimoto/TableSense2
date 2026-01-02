"""
Script to identify problematic (corrupted/unreadable) annotation files.
"""

import json
from pathlib import Path
from collections import defaultdict
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_io.annotations import AnnotationLoader
from utils.excel_utils import load_workbook


def check_file_problematic(ann: dict, project_root: Path) -> tuple[bool, str]:
    """
    Check if a file has problems loading or accessing.
    
    Returns:
        Tuple of (is_problematic, error_message)
        is_problematic: True if file has issues, False otherwise
        error_message: Description of the problem if problematic, empty string otherwise
    """
    base_path = project_root / ann['file_path']
    file_name = ann['file_name']
    file_path = base_path / file_name
    
    # Try to find matching file if exact match doesn't exist
    if not file_path.exists():
        all_files = list(base_path.glob("*.xlsx"))
        matching_file = None
        for f in all_files:
            if f.name.endswith(file_name):
                matching_file = f
                break
        if matching_file:
            file_path = matching_file
        else:
            return True, "File not found"
    
    # Try to load the workbook
    try:
        workbook = load_workbook(file_path, data_only=False)
    except Exception as e:
        return True, f"Failed to load workbook: {str(e)}"
    
    # Check if sheet exists
    sheet_name = ann['sheet_name']
    if sheet_name not in workbook.sheetnames:
        workbook.close()
        return True, f"Sheet '{sheet_name}' not found in workbook"
    
    # Try to access the sheet to check for XML parsing errors
    try:
        sheet = workbook[sheet_name]
        # Try to access sheet properties
        _ = sheet.max_row
        _ = sheet.max_column
        # Try to iterate over a few cells to catch parsing errors
        for row in sheet.iter_rows(min_row=1, max_row=min(10, sheet.max_row), values_only=False):
            if row:
                break
    except Exception as e:
        workbook.close()
        return True, f"Failed to access sheet data: {str(e)}"
    
    workbook.close()
    return False, ""


def main():
    """Generate a markdown report of problematic files."""
    project_root = Path(__file__).parent.parent
    
    # Load annotations
    loader = AnnotationLoader(
        project_root=project_root,
        dataset_names=["tablesense"],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        split_seed=42,
    )
    loader.load_annotations()
    
    # Get all annotations
    all_annotations = loader.annotations
    
    print(f"Checking {len(all_annotations)} annotations for problematic files...")
    
    # Check which files are problematic
    problematic_annotations = []
    for i, ann in enumerate(all_annotations):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_annotations)}...")
        
        is_problematic, error_msg = check_file_problematic(ann, project_root)
        if is_problematic:
            problematic_annotations.append({
                'annotation': ann,
                'error': error_msg
            })
    
    # Group by split
    problematic_by_split = defaultdict(list)
    for item in problematic_annotations:
        ann = item['annotation']
        workbook_id = ann['workbook_id']
        split = loader.workbook_splits.get(workbook_id, 'unknown')
        problematic_by_split[split].append(item)
    
    # Group by error type
    error_types = defaultdict(list)
    for item in problematic_annotations:
        error_type = item['error'].split(':')[0] if ':' in item['error'] else item['error']
        error_types[error_type].append(item)
    
    # Generate markdown report
    report_lines = [
        "# Problematic Files Report",
        "",
        f"**Total Annotations:** {len(all_annotations)}",
        f"**Problematic Files:** {len(problematic_annotations)}",
        f"**Valid Files:** {len(all_annotations) - len(problematic_annotations)}",
        "",
        "## Summary by Split",
        "",
        "| Split | Problematic | Total | Percentage |",
        "|-------|-------------|-------|------------|",
    ]
    
    for split in ['train', 'val', 'test']:
        split_annotations = loader.get_annotations_by_split(split)
        problematic_in_split = problematic_by_split[split]
        percentage = (len(problematic_in_split) / len(split_annotations) * 100) if split_annotations else 0
        report_lines.append(f"| {split} | {len(problematic_in_split)} | {len(split_annotations)} | {percentage:.1f}% |")
    
    report_lines.extend([
        "",
        "## Summary by Error Type",
        "",
        "| Error Type | Count |",
        "|------------|-------|",
    ])
    
    for error_type, items in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
        report_lines.append(f"| {error_type} | {len(items)} |")
    
    report_lines.extend([
        "",
        "## Problematic Files by Split",
        "",
    ])
    
    # List problematic files by split
    for split in ['train', 'val', 'test']:
        problematic_in_split = problematic_by_split[split]
        if problematic_in_split:
            report_lines.append(f"### {split.upper()} Split ({len(problematic_in_split)} files)")
            report_lines.append("")
            report_lines.append("| File Name | File Path | Sheet Name | Error | Workbook ID |")
            report_lines.append("|-----------|-----------|------------|-------|-------------|")
            
            # Sort by file_name for easier reading
            problematic_in_split_sorted = sorted(problematic_in_split, key=lambda x: x['annotation']['file_name'])
            
            for item in problematic_in_split_sorted:
                ann = item['annotation']
                file_name = ann['file_name']
                file_path = ann['file_path']
                sheet_name = ann['sheet_name']
                error = item['error']
                workbook_id = ann['workbook_id']
                # Truncate long error messages
                error_display = error[:100] + "..." if len(error) > 100 else error
                report_lines.append(f"| `{file_name}` | `{file_path}` | `{sheet_name}` | `{error_display}` | `{workbook_id}` |")
            
            report_lines.append("")
    
    # Add a section with unique problematic files
    unique_problematic_files = {}
    for item in problematic_annotations:
        ann = item['annotation']
        file_key = (ann['file_path'], ann['file_name'])
        if file_key not in unique_problematic_files:
            unique_problematic_files[file_key] = {
                'file_path': ann['file_path'],
                'file_name': ann['file_name'],
                'errors': []
            }
        unique_problematic_files[file_key]['errors'].append({
            'sheet': ann['sheet_name'],
            'error': item['error']
        })
    
    report_lines.extend([
        "## Unique Problematic Files",
        "",
        f"**Total Unique Problematic Files:** {len(unique_problematic_files)}",
        "",
        "| File Name | File Path | Expected Location | Issues |",
        "|-----------|-----------|-------------------|--------|",
    ])
    
    for (file_path, file_name), file_info in sorted(unique_problematic_files.items(), key=lambda x: x[0][1]):
        expected_location = project_root / file_path / file_name
        issues = f"{len(file_info['errors'])} sheet(s) with errors"
        report_lines.append(f"| `{file_name}` | `{file_path}` | `{expected_location}` | {issues} |")
    
    # Add detailed error breakdown for unique files
    report_lines.extend([
        "",
        "### Detailed Error Breakdown",
        "",
    ])
    
    for (file_path, file_name), file_info in sorted(unique_problematic_files.items(), key=lambda x: x[0][1]):
        report_lines.append(f"#### `{file_name}`")
        report_lines.append("")
        for error_info in file_info['errors']:
            report_lines.append(f"- **Sheet:** `{error_info['sheet']}` - {error_info['error']}")
        report_lines.append("")
    
    # Save report
    report_path = project_root / "docs" / "problematic_files_report.md"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {report_path}")
    print(f"\nSummary:")
    print(f"  Total annotations: {len(all_annotations)}")
    print(f"  Problematic files: {len(problematic_annotations)}")
    print(f"  Valid files: {len(all_annotations) - len(problematic_annotations)}")
    print(f"  Unique problematic files: {len(unique_problematic_files)}")
    
    # Print error type summary
    print(f"\nError Types:")
    for error_type, items in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {error_type}: {len(items)}")


if __name__ == "__main__":
    main()
