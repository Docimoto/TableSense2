#!/usr/bin/env python3
"""
Script to convert tablesense annotation format to TableSense2 format.

Usage:
    python fix_annotations.py <input_annotation_file> <output_annotation_file> <output_file_path>
"""

import json
import sys
from typing import Dict, List, Any


def validate_required_fields(data: Dict[str, Any], line_num: int) -> List[str]:
    """Check for required fields and return list of missing fields."""
    required_fields = ['original_file', 'sheet_name', 'split', 'table_regions']
    missing = []
    
    for field in required_fields:
        if field not in data or data[field] is None:
            missing.append(field)
    
    return missing


def process_annotations(input_file: str, output_file: str, output_file_path: str) -> tuple:
    """
    Process annotation file and convert format.
    
    Returns:
        tuple: (total_lines, processed_lines, errors_list)
    """
    total_lines = 0
    processed_lines = 0
    errors = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as in_fh, \
             open(output_file, 'w', encoding='utf-8') as out_fh:
            
            for line in in_fh:
                total_lines += 1
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Parse JSON
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    error_msg = f"Line {total_lines}: JSON parse error - {str(e)}"
                    errors.append(error_msg)
                    print(f"ERROR: {error_msg}", file=sys.stderr)
                    continue
                
                # Check for required fields
                missing_fields = validate_required_fields(data, total_lines)
                if missing_fields:
                    error_msg = f"Line {total_lines}: Missing required fields: {', '.join(missing_fields)}"
                    errors.append(error_msg)
                    print(f"ERROR: {error_msg}", file=sys.stderr)
                    continue
                
                # Validate table_regions is a list
                if not isinstance(data['table_regions'], list):
                    error_msg = f"Line {total_lines}: 'table_regions' must be an array"
                    errors.append(error_msg)
                    print(f"ERROR: {error_msg}", file=sys.stderr)
                    continue
                
                # Create output JSON structure
                output_data = {
                    'file_name': data['original_file'],
                    'file_path': output_file_path,
                    'sheet_name': data['sheet_name'],
                    'split_code': data['split'],
                    'table_regions': data['table_regions']
                }
                
                # Write output JSON line
                output_line = json.dumps(output_data, ensure_ascii=False)
                out_fh.write(output_line + '\n')
                processed_lines += 1
                
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"ERROR: I/O error: {e}", file=sys.stderr)
        sys.exit(1)
    
    return total_lines, processed_lines, errors


def main():
    """Main function."""
    if len(sys.argv) != 4:
        print("Usage: python fix_annotations.py <input_annotation_file> <output_annotation_file> <output_file_path>", 
              file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    output_file_path = sys.argv[3]
    
    # Process annotations
    total_lines, processed_lines, errors = process_annotations(
        input_file, output_file, output_file_path
    )
    
    error_count = len(errors)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Total lines read:    {total_lines}")
    print(f"Successfully processed: {processed_lines}")
    print(f"Errors encountered:    {error_count}")
    print("=" * 70)
    
    if errors:
        print("\nERROR DETAILS:")
        print("-" * 70)
        for error in errors:
            print(error)
        print("-" * 70)
        print("\nWARNING: Some lines could not be processed. Please review the errors above.")
        sys.exit(1)
    else:
        print("\nConversion completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()

