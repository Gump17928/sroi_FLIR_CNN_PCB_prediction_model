#!/usr/bin/env python3
"""
Configuration File Finder
Scans the inputs directory structure and displays all FLIR and thermistor files
organized by board and test, making it easy to populate JSON config files.

Usage:
    python config_file_finder.py
    python config_file_finder.py --inputs_dir custom/path
    python config_file_finder.py --format json
"""

import os
import glob
import argparse
from pathlib import Path
from collections import defaultdict


def find_files_by_pattern(directory, patterns):
    """Find files matching any of the given patterns."""
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
    return sorted(files)


def scan_inputs_directory(inputs_dir):
    """
    Scan inputs directory and organize files by board and test.
    
    Returns:
        dict: {board_name: {test_id: {file_type: [files]}}}
    """
    structure = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    if not os.path.exists(inputs_dir):
        print(f"Warning: Inputs directory not found: {inputs_dir}")
        return structure
    
    # Patterns for FLIR files (both .flir and .csv exports)
    flir_patterns = ['*.flir', '*.FLIR', '*.fir', '*.FIR', '*FLIR*.csv', '*Camera*.csv']
    
    # Patterns for thermistor files (both device formats)
    therm_patterns = ['USB-TEMP*.txt', 'USBTEMP*.txt', 'USB-TEMP*.csv', 'USBTEMP*.csv', 
                      '*usb_temp*.csv', '*USB_TEMP*.csv']
    
    # Scan all subdirectories
    for root, dirs, files in os.walk(inputs_dir):
        rel_path = os.path.relpath(root, inputs_dir)
        path_parts = Path(rel_path).parts
        
        # Skip the root inputs directory itself
        if rel_path == '.':
            continue
        
        # Try to identify board and test from path structure
        board_name = path_parts[0] if len(path_parts) > 0 else 'Unknown'
        
        # Check if this is a test subdirectory (handle "Test 1", "Test1", "Test_1", etc.)
        test_id = None
        for part in path_parts:
            # Remove spaces and check for Test pattern
            part_no_space = part.replace(' ', '')
            if part_no_space.startswith('Test') and any(c.isdigit() for c in part_no_space):
                test_id = part  # Keep original with spaces
                break
        
        # If no test subdirectory, treat board-level folder as Test1
        if test_id is None and len(path_parts) == 1:
            test_id = 'Board_Level'
        elif test_id is None:
            # Skip intermediate directories
            continue
        
        # Find FLIR files
        for pattern in flir_patterns:
            for file in glob.glob(os.path.join(root, pattern)):
                rel_file = os.path.relpath(file, inputs_dir)
                if rel_file not in structure[board_name][test_id]['flir']:
                    structure[board_name][test_id]['flir'].append(rel_file)
        
        # Find thermistor files
        for pattern in therm_patterns:
            for file in glob.glob(os.path.join(root, pattern)):
                rel_file = os.path.relpath(file, inputs_dir)
                filename = os.path.basename(file).upper()
                # Categorize by AIR vs SAND (check filename for keywords)
                if 'AIR' in filename or 'USBTEMPAI' in filename:
                    if rel_file not in structure[board_name][test_id]['therm_air']:
                        structure[board_name][test_id]['therm_air'].append(rel_file)
                elif 'SAND' in filename:
                    if rel_file not in structure[board_name][test_id]['therm_sand']:
                        structure[board_name][test_id]['therm_sand'].append(rel_file)
                else:
                    # Default to air if can't determine
                    if rel_file not in structure[board_name][test_id]['therm_air']:
                        structure[board_name][test_id]['therm_air'].append(rel_file)
    
    return structure


def display_text_format(structure, inputs_dir):
    """Display found files in human-readable text format."""
    print("=" * 80)
    print("THERMAL CALIBRATION FILES FOUND")
    print(f"Inputs Directory: {inputs_dir}")
    print("=" * 80)
    print()
    
    if not structure:
        print("No files found. Check your inputs directory structure.")
        return
    
    for board_name in sorted(structure.keys()):
        print(f"📋 BOARD: {board_name}")
        print("-" * 80)
        
        for test_id in sorted(structure[board_name].keys()):
            print(f"  🧪 TEST: {test_id}")
            
            # FLIR files
            flir_files = structure[board_name][test_id].get('flir', [])
            if flir_files:
                print(f"    FLIR Files ({len(flir_files)}):")
                for f in flir_files:
                    print(f"      - {f}")
            else:
                print(f"    FLIR Files: ❌ NONE FOUND")
            
            # Thermistor Air files
            air_files = structure[board_name][test_id].get('therm_air', [])
            if air_files:
                print(f"    Thermistor Air Files ({len(air_files)}):")
                for f in air_files:
                    print(f"      - {f}")
            else:
                print(f"    Thermistor Air Files: ❌ NONE FOUND")
            
            # Thermistor Sand files
            sand_files = structure[board_name][test_id].get('therm_sand', [])
            if sand_files:
                print(f"    Thermistor Sand Files ({len(sand_files)}):")
                for f in sand_files:
                    print(f"      - {f}")
            else:
                print(f"    Thermistor Sand Files: (Optional - Air only test)")
            
            print()
        
        print()


def display_json_template(structure, inputs_dir):
    """Display a JSON template ready to fill in with component mappings."""
    import json
    
    template = {
        "boards": []
    }
    
    for board_name in sorted(structure.keys()):
        board = {
            "name": board_name,
            "pcb": f"{board_name}.brd",  # Placeholder - update as needed
            "tests": []
        }
        
        for test_id in sorted(structure[board_name].keys()):
            test_data = structure[board_name][test_id]
            
            # Get first file of each type (or empty string if none)
            flir_file = test_data.get('flir', [''])[0]
            air_files = test_data.get('therm_air', [])
            sand_files = test_data.get('therm_sand', [])
            
            test = {
                "test_id": test_id,
                "flir_file": f"inputs/{flir_file}" if flir_file else "REQUIRED - ADD FLIR FILE",
                "therm_air_file": f"inputs/{air_files[0]}" if air_files else "REQUIRED - ADD AIR FILE",
                "pairs": [
                    {
                        "flir_roi": "ROI_NAME_HERE",
                        "therm_channel": "AI0",
                        "component": "COMPONENT_DESIGNATOR",
                        "type": "fet"
                    }
                ]
            }
            
            # Add sand file only if it exists
            if sand_files:
                test["therm_sand_file"] = f"inputs/{sand_files[0]}"
            
            board["tests"].append(test)
        
        template["boards"].append(board)
    
    print("=" * 80)
    print("JSON CONFIGURATION TEMPLATE")
    print("=" * 80)
    print()
    print(json.dumps(template, indent=2))
    print()
    print("📝 Instructions:")
    print("  1. Copy the JSON above to your config file")
    print("  2. Update 'pcb' fields with actual .brd filenames")
    print("  3. Add component pairs for each test with correct:")
    print("     - flir_roi: ROI name from FLIR file")
    print("     - therm_channel: AI0, AI1, AI2, etc. (or Device1_AI0 for multi-device)")
    print("     - component: Component designator (U1, U3, etc.)")
    print("     - type: 'fet', 'diode', 'ldo', etc.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Find and organize thermal calibration input files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python config_file_finder.py                    # Scan default inputs/ directory
  python config_file_finder.py --format json      # Output JSON template
  python config_file_finder.py --inputs_dir data  # Custom inputs directory
        """
    )
    
    parser.add_argument(
        '--inputs_dir',
        default='inputs',
        help='Path to inputs directory (default: inputs)'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    args = parser.parse_args()
    
    # Scan directory structure
    structure = scan_inputs_directory(args.inputs_dir)
    
    # Display results
    if args.format == 'json':
        display_json_template(structure, args.inputs_dir)
    else:
        display_text_format(structure, args.inputs_dir)
        print()
        print("💡 Tip: Run with --format json to get a ready-to-use config template")
        print()


if __name__ == '__main__':
    main()
