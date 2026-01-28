"""
Generate Thermistor Mapping Template
=====================================
Creates a template JSON mapping file for a PCB with both FLIR and thermistor data.

This tool:
1. Parses ResearchIR Stats files to extract all FLIR component names
2. Loads thermistor CSV to extract all available channels
3. Generates a template JSON with placeholders for user to fill in
4. Allows user to map which thermistor channels correspond to which FLIR components

Usage:
    python generate_thermistor_mapping.py <ResearchIR_dir> <thermistor_air_csv>

Author: Thermal Analysis Pipeline
Date: November 25, 2025
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
from loader_researchir import ResearchIRStatsParser


def load_thermistor_channels(thermistor_file: Path) -> List[str]:
    """Extract thermistor channel names from USB-TEMP CSV file."""
    print(f"\nLoading thermistor data: {thermistor_file}")
    
    # Read header to find channel names
    with open(thermistor_file, 'r') as f:
        lines = f.readlines()
    
    # Find the line with channel names (contains "Date/Time" and "AI")
    for line in lines:
        if 'Date/Time' in line and 'AI' in line:
            # Split and extract AI channels
            parts = line.strip().split(',')
            channels = [p.split('(')[0].strip() for p in parts if 'AI' in p]
            return channels
    
    raise ValueError(f"Could not find thermistor channel names in {thermistor_file}")


def generate_mapping_template(
    researchir_dir: Path,
    thermistor_file: Path,
    output_file: Path,
    pcb_name: str
) -> Dict:
    """Generate thermistor mapping template JSON."""
    
    print("="*80)
    print("THERMISTOR MAPPING TEMPLATE GENERATOR")
    print("="*80)
    
    # Parse ResearchIR data to get FLIR components
    print(f"\nParsing ResearchIR data: {researchir_dir}")
    parser = ResearchIRStatsParser(researchir_dir)
    df = parser.parse_all_frames()
    flir_components = parser.get_component_list()
    
    print(f"  Found {len(flir_components)} FLIR components")
    
    # Load thermistor channels
    thermistor_channels = load_thermistor_channels(thermistor_file)
    print(f"  Found {len(thermistor_channels)} thermistor channels: {', '.join(thermistor_channels)}")
    
    # Build template structure
    mapping = {
        "pcb_name": pcb_name,
        "description": f"Mapping between FLIR component designators and thermistor channels for {pcb_name}",
        "thermistor_to_flir_mapping": {},
        "available_flir_components": flir_components,
        "component_type_examples": [
            "PowerSupply",
            "IC",
            "Resistor",
            "LED",
            "Inductor",
            "Capacitor",
            "Diode",
            "Connector",
            "Switch",
            "VoltageRegulator",
            "TestPoint"
        ]
    }
    
    # Create placeholder for each thermistor channel
    for channel in thermistor_channels:
        mapping["thermistor_to_flir_mapping"][channel] = {
            "flir_component": "REPLACE_WITH_COMPONENT",
            "component_type": "REPLACE_WITH_TYPE",
            "notes": ""
        }
    
    # Save template
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TEMPLATE GENERATED")
    print(f"{'='*80}")
    print(f"\nTemplate saved to: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Open {output_file.name} in a text editor")
    print(f"  2. For each thermistor channel ({', '.join(thermistor_channels)}):")
    print(f"     - Replace 'REPLACE_WITH_COMPONENT' with the FLIR component name")
    print(f"     - Replace 'REPLACE_WITH_TYPE' with the component type")
    print(f"     - Add notes if needed")
    print(f"  3. Available FLIR components are listed in 'available_flir_components'")
    print(f"  4. Save the file and use it with phase6_thermal_calibration.py")
    
    return mapping


def main():
    """Main execution."""
    
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python generate_thermistor_mapping.py <ResearchIR_dir> <thermistor_air_csv> [pcb_name]")
        print("\nExample:")
        print("  python generate_thermistor_mapping.py")
        print("    inputs/ResearchIR_Outputs_Load_Shedding")
        print("    inputs/Test_3_AIR_usb_temp_DAQami.csv")
        print("    LoadShedding_AC_Switch")
        sys.exit(1)
    
    researchir_dir = Path(sys.argv[1])
    thermistor_file = Path(sys.argv[2])
    pcb_name = sys.argv[3] if len(sys.argv) > 3 else researchir_dir.name.replace('ResearchIR_Outputs_', '')
    
    # Validate inputs
    if not researchir_dir.exists():
        print(f"ERROR: ResearchIR directory not found: {researchir_dir}")
        sys.exit(1)
    
    if not thermistor_file.exists():
        print(f"ERROR: Thermistor file not found: {thermistor_file}")
        sys.exit(1)
    
    # Generate output filename
    output_file = Path(__file__).parent / f"{pcb_name.lower()}_thermistor_mapping.json"
    
    # Generate template
    generate_mapping_template(researchir_dir, thermistor_file, output_file, pcb_name)


if __name__ == "__main__":
    main()
