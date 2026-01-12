"""
===============================================================================
PHASE 1: DATA LOADING
===============================================================================
Loads and parses ResearchIR thermal test data exports.

This module handles:
- Loading ResearchIR stats TXT files
- Parsing component temperature data
- Creating time-series DataFrames for each component
- Component classification by type (R, C, U, VR, etc.)
===============================================================================
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


# Default component type mappings with visual styling
DEFAULT_COMPONENT_TYPES = {
    'R': {'name': 'Resistors', 'color': '#1f77b4', 'marker': 'o'},
    'C': {'name': 'Capacitors', 'color': '#ff7f0e', 'marker': 's'},
    'L': {'name': 'Inductors', 'color': '#2ca02c', 'marker': '^'},
    'U': {'name': 'Integrated Circuits', 'color': '#d62728', 'marker': 'D'},
    'IC': {'name': 'Integrated Circuits', 'color': '#d62728', 'marker': 'D'},
    'J': {'name': 'Connectors', 'color': '#9467bd', 'marker': 'v'},
    'F': {'name': 'Fuses', 'color': '#3cb371', 'marker': 'D'},
    'VR': {'name': 'Voltage Regulators', 'color': '#8c564b', 'marker': 'p'},
    'DL': {'name': 'Diodes/LEDs', 'color': '#e377c2', 'marker': 'h'},
    'D': {'name': 'Diodes', 'color': '#e377c2', 'marker': 'h'},
    'Q': {'name': 'Transistors', 'color': '#7f7f7f', 'marker': '*'},
    'CR': {'name': 'Crystals', 'color': '#bcbd22', 'marker': '+'},
    'TP': {'name': 'Test Points', 'color': '#17becf', 'marker': 'x'},
    'PS': {'name': 'Power Supplies', 'color': '#ff9900', 'marker': 'P'},
    'SW': {'name': 'Switches', 'color': '#cc0000', 'marker': '8'},
    'MISC': {'name': 'Miscellaneous', 'color': '#888888', 'marker': '.'}
}


def load_researchir_files(input_folder: str, debug: bool = False) -> Dict:
    """
    Load all ResearchIR stats TXT files from export folder
    
    ResearchIR exports multi-frame thermal data as TXT files with statistics
    for each ROI (component). This function parses these files and creates
    time-series temperature data for each component.
    
    Args:
        input_folder: Path to folder containing ResearchIR exports
        debug: Enable verbose debug output
    
    Returns:
        Dictionary mapping component names to pandas DataFrames with columns:
        - Time: Time in seconds
        - Temperature: Temperature in Celsius
    
    Raises:
        FileNotFoundError: If input folder doesn't exist
        ValueError: If no valid stats files found or no data loaded
    """
    
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    # Find TXT stat files (these contain the component temperature data we need)
    txt_files = glob.glob(os.path.join(input_folder, "*Stats.txt"))
    
    if not txt_files:
        # Fallback: look for any TXT files
        txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
        
    if not txt_files:
        raise ValueError(f"No ResearchIR stats TXT files found in {input_folder}")
    
    print(f"Found {len(txt_files)} ResearchIR stats files")
    
    component_data = {}
    
    for file_path in txt_files:
        try:
            filename = os.path.basename(file_path)
            
            if debug:
                print(f"Processing {filename}")
            
            # Parse ResearchIR stats file
            component_temps = parse_researchir_stats_file(file_path, debug=debug)
            
            if component_temps:
                # Create time series data for each component
                for component_name, temp_value in component_temps.items():
                    if component_name not in component_data:
                        component_data[component_name] = []
                    
                    # Build time series (ResearchIR exports one stats file per frame)
                    # Each file represents a time point in the thermal test
                    time_point = len(component_data[component_name])
                    component_data[component_name].append({
                        'Time': time_point * 60,  # Time in seconds (assuming 1 min intervals)
                        'Temperature': temp_value
                    })
            
            if debug:
                print(f"Extracted {len(component_temps)} components from {filename}")
                    
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            continue
    
    # Convert lists to DataFrames
    final_component_data = {}
    for component_name, temp_list in component_data.items():
        if temp_list:
            df = pd.DataFrame(temp_list)
            df = df.sort_values('Time').reset_index(drop=True)
            final_component_data[component_name] = df
            
            if debug:
                temp_min = df['Temperature'].min()
                temp_max = df['Temperature'].max()
                print(f"Component {component_name}: {len(df)} data points, "
                      f"temp range {temp_min:.1f}-{temp_max:.1f}°C")
    
    if not final_component_data:
        raise ValueError("No valid component data could be loaded from ResearchIR files")
    
    print(f"Successfully loaded {len(final_component_data)} components from ResearchIR stats")
    return final_component_data


def parse_researchir_stats_file(file_path: str, debug: bool = False) -> Dict[str, float]:
    """
    Parse ResearchIR stats TXT file format
    
    ResearchIR stats files are space-delimited tables with:
    - Header row with component names
    - Multiple statistic rows (Mean, Min, Max, etc.)
    
    This function extracts the Mean temperature for each component.
    
    Args:
        file_path: Path to ResearchIR stats TXT file
        debug: Enable verbose debug output
    
    Returns:
        Dictionary mapping component names to mean temperatures (Celsius)
    """
    
    component_temps = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the header and mean temperature lines
    header_line = None
    mean_line = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Look for the header with component names (first line)
        if i == 0 and 'Statistic' in line_stripped:
            header_line = line_stripped
        # Look for mean temperature row
        elif 'Mean [C]' in line_stripped or 'Mean [°C]' in line_stripped:
            mean_line = line_stripped
            break
    
    if not header_line or not mean_line:
        if debug:
            print(f"Could not find header or mean temperature line in {file_path}")
        return component_temps
    
    # Parse ResearchIR space-delimited format
    # Use regex to split on multiple whitespace characters (2 or more spaces)
    header_parts = re.split(r'\s{2,}', header_line.strip())
    mean_parts = re.split(r'\s{2,}', mean_line.strip())
    
    if debug:
        print(f"Header parts: {len(header_parts)}")
        print(f"Mean parts: {len(mean_parts)}")
    
    # Skip the first column (statistic description like "Mean [C]")
    start_idx = 1
    
    for i in range(start_idx, min(len(header_parts), len(mean_parts))):
        component_name = header_parts[i].strip()
        temp_str = mean_parts[i].strip()
        
        # Skip non-component columns and empty names
        if component_name.lower() in ['image', 'statistic', 'n/a', ''] or not component_name:
            continue
        
        try:
            temp_value = float(temp_str)
            component_temps[component_name] = temp_value
            
            if debug:
                print(f"  {component_name}: {temp_value}°C")
                
        except (ValueError, IndexError):
            if debug:
                print(f"  Could not parse temperature for '{component_name}': '{temp_str}'")
            continue
    
    return component_temps


def classify_components(component_data: Dict, custom_groups: Optional[Dict] = None,
                       debug: bool = False) -> Dict:
    """
    Classify components by type based on designator prefixes
    
    Groups components using standard PCB designator prefixes:
    - R: Resistors
    - C: Capacitors
    - U/IC: Integrated Circuits
    - VR: Voltage Regulators
    - J: Connectors
    - etc.
    
    Args:
        component_data: Dictionary of component DataFrames from load_researchir_files()
        custom_groups: Optional dict to override/extend default component types
        debug: Enable verbose output
    
    Returns:
        Dictionary mapping component type keys to lists of component names
        Example: {'R': ['R1', 'R2'], 'C': ['C1', 'C3'], ...}
    """
    
    component_types = DEFAULT_COMPONENT_TYPES.copy()
    if custom_groups:
        component_types.update(custom_groups)
    
    grouped_components = {}
    unclassified = []
    
    for component_name in component_data.keys():
        classified = False
        
        # Try to match component prefix (longest first to match VR before V, etc.)
        for prefix in sorted(component_types.keys(), key=len, reverse=True):
            if component_name.upper().startswith(prefix.upper()):
                group_key = prefix
                if group_key not in grouped_components:
                    grouped_components[group_key] = []
                grouped_components[group_key].append(component_name)
                classified = True
                break
        
        if not classified:
            unclassified.append(component_name)
    
    # Handle unclassified components
    if unclassified:
        print(f"Unclassified components ({len(unclassified)}): "
              f"{unclassified[:10]}{'...' if len(unclassified) > 10 else ''}")
        
        # Analyze patterns and group or put in miscellaneous
        for component in unclassified:
            matched_pattern = False
            
            # Common pattern matching for misc components
            patterns = [
                (r'^(FB|FERR)', 'FB'),   # Ferrite beads
                (r'^(X|XTAL|Y)', 'CR'),  # Crystals
                (r'^(P|PAD)', 'TP'),     # Pads/test points
            ]
            
            for pattern, group in patterns:
                if re.match(pattern, component, re.IGNORECASE):
                    if group not in grouped_components:
                        grouped_components[group] = []
                    grouped_components[group].append(component)
                    matched_pattern = True
                    break
            
            # Put remaining in miscellaneous
            if not matched_pattern:
                if 'MISC' not in grouped_components:
                    grouped_components['MISC'] = []
                grouped_components['MISC'].append(component)
    
    # Print classification summary
    print("\nComponent Classification:")
    for group, components in grouped_components.items():
        group_name = component_types.get(group, {}).get('name', group)
        print(f"  {group_name}: {len(components)} components")
    
    return grouped_components
