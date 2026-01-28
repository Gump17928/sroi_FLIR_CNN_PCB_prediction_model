"""
===============================================================================
PHASE 3: COMPONENT ANALYSIS
===============================================================================
Analyzes thermal behavior by component groups and generates statistical summaries.

This module provides:
- Component grouping by type
- Statistical analysis (mean, max, std dev)
- CSV export of analysis results
- MATLAB data export for further processing
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime

# Import scipy for MATLAB export (optional)
try:
    from scipy.io import savemat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def export_summary_csv(grouped_components: Dict, analysis_results: Dict,
                       output_dir: str, component_types: Dict = None) -> str:
    """
    Export comprehensive thermal analysis summary to CSV
    
    Creates a table with all components and their thermal characteristics:
    - Component type and name
    - Temperature statistics (initial, final, delta, max, min, mean, std)
    - Peak timing and rate of change
    
    Args:
        grouped_components: Dictionary from Phase 1 classification
        analysis_results: Dictionary from Phase 2 transient analysis
        output_dir: Output directory for CSV file
        component_types: Optional component type info for display names
    
    Returns:
        Path to saved CSV file
    """
    
    summary_data = []
    
    for group, components in grouped_components.items():
        group_name = group
        if component_types and group in component_types:
            group_name = component_types[group].get('name', group)
        
        for component in components:
            if component in analysis_results:
                stats = analysis_results[component]
                
                summary_data.append({
                    'Component_Type': group_name,
                    'Component': component,
                    'Initial_Temp_C': f'{stats["initial_temp"]:.1f}',
                    'Final_Temp_C': f'{stats["final_temp"]:.1f}',
                    'Delta_Temp_C': f'{stats["delta_temp"]:.1f}',
                    'Max_Temp_C': f'{stats["max_temp"]:.1f}',
                    'Min_Temp_C': f'{stats["min_temp"]:.1f}',
                    'Mean_Temp_C': f'{stats["mean_temp"]:.1f}',
                    'Std_Dev_C': f'{stats["std_temp"]:.1f}',
                    'Temp_Range_C': f'{stats["temp_range"]:.1f}',
                    'Peak_Time_s': f'{stats["peak_time"]:.0f}',
                    'Max_Rate_C_per_s': f'{stats["max_rate"]:.3f}',
                    'Data_Points': stats["data_points"]
                })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_filename = os.path.join(output_dir, 'thermal_analysis_summary.csv')
    df_summary.to_csv(csv_filename, index=False)
    
    print(f"Summary CSV saved to: {csv_filename}")
    print(f"  Total components: {len(summary_data)}")
    
    return csv_filename


def export_matlab_data(component_data: Dict, grouped_components: Dict,
                       analysis_results: Dict, output_dir: str) -> str:
    """
    Export thermal data to MATLAB .mat file for further analysis
    
    Creates structured MATLAB file with:
    - Time series data organized by component type
    - Statistical analysis results
    - Component groupings
    
    Args:
        component_data: Dictionary of filtered component DataFrames
        grouped_components: Component type groupings
        analysis_results: Analysis statistics
        output_dir: Output directory for .mat file
    
    Returns:
        Path to saved .mat file (or None if scipy unavailable)
    """
    
    if not SCIPY_AVAILABLE:
        print("Warning: scipy not available. MATLAB export skipped.")
        return None
    
    # Organize data by component groups
    matlab_data = {}
    
    for group, components in grouped_components.items():
        group_data = {}
        
        for component in components:
            if component in component_data:
                df = component_data[component]
                
                # Store time and temperature arrays
                comp_data = {
                    'Time': df['Time'].values,
                    'Temperature': df['Temperature'].values
                }
                
                # Add analysis results if available
                if component in analysis_results:
                    stats = analysis_results[component]
                    comp_data['delta_temp'] = stats['delta_temp']
                    comp_data['max_temp'] = stats['max_temp']
                    comp_data['mean_temp'] = stats['mean_temp']
                
                group_data[component] = comp_data
        
        if group_data:
            matlab_data[group] = group_data
    
    # Save to MATLAB file
    mat_filename = os.path.join(output_dir, 'thermal_data.mat')
    
    try:
        savemat(mat_filename, matlab_data)
        print(f"MATLAB data saved to: {mat_filename}")
        return mat_filename
    except Exception as e:
        print(f"Error saving MATLAB file: {e}")
        return None


def calculate_group_statistics(grouped_components: Dict, analysis_results: Dict) -> Dict:
    """
    Calculate aggregate statistics for each component type group
    
    Computes mean, std dev, min, max across all components in each group.
    Useful for understanding thermal behavior patterns by component type.
    
    Args:
        grouped_components: Component type groupings
        analysis_results: Individual component statistics
    
    Returns:
        Dictionary mapping group names to aggregate statistics
    """
    
    group_stats = {}
    
    for group, components in grouped_components.items():
        # Collect statistics for all components in group
        delta_temps = []
        max_temps = []
        mean_temps = []
        
        for component in components:
            if component in analysis_results:
                stats = analysis_results[component]
                delta_temps.append(stats['delta_temp'])
                max_temps.append(stats['max_temp'])
                mean_temps.append(stats['mean_temp'])
        
        if delta_temps:
            group_stats[group] = {
                'component_count': len(delta_temps),
                'delta_temp_mean': np.mean(delta_temps),
                'delta_temp_std': np.std(delta_temps),
                'delta_temp_max': np.max(delta_temps),
                'max_temp_mean': np.mean(max_temps),
                'max_temp_max': np.max(max_temps),
                'mean_temp_avg': np.mean(mean_temps)
            }
    
    return group_stats


def identify_thermal_hotspots(analysis_results: Dict, temp_threshold: float = 70.0,
                              delta_threshold: float = 10.0) -> Dict:
    """
    Identify components with high temperatures or large temperature rises
    
    Flags components that may require thermal management attention.
    
    Args:
        analysis_results: Component analysis statistics
        temp_threshold: Absolute temperature threshold (Celsius)
        delta_threshold: Temperature rise threshold (Celsius)
    
    Returns:
        Dictionary with 'high_temp' and 'high_delta' component lists
    """
    
    hotspots = {
        'high_temp': [],
        'high_delta': [],
        'combined': []
    }
    
    for component, stats in analysis_results.items():
        if stats['max_temp'] > temp_threshold:
            hotspots['high_temp'].append({
                'component': component,
                'max_temp': stats['max_temp'],
                'mean_temp': stats['mean_temp']
            })
        
        if stats['delta_temp'] > delta_threshold:
            hotspots['high_delta'].append({
                'component': component,
                'delta_temp': stats['delta_temp'],
                'final_temp': stats['final_temp']
            })
        
        if stats['max_temp'] > temp_threshold and stats['delta_temp'] > delta_threshold:
            hotspots['combined'].append({
                'component': component,
                'max_temp': stats['max_temp'],
                'delta_temp': stats['delta_temp']
            })
    
    # Sort by severity
    hotspots['high_temp'].sort(key=lambda x: x['max_temp'], reverse=True)
    hotspots['high_delta'].sort(key=lambda x: x['delta_temp'], reverse=True)
    hotspots['combined'].sort(key=lambda x: x['max_temp'] + x['delta_temp'], reverse=True)
    
    return hotspots


def create_matlab_usage_script(output_dir: str, grouped_components: Dict, component_types: Dict = None):
    """
    Create MATLAB script showing how to use the exported data
    
    Args:
        output_dir: Output directory for script
        grouped_components: Dictionary grouping components by type
        component_types: Optional component type info
    """
    
    script_content = f"""% MATLAB Script for Thermal Analysis Data
% Generated on {datetime.now().isoformat()}
% Load the exported thermal data

clear; clc;

% Load data
data = load('thermal_analysis_data.mat');

% Available component groups:
"""
    
    for group_key, components in grouped_components.items():
        if component_types:
            group_info = component_types.get(group_key, {'name': group_key})
        else:
            group_info = {'name': group_key}
        
        group_name = group_info['name'].replace(' ', '_').replace('/', '_')
        
        script_content += f"""
%% {group_info['name']} ({len(components)} components)
{group_name}_names = data.{group_name}_names;
{group_name}_time = data.{group_name}_time;
{group_name}_temp = data.{group_name}_temperature;
{group_name}_stats = data.{group_name}_statistics;

% Plot example for {group_info['name']}
figure;
hold on;
for i = 1:length({group_name}_names)
    plot({group_name}_time{{i}}, {group_name}_temp{{i}}, 'DisplayName', {group_name}_names{{i}});
end
title('{group_info['name']} Temperature Transients');
xlabel('Time (minutes)');
ylabel('Temperature (°C)');
legend('show', 'Location', 'best');
grid on;

% Convert time axis to minutes
ax = gca;
ax.XTickLabel = arrayfun(@(x) sprintf('%.0f', x/60), ax.XTick, 'UniformOutput', false);
"""
    
    script_content += """
% Statistics columns:
% [initial_temp, final_temp, delta_temp, max_temp, min_temp, 
%  mean_temp, std_temp, peak_time, max_rate, temp_range]

% Example: Calculate overall statistics
fprintf('\\n=== Overall Thermal Analysis Summary ===\\n');
"""
    
    for group_key, components in grouped_components.items():
        if component_types:
            group_info = component_types.get(group_key, {'name': group_key})
        else:
            group_info = {'name': group_key}
        
        group_name = group_info['name'].replace(' ', '_').replace('/', '_')
        
        script_content += f"""
if exist('{group_name}_stats', 'var') && ~isempty({group_name}_stats)
    fprintf('{group_info['name']}:\\n');
    fprintf('  Mean Temperature: %.1f ± %.1f °C\\n', ...
            mean({group_name}_stats(:, 6)), std({group_name}_stats(:, 6)));
    fprintf('  Temperature Rise: %.1f ± %.1f °C\\n', ...
            mean({group_name}_stats(:, 3)), std({group_name}_stats(:, 3)));
end
"""
    
    script_path = os.path.join(output_dir, 'analyze_thermal_data.m')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"MATLAB usage script created: {script_path}")


def export_component_statistics(component_data: Dict, output_dir: str,
                                board_name: str = "HBridge") -> str:
    """
    Export component-level statistical summary to CSV.
    
    Provides thermal statistics for each component for standalone analysis.
    
    Args:
        component_data: Dictionary of component DataFrames
        output_dir: Directory to save CSV
        board_name: Board identifier for filename
    
    Returns:
        Path to exported CSV file
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{board_name}_phase3_component_statistics.csv")
    
    rows = []
    
    for component_name, df in component_data.items():
        temps = df['Temperature'].values
        
        # Calculate statistics
        mean_temp = np.mean(temps)
        max_temp = np.max(temps)
        min_temp = np.min(temps)
        std_temp = np.std(temps)
        n_samples = len(temps)
        temp_rise = max_temp - min_temp
        
        rows.append({
            'component': component_name,
            'mean_temp': mean_temp,
            'max_temp': max_temp,
            'min_temp': min_temp,
            'std_temp': std_temp,
            'temp_rise': temp_rise,
            'n_samples': n_samples
        })
    
    # Export to CSV
    df = pd.DataFrame(rows)
    df = df.sort_values('mean_temp', ascending=False)  # Sort by hottest first
    df.to_csv(csv_path, index=False, float_format='%.2f')
    
    print(f"  ✓ Exported component statistics: {os.path.basename(csv_path)}")
    print(f"    Components: {len(rows)}")
    
    return csv_path


def export_component_metadata(roi_pixel_map: Dict, output_dir: str,
                              board_name: str = "HBridge") -> str:
    """
    Export component metadata (ROI info) to CSV.
    
    Provides spatial information for each component.
    
    Args:
        roi_pixel_map: Dictionary mapping component names to pixel coordinates
                      Format: {'U29': [(y1,x1), (y2,x2), ...], ...}
        output_dir: Directory to save CSV
        board_name: Board identifier for filename
    
    Returns:
        Path to exported CSV file
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{board_name}_phase3_component_metadata.csv")
    
    rows = []
    
    for component_name, pixels in roi_pixel_map.items():
        if not pixels:
            continue
        
        # Calculate ROI properties
        roi_pixels = len(pixels)
        
        # Calculate centroid
        y_coords = [p[0] for p in pixels]
        x_coords = [p[1] for p in pixels]
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        
        # Estimate area (assuming ~0.03125 mm² per pixel for typical FLIR resolution)
        # This is approximate - actual pixel size depends on camera and distance
        roi_area_mm2 = roi_pixels * 0.03125
        
        rows.append({
            'component': component_name,
            'roi_pixels': roi_pixels,
            'roi_area_mm2': roi_area_mm2,
            'centroid_y': centroid_y,
            'centroid_x': centroid_x
        })
    
    # Export to CSV
    df = pd.DataFrame(rows)
    df = df.sort_values('roi_pixels', ascending=False)  # Sort by largest ROI first
    df.to_csv(csv_path, index=False, float_format='%.2f')
    
    print(f"  ✓ Exported component metadata: {os.path.basename(csv_path)}")
    print(f"    Components: {len(rows)}")
    
    return csv_path


def export_roi_pixel_map_for_cnn(roi_pixel_map: Dict, output_dir: str,
                                  board_name: str = "HBridge") -> str:
    """
    Export ROI pixel map with full pixel lists for CNN training.
    
    This is specifically for Phase 8 ML training - saves the actual pixel 
    coordinates so CNN can build datasets without rerunning Phase 3/4.
    
    Args:
        roi_pixel_map: Dictionary mapping component names to pixel coordinates
                      Format: {'U29': [(y1,x1), (y2,x2), ...], ...}
        output_dir: Directory to save CSV
        board_name: Board identifier for filename
    
    Returns:
        Path to exported CSV file with pixel_list column
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{board_name}_roi_pixel_map.csv")
    
    rows = []
    
    for component_name, pixels in roi_pixel_map.items():
        if not pixels:
            # Still include components with no ROI (will be empty list)
            rows.append({
                'component': component_name,
                'pixel_count': 0,
                'pixel_list': '[]'
            })
            continue
        
        # Convert pixel list to string representation
        pixel_list_str = str(pixels)
        
        rows.append({
            'component': component_name,
            'pixel_count': len(pixels),
            'pixel_list': pixel_list_str
        })
    
    # Export to CSV
    df = pd.DataFrame(rows)
    df = df.sort_values('pixel_count', ascending=False)  # Sort by largest ROI first
    df.to_csv(csv_path, index=False)
    
    print(f"  ✓ Exported ROI pixel map for CNN: {os.path.basename(csv_path)}")
    print(f"    Components: {len(rows)}")
    print(f"    Total pixels: {df['pixel_count'].sum()}")
    
    return csv_path
