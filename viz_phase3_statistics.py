"""
===============================================================================
PHASE 3 VISUALIZATION - Component Statistics & IEEE Plots
===============================================================================
Visualization functions for Phase 3 (Component Analysis):
- IEEE-format summary plots
- Component type statistical comparisons
- Temperature rise and variability charts

Used by: phase3_component_analysis.py
Data source: Filtered FLIR thermal camera measurements
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import os

# Import visualization helpers
import viz_helpers as viz
import phase1_data_loading as phase1


def create_ieee_plots(component_data: Dict, grouped_components: Dict, 
                     analysis_results: Dict, output_dir: str,
                     pcb_name: str = None,
                     debug: bool = False) -> List[str]:
    """
    Create IEEE-format plots with subplots by component type
    
    Args:
        component_data: Dict of temperature DataFrames
        grouped_components: Dict grouping components by type
        analysis_results: Dict of transient analysis results
        output_dir: Output directory for files
        pcb_name: Name of PCB/board for filename labeling (optional)
        debug: Enable debug output
        
    Returns:
        List of output file paths
    """
    
    viz.setup_ieee_plot_style()
    output_files = []
    
    # Create summary statistics plot
    create_summary_statistics_plot(grouped_components, analysis_results, output_dir, output_files, pcb_name)
    
    return output_files


def create_summary_statistics_plot(grouped_components: Dict, analysis_results: Dict, 
                                  output_dir: str, output_files: List[str],
                                  pcb_name: str = None):
    """Create summary statistics comparison plot
    
    Args:
        grouped_components: Dict grouping components by type
        analysis_results: Dict of transient analysis results
        output_dir: Output directory for files
        output_files: List to append output file paths to
        pcb_name: Name of PCB/board for filename labeling (optional)
    """
    
    # ============================================================================
    # OPTION A: Max Temperature Bar Plot (Sorted Hottest to Coolest)
    # Export detailed metrics to CSV for LaTeX table generation
    # ============================================================================
    
    # Prepare data for all metrics
    group_names = []
    group_keys_list = []
    mean_temps = []
    std_temps = []
    delta_temps = []
    max_temps = []
    temp_ranges = []
    max_rates = []
    peak_times = []
    component_counts = []
    fontsize = 12
    
    for group_key, components in grouped_components.items():
        group_info = phase1.DEFAULT_COMPONENT_TYPES.get(group_key, {'name': group_key})
        group_names.append(group_info['name'])
        group_keys_list.append(group_key)
        
        group_means = [analysis_results[comp]['mean_temp'] for comp in components if comp in analysis_results]
        group_stds = [analysis_results[comp]['std_temp'] for comp in components if comp in analysis_results]
        group_deltas = [analysis_results[comp]['delta_temp'] for comp in components if comp in analysis_results]
        group_maxes = [analysis_results[comp]['max_temp'] for comp in components if comp in analysis_results]
        group_ranges = [analysis_results[comp]['temp_range'] for comp in components if comp in analysis_results]
        group_rates = [analysis_results[comp]['max_rate'] for comp in components if comp in analysis_results]
        group_peak_times = [analysis_results[comp]['peak_time'] for comp in components if comp in analysis_results]
        
        mean_temps.append(np.mean(group_means) if group_means else 0)
        std_temps.append(np.mean(group_stds) if group_stds else 0)
        delta_temps.append(np.mean(group_deltas) if group_deltas else 0)
        max_temps.append(np.mean(group_maxes) if group_maxes else 0)
        temp_ranges.append(np.mean(group_ranges) if group_ranges else 0)
        max_rates.append(np.mean(group_rates) if group_rates else 0)
        peak_times.append(np.mean(group_peak_times) / 60 if group_peak_times else 0)  # Convert to minutes
        component_counts.append(len(components))
    
    # Sort all data by max temperature (descending - hottest to coolest)
    sorted_indices = np.argsort(max_temps)[::-1]
    group_names_sorted = [group_names[i] for i in sorted_indices]
    group_keys_sorted = [group_keys_list[i] for i in sorted_indices]
    mean_temps_sorted = [mean_temps[i] for i in sorted_indices]
    std_temps_sorted = [std_temps[i] for i in sorted_indices]
    delta_temps_sorted = [delta_temps[i] for i in sorted_indices]
    max_temps_sorted = [max_temps[i] for i in sorted_indices]
    temp_ranges_sorted = [temp_ranges[i] for i in sorted_indices]
    max_rates_sorted = [max_rates[i] for i in sorted_indices]
    peak_times_sorted = [peak_times[i] for i in sorted_indices]
    component_counts_sorted = [component_counts[i] for i in sorted_indices]
    
    colors_sorted = [phase1.DEFAULT_COMPONENT_TYPES.get(group_keys_sorted[i], {'color': '#888888'})['color'] 
                    for i in range(len(group_keys_sorted))]
    
    # Create single bar plot of maximum temperatures (sorted)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    ax.bar(group_names_sorted, max_temps_sorted, color=colors_sorted)
    ax.set_title('Maximum Temperature by Component Type', fontsize=12)
    ax.set_ylabel('Max Temperature (°C)', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    plt.tight_layout()
    
    # Save plot
    if pcb_name:
        summary_file = os.path.join(output_dir, f'{pcb_name}_max_temperature_by_type')
    else:
        summary_file = os.path.join(output_dir, 'max_temperature_by_type')
    
    png_file = viz.save_plot(summary_file, 'png')
    pdf_file = viz.save_plot(summary_file, 'pdf')
    output_files.extend([png_file, pdf_file])
    
    plt.close()
    
    # Export comprehensive metrics to CSV for LaTeX table generation
    metrics_df = pd.DataFrame({
        'Component_Type': group_names_sorted,
        'Component_Count': component_counts_sorted,
        'Mean_Temp_C': [f'{t:.1f}' for t in mean_temps_sorted],
        'Std_Dev_C': [f'{t:.2f}' for t in std_temps_sorted],
        'Temp_Rise_C': [f'{t:.1f}' for t in delta_temps_sorted],
        'Max_Temp_C': [f'{t:.1f}' for t in max_temps_sorted],
        'Temp_Range_C': [f'{t:.1f}' for t in temp_ranges_sorted],
        'Max_Rate_C_per_s': [f'{t:.3f}' for t in max_rates_sorted],
        'Peak_Time_min': [f'{t:.1f}' for t in peak_times_sorted]
    })
    
    if pcb_name:
        csv_file = os.path.join(output_dir, f'{pcb_name}_component_thermal_metrics.csv')
    else:
        csv_file = os.path.join(output_dir, 'component_thermal_metrics.csv')
    
    metrics_df.to_csv(csv_file, index=False)
    print(f"  Exported thermal metrics table: {csv_file}")
    
    
    # ============================================================================
    # ORIGINAL 2x2 BAR CHART LAYOUT (COMMENTED OUT FOR POTENTIAL FUTURE USE)
    # ============================================================================
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 6))
    # 
    # # Prepare data
    # group_names = []
    # mean_temps = []
    # std_temps = []
    # delta_temps = []
    # delta_temp_stds = []
    # 
    # for group_key, components in grouped_components.items():
    #     group_info = phase1.DEFAULT_COMPONENT_TYPES.get(group_key, {'name': group_key})
    #     group_names.append(group_info['name'])
    #     
    #     group_means = [analysis_results[comp]['mean_temp'] for comp in components if comp in analysis_results]
    #     group_stds = [analysis_results[comp]['std_temp'] for comp in components if comp in analysis_results]
    #     group_deltas = [analysis_results[comp]['delta_temp'] for comp in components if comp in analysis_results]
    #     
    #     mean_temps.append(np.mean(group_means) if group_means else 0)
    #     std_temps.append(np.mean(group_stds) if group_stds else 0)
    #     delta_temps.append(np.mean(group_deltas) if group_deltas else 0)
    #     delta_temp_stds.append(np.std(group_deltas) if len(group_deltas) > 1 else 0)
    # 
    # colors = [phase1.DEFAULT_COMPONENT_TYPES.get(k, {'color': '#888888'})['color'] 
    #           for k in grouped_components.keys()]
    # 
    # # Plot 1: Mean temperatures
    # ax1.bar(group_names, mean_temps, color=colors)
    # ax1.set_title('Mean Temperature by Component Type', fontsize=8)
    # ax1.set_ylabel('Temperature (°C)', fontsize=8)
    # ax1.tick_params(axis='x', rotation=45, labelsize=7)
    # plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # 
    # # Plot 2: Temperature variability
    # ax2.bar(group_names, std_temps, color=colors)
    # ax2.set_title('Temperature Variability (Std Dev)', fontsize=8)
    # ax2.set_ylabel('Std Dev (°C)', fontsize=8)
    # ax2.tick_params(axis='x', rotation=45, labelsize=7)
    # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # 
    # # Plot 3: Temperature rise
    # ax3.bar(group_names, delta_temps, color=colors)
    # ax3.set_title('Temperature Rise (Final - Initial)', fontsize=8)
    # ax3.set_ylabel('ΔT (°C)', fontsize=8)
    # ax3.tick_params(axis='x', rotation=45, labelsize=7)
    # plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # 
    # # Plot 4: Temperature rise variability
    # ax4.bar(group_names, delta_temp_stds, color=colors)
    # ax4.set_title('Temperature Rise Variability (Std Dev)', fontsize=8)
    # ax4.set_ylabel('Std Dev of ΔT (°C)', fontsize=8)
    # ax4.tick_params(axis='x', rotation=45, labelsize=7)
    # plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # 
    # plt.tight_layout()
    # 
    # # Save with PCB name in filename if provided
    # if pcb_name:
    #     summary_file = os.path.join(output_dir, f'{pcb_name}_thermal_analysis_summary')
    # else:
    #     summary_file = os.path.join(output_dir, 'thermal_analysis_summary')
    # 
    # png_file = viz.save_plot(summary_file, 'png')
    # pdf_file = viz.save_plot(summary_file, 'pdf')
    # output_files.extend([png_file, pdf_file])
    # 
    # plt.close()
