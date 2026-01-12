"""
===============================================================================
PHASE 2 VISUALIZATION - Filtering Comparison Plots
===============================================================================
Visualization functions for Phase 2 (Filtering & Transient Analysis):
- Original vs filtered data comparison plots
- Multi-filter method comparison grids
- Filtering metrics tables

Used by: phase2_filtering.py
Data source: FLIR thermal camera measurements (ResearchIR exports)
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import Dict, List, Tuple
import os

# Import visualization helpers
import viz_helpers as viz
import phase1_data_loading as phase1


def create_filtering_comparison_plots(original_data: Dict, filtered_data: Dict, 
                                     grouped_components: Dict, output_dir: str,
                                     filter_type: str = 'median',
                                     debug: bool = False) -> List[str]:
    """
    Create side-by-side comparison plots showing original vs filtered data
    
    Args:
        original_data: Dict of original temperature DataFrames
        filtered_data: Dict of filtered temperature DataFrames
        grouped_components: Dict grouping components by type
        output_dir: Output directory for plot files
        filter_type: Type of filter applied
        debug: Enable debug output
        
    Returns:
        List of output file paths
    """
    
    if len(grouped_components) == 0:
        return []
    
    # Set up IEEE-style plotting
    viz.setup_ieee_plot_style()
    
    output_files = []
    
    # Select top 6 component groups with most components
    sorted_groups = sorted(grouped_components.items(), key=lambda x: len(x[1]), reverse=True)
    top_groups = sorted_groups[:6]
    
    if len(top_groups) == 0:
        return output_files
    
    # Create figure
    n_rows = len(top_groups)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7, 1.5 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Thermal Filtering Comparison: Original vs {filter_type.title()} Filtered', 
                fontsize=12, y=0.98)
    
    for row_idx, (group_key, components) in enumerate(top_groups):
        group_info = phase1.DEFAULT_COMPONENT_TYPES.get(group_key, 
                                                    {'name': group_key, 'color': '#888888', 'marker': 'o'})
        
        ax_orig = axes[row_idx, 0]
        ax_filt = axes[row_idx, 1]
        
        # Plot up to 8 components per group
        plot_components = components[:8]
        
        for i, component in enumerate(plot_components):
            if component in original_data and component in filtered_data:
                orig_df = original_data[component]
                filt_df = filtered_data[component]
                
                orig_temps = orig_df['Temperature'].values
                filt_temps = filt_df['Temperature'].values
                times = orig_df['Time'].values
                
                line_style = ['-', '--', '-.', ':'][i % 4]
                alpha = 0.7 if len(plot_components) > 4 else 1.0
                
                # Plot original
                ax_orig.plot(times, orig_temps, 
                           color=group_info['color'], 
                           linestyle=line_style,
                           alpha=alpha,
                           linewidth=viz.IEEE_CONFIG['line_width'],
                           label=component if len(plot_components) <= 4 else None)
                
                # Plot filtered
                ax_filt.plot(times, filt_temps, 
                           color=group_info['color'], 
                           linestyle=line_style,
                           alpha=alpha,
                           linewidth=viz.IEEE_CONFIG['line_width'],
                           label=component if len(plot_components) <= 4 else None)
        
        # Format subplots
        for ax, title_suffix in [(ax_orig, 'Original'), (ax_filt, f'{filter_type.title()} Filtered')]:
            ax.set_title(f"{group_info['name']} - {title_suffix}", fontsize=12)
            ax.set_ylabel('Temp. (°C)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(FuncFormatter(viz.seconds_to_minutes_formatter))
            
            if row_idx == len(top_groups) - 1:
                ax.set_xlabel('Time (min)', fontsize=12)
            
            if len(plot_components) <= 4 and any(comp in original_data for comp in plot_components):
                ax.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    
    # Save
    base_filename = os.path.join(output_dir, f'filtering_comparison_{filter_type}')
    png_file = viz.save_plot(base_filename, 'png')
    pdf_file = viz.save_plot(base_filename, 'pdf')
    output_files.extend([png_file, pdf_file])
    
    plt.close()
    
    if debug:
        print(f"Created filtering comparison plots: {len(output_files)} files")
    
    return output_files


def create_multifilter_component_grid(original_data: Dict, grouped_components: Dict, 
                                     output_dir: str, filter_params: Dict,
                                     debug: bool = False) -> Tuple[str, pd.DataFrame]:
    """
    Create comprehensive grid comparing all filtering methods
    
    Args:
        original_data: Dict of original temperature DataFrames
        grouped_components: Dict grouping components by type
        output_dir: Output directory for files
        filter_params: Dictionary with filter configuration
        debug: Enable debug output
        
    Returns:
        Tuple of (plot_filename, comparison_dataframe)
    """
    
    import phase2_filtering as phase2
    
    if len(grouped_components) == 0:
        return None, None
    
    # Define filter methods to compare
    filter_methods = [
        ('none', 'Original', {}),
        ('savgol', 'Savitzky-Golay', {'window_length': 11, 'polyorder': 3}),
        ('hybrid', 'Hybrid', {'z_threshold': 2.0, 'window_length': 11, 'polyorder': 3}),
        ('median', 'Median', {'kernel_size': 5}),
        ('lowpass', 'Butterworth', {'cutoff_freq': 0.01, 'filter_order': 4}),
        ('outlier', 'Outlier Removal', {'z_threshold': 2.0})
    ]
    
    # Select top 6 component types
    sorted_groups = sorted(grouped_components.items(), key=lambda x: len(x[1]), reverse=True)
    top_groups = sorted_groups[:6]
    
    if len(top_groups) == 0:
        return None, None
    
    # Select one representative component from each group
    representative_components = {}
    for group_key, components in top_groups:
        best_component = None
        max_points = 0
        for comp in components:
            if comp in original_data:
                n_points = len(original_data[comp])
                if n_points > max_points:
                    max_points = n_points
                    best_component = comp
        if best_component:
            representative_components[group_key] = best_component
    
    if len(representative_components) == 0:
        return None, None
    
    # Set up plotting
    viz.setup_ieee_plot_style()
    
    n_rows = len(representative_components)
    n_cols = len(filter_methods)
    
    # Fixed 7x9 inch size for paper printing with 12pt fonts
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 9))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Filtering Method Comparison Across Component Types', 
                fontsize=12, y=0.995)
    
    comparison_metrics = []
    
    # Process each component type (rows)
    for row_idx, (group_key, component) in enumerate(representative_components.items()):
        group_info = phase1.DEFAULT_COMPONENT_TYPES.get(group_key, 
                                                    {'name': group_key, 'color': '#888888'})
        
        if component not in original_data:
            continue
        
        orig_df = original_data[component]
        orig_temps = orig_df['Temperature'].values
        times = orig_df['Time'].values
        
        steady_state_window = max(1, len(orig_temps) // 10)
        orig_steady_state = np.mean(orig_temps[-steady_state_window:])
        
        # Process each filter method (columns)
        for col_idx, (filter_type, filter_name, params) in enumerate(filter_methods):
            ax = axes[row_idx, col_idx]
            
            # Apply filtering
            if filter_type == 'none':
                filtered_temps = orig_temps.copy()
            else:
                filtered_temps = phase2.apply_thermal_filtering(orig_temps, times, 
                                                                filter_type=filter_type,
                                                                filter_params=params)
            
            # Calculate metrics
            filtered_steady_state = np.mean(filtered_temps[-steady_state_window:])
            steady_state_error = abs(filtered_steady_state - orig_steady_state)
            steady_state_error_pct = (steady_state_error / orig_steady_state * 100) if orig_steady_state != 0 else 0
            noise_std = np.std(filtered_temps)
            
            comparison_metrics.append({
                'Component_Type': group_info['name'],
                'Component': component,
                'Filter_Method': filter_name,
                'Original_Steady_State_C': f'{orig_steady_state:.2f}',
                'Filtered_Steady_State_C': f'{filtered_steady_state:.2f}',
                'Steady_State_Error_C': f'{steady_state_error:.3f}',
                'Steady_State_Error_Pct': f'{steady_state_error_pct:.2f}',
                'Noise_StdDev_C': f'{noise_std:.3f}'
            })
            
            # Plot
            ax.plot(times, filtered_temps, color=group_info['color'], linewidth=1.5)
            ax.xaxis.set_major_formatter(FuncFormatter(viz.seconds_to_minutes_formatter))
            # Set x-axis ticks manually every 30 minutes
            max_time_min = times[-1] / 60
            ax.set_xticks(range(0, int(max_time_min) + 30, 30))
            ax.grid(True, alpha=0.3)
            
            # Increase font sizes to 12pt
            ax.tick_params(labelsize=10)  # Slightly smaller for tick labels to fit
            
            if row_idx == 0:
                ax.set_title(filter_name, fontsize=12)
            
            if col_idx == 0:
                ax.set_ylabel(f"{group_info['name']}\nTemp (°C)", fontsize=12)
            
            if row_idx == len(representative_components) - 1:
                ax.set_xlabel('Time (min)', fontsize=12)
            
            metrics_text = f"σ={noise_std:.2f}°C\nΔSS={steady_state_error_pct:.1f}%"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,  # Slightly smaller for metric annotations
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save grid plot
    base_filename = os.path.join(output_dir, 'filtering_method_comparison_grid')
    png_file = viz.save_plot(base_filename, 'png')
    pdf_file = viz.save_plot(base_filename, 'pdf')
    
    plt.close()
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_metrics)
    
    # Save comparison table
    csv_file = os.path.join(output_dir, 'filtering_method_comparison_table.csv')
    comparison_df.to_csv(csv_file, index=False)
    
    print(f"Created multi-filter comparison grid: {png_file}")
    print(f"Created comparison metrics table: {csv_file}")
    
    return png_file, comparison_df


def create_three_condition_comparison(flir_data: Dict, air_data: Dict, sand_data: Dict,
                                     component_names: List[str], output_dir: str,
                                     pcb_name: str = None, plot_settings: Dict = None) -> Tuple[List[str], List[Dict]]:
    """
    Create single 3x1 comparison plot showing FLIR, Air, and Sand data for all overlapping components.
    
    All components are plotted on the same figure with consistent colors across all three subplots.
    Applies median filtering to smooth data and make traces easier to distinguish.
    
    Args:
        flir_data: Dict of FLIR temperature DataFrames (filtered)
        air_data: Dict of air thermocouple temperature DataFrames
        sand_data: Dict of sand/embedded thermocouple temperature DataFrames
        component_names: List of component names to plot (must exist in all three datasets)
        output_dir: Output directory for plot files
        pcb_name: Name of PCB for filename labeling
        plot_settings: Optional dict with y_axis_min, y_axis_max, unified_y_axis, max_sand_time_hours,
                       max_air_time_hours, flir_subplot_max_hours, air_subplot_max_hours,
                       sand_subplot_max_hours, test2_truncate_hours, air_test3_truncate_hours
    Returns:
        Tuple of (output_file_paths, valid_entries) where valid_entries contains the component data info
    """
    from loader_thermistor import SteadyStateAnalyzer
    from scipy.signal import medfilt
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract plot settings with defaults
    if plot_settings is None:
        plot_settings = {}
    y_min = plot_settings.get('y_axis_min', 20)
    y_max = plot_settings.get('y_axis_max', None)  # None means auto with cap at 35
    unified_y_axis = plot_settings.get('unified_y_axis', True)
    max_sand_hours = plot_settings.get('max_sand_time_hours', 7)
    max_air_hours = plot_settings.get('max_air_time_hours', None)
    test2_truncate_hours = plot_settings.get('test2_truncate_hours', None)  # Hours to truncate Test2 sand data
    air_test3_truncate_hours = plot_settings.get('air_test3_truncate_hours', None)  # Hours to truncate Test3 air data
    flir_subplot_max_hours = plot_settings.get('flir_subplot_max_hours', None)
    air_subplot_max_hours = plot_settings.get('air_subplot_max_hours', None)
    sand_subplot_max_hours = plot_settings.get('sand_subplot_max_hours', None)
    
    viz.setup_ieee_plot_style()
    output_files = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper function to apply median filtering
    def apply_median_filter(temp, kernel_size=5):
        """
        Apply median filter to smooth temperature data.
        
        Args:
            temp: Temperature array
            kernel_size: Kernel size for median filter (must be odd)
            
        Returns:
            Filtered temperature array
        """
        if len(temp) >= kernel_size:
            return medfilt(temp, kernel_size=kernel_size)
        else:
            return temp
    
    # Filter to only components that exist in all three datasets
    # Extract unique component names from the keys (format: "Component_TestID")
    flir_comp_names = set(flir_data.keys())
    air_comp_keys = list(air_data.keys())
    sand_comp_keys = list(sand_data.keys())
    
    # Build mapping of component name to sand keys
    sand_by_component = {}
    for sand_key in sand_comp_keys:
        sand_df = sand_data[sand_key]
        if 'Component' in sand_df.columns:
            comp_name = sand_df['Component'].iloc[0]
        else:
            comp_name = sand_key.split('_')[0]
        if comp_name not in sand_by_component:
            sand_by_component[comp_name] = []
        sand_by_component[comp_name].append(sand_key)
    
    # Build list of valid entries that have all three measurements
    valid_entries = []
    for air_key in air_comp_keys:
        # Extract component name from DataFrame
        air_df = air_data[air_key]
        if 'Component' in air_df.columns:
            comp_name = air_df['Component'].iloc[0]
            air_test_id = air_df['Test'].iloc[0] if 'Test' in air_df.columns else 'unknown'
        else:
            comp_name = air_key.split('_')[0]
            air_test_id = '_'.join(air_key.split('_')[1:]) if '_' in air_key else 'unknown'
        
        # Check if this component exists in FLIR data and has sand data (from any test)
        if comp_name in flir_comp_names and comp_name in sand_by_component:
            # Use the first available sand measurement for this component
            sand_key = sand_by_component[comp_name][0]
            sand_df = sand_data[sand_key]
            sand_test_id = sand_df['Test'].iloc[0] if 'Test' in sand_df.columns else 'unknown'
            
            valid_entries.append({
                'air_key': air_key,
                'sand_key': sand_key,
                'component': comp_name,
                'air_test': air_test_id,
                'sand_test': sand_test_id
            })
    
    if not valid_entries:
        print(f"Warning: No components found in all three datasets")
        return output_files, []
    
    # Sort entries by component name (natural sort for U1, U2, U10, etc.)
    import re
    def natural_sort_key(entry):
        """Extract numbers from component name for natural sorting"""
        comp = entry['component']
        # Split component into letter prefix and number
        parts = re.split(r'(\d+)', comp)
        return [int(part) if part.isdigit() else part for part in parts]
    
    valid_entries.sort(key=natural_sort_key)
    
    print(f"  Creating combined plot for {len(valid_entries)} measurements")
    print(f"  Component order: {[e['component'] for e in valid_entries]}")
    
    # Generate ROYGBIV rainbow colors from top to bottom
    import matplotlib.cm as cm
    n_colors = len(valid_entries)
    # Use rainbow colormap for ROYGBIV effect
    colors = [cm.rainbow(i / n_colors) for i in range(n_colors)]
    
    # Create figure with 3x1 subplots (independent x-axis for each)
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 7), sharex=False)
    
    # === Plot 1: FLIR Camera Data ===
    ax = axes[0]
    flir_max_temp = 20.0  # Track maximum temperature
    flir_max_time = 0  # Track maximum time
    
    # Track maximum temperatures across all datasets for unified y-axis
    all_max_temps = []
    
    for i, entry in enumerate(valid_entries):
        comp_name = entry['component']
        air_test_id = entry['air_test']
        color = colors[i]
        
        flir_df = flir_data[comp_name]
        t_flir_full = flir_df['Time'].values
        temp_flir_full = flir_df['Temperature'].values

        # Apply FLIR time truncation if configured
        if flir_subplot_max_hours is not None:
            flir_limit = flir_subplot_max_hours * 3600
            mask = t_flir_full <= flir_limit
            t_flir = t_flir_full[mask]
            temp_flir = temp_flir_full[mask]
        else:
            t_flir = t_flir_full
            temp_flir = temp_flir_full
        
        # Track maximum temperature and time for axis scaling
        flir_max_temp = max(flir_max_temp, np.max(temp_flir))
        all_max_temps.append(np.max(temp_flir))
        flir_max_time = max(flir_max_time, np.max(t_flir))
        
        # Determine time scale for this subplot
        if flir_max_time > 7200:  # > 2 hours
            t_flir_display = t_flir / 3600
        else:
            t_flir_display = t_flir / 60
        
        # Calculate steady-state
        ss_flir = SteadyStateAnalyzer.steady_value(t_flir, temp_flir, smooth_win=5)
        
        # Create label with component and test
        label = f'{comp_name} (Air:{air_test_id})'
        
        # Plot time series
        ax.plot(t_flir_display, temp_flir, '-', linewidth=1.5, 
               color=color, label=label, alpha=0.8)
        
        # Plot steady-state line
        ax.axhline(ss_flir['value'], linestyle='--', linewidth=1.0,
                  color=color, alpha=0.4)
    
    # Set time label based on actual data duration
    if flir_max_time > 7200:
        flir_time_label = 'Time (hr)'
    else:
        flir_time_label = 'Time (min)'
    
    ax.set_xlabel(flir_time_label, fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('FLIR SROI Measurements', fontsize=12)
    ax.set_xlim(t_flir_display[0] if len(valid_entries) > 0 else 0, 
                max([flir_data[e['component']]['Time'].values[-1] / (3600 if flir_max_time > 7200 else 60) 
                     for e in valid_entries]) if len(valid_entries) > 0 else flir_max_time / (3600 if flir_max_time > 7200 else 60))
    ax.grid(True, alpha=0.3)
    # Set x-axis ticks manually every 10 minutes
    if flir_max_time <= 7200:  # Minutes scale
        max_val = int(flir_max_time / 60) + 10
        ax.set_xticks([i for i in range(0, max_val, 10)])
    else:  # Hours scale
        max_val_hours = flir_max_time / 3600
        if max_val_hours <= 6:
            # Every 30 minutes (0.5 hours) for tests <= 6 hours
            ax.set_xticks([i * 0.5 for i in range(0, int(max_val_hours * 2) + 1)])
        elif max_val_hours <= 12:
            # Every 1 hour for tests between 6-12 hours
            ax.set_xticks([i for i in range(0, int(max_val_hours) + 2)])
        elif max_val_hours <= 24:
            # Every 4 hours for tests between 12-24 hours
            ax.set_xticks([i * 4 for i in range(0, int(max_val_hours / 4) + 2)])
        else:
            # Every 5 hours for tests > 24 hours (ideal for ~40 hour tests)
            ax.set_xticks([i * 5 for i in range(0, int(max_val_hours / 5) + 2)])
    
    # === Plot 2: Air Thermocouple Data ===
    ax = axes[1]
    air_max_temp = 20.0  # Track maximum temperature
    air_max_time = 0  # Track maximum time
    
    for i, entry in enumerate(valid_entries):
        air_key = entry['air_key']
        comp_name = entry['component']
        air_test_id = entry['air_test']
        color = colors[i]
        
        air_df = air_data[air_key]
        t_air_full = air_df['Time'].values
        temp_air_full = air_df['Temperature'].values
        
        # Apply time limit if specified (use stricter of max_air_hours or air_subplot_max_hours)
        # For Test3_Air, also apply air_test3_truncate_hours
        air_limits = []
        if max_air_hours is not None:
            air_limits.append(max_air_hours * 3600)
        if air_subplot_max_hours is not None:
            air_limits.append(air_subplot_max_hours * 3600)
        if air_test3_truncate_hours is not None and 'Test3' in air_test_id:
            air_limits.append(air_test3_truncate_hours * 3600)
        if air_limits:
            air_cutoff = min(air_limits)
            mask = t_air_full <= air_cutoff
            t_air = t_air_full[mask]
            temp_air = temp_air_full[mask]
        else:
            t_air = t_air_full
            temp_air = temp_air_full
        
        # Track maximum temperature and time for axis scaling
        air_max_temp = max(air_max_temp, np.max(temp_air))
        all_max_temps.append(np.max(temp_air))
        air_max_time = max(air_max_time, np.max(t_air))
        
        # Determine time scale for this subplot
        if air_max_time > 7200:  # > 2 hours
            t_air_display = t_air / 3600
        else:
            t_air_display = t_air / 60
        
        # Calculate steady-state
        ss_air = SteadyStateAnalyzer.steady_value(t_air, temp_air, smooth_win=5)
        
        # Create label with component and test
        label = f'{comp_name} (Air:{air_test_id})'
        
        # Plot time series
        ax.plot(t_air_display, temp_air, '-', linewidth=1.5,
               color=color, label=label, alpha=0.8)
        
        # Plot steady-state line
        ax.axhline(ss_air['value'], linestyle='--', linewidth=1.0,
                  color=color, alpha=0.4)
    
    # Set time label based on actual data duration
    if air_max_time > 7200:
        air_time_label = 'Time (hr)'
    else:
        air_time_label = 'Time (min)'
    
    ax.set_xlabel(air_time_label, fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Thermocouple Measurements (Air)', fontsize=12)
    ax.set_xlim(0, air_max_time / (3600 if air_max_time > 7200 else 60))
    ax.grid(True, alpha=0.3)
    # Set x-axis ticks manually every 10 minutes
    if air_max_time <= 7200:  # Minutes scale
        max_val = int(air_max_time / 60) + 10
        ax.set_xticks([i for i in range(0, max_val, 10)])
    else:  # Hours scale
        max_val_hours = air_max_time / 3600
        if max_val_hours <= 6:
            # Every 30 minutes (0.5 hours) for tests <= 6 hours
            ax.set_xticks([i * 0.5 for i in range(0, int(max_val_hours * 2) + 1)])
        elif max_val_hours <= 12:
            # Every 1 hour for tests between 6-12 hours
            ax.set_xticks([i for i in range(0, int(max_val_hours) + 2)])
        elif max_val_hours <= 24:
            # Every 4 hours for tests between 12-24 hours
            ax.set_xticks([i * 4 for i in range(0, int(max_val_hours / 4) + 2)])
        else:
            # Every 5 hours for tests > 24 hours (ideal for ~40 hour tests)
            ax.set_xticks([i * 5 for i in range(0, int(max_val_hours / 5) + 2)])
    
    # === Plot 3: Sand/Embedded Thermocouple Data ===
    ax = axes[2]
    sand_max_temp = 20.0  # Track maximum temperature
    sand_max_time = 0  # Track maximum time
    max_sand_time = max_sand_hours * 3600  # Convert hours to seconds
    sand_limit_seconds = max_sand_time
    if sand_subplot_max_hours is not None:
        sand_limit_seconds = min(sand_limit_seconds, sand_subplot_max_hours * 3600)
    
    for i, entry in enumerate(valid_entries):
        sand_key = entry['sand_key']
        comp_name = entry['component']
        sand_test_id = entry['sand_test']
        color = colors[i]
        
        sand_df = sand_data[sand_key]
        t_sand_full = sand_df['Time'].values
        temp_sand_full = sand_df['Temperature'].values
        
        # Calculate steady-state using truncated data where applicable
        if test2_truncate_hours is not None and 'Test2' in sand_test_id:
            test2_truncate_time = test2_truncate_hours * 3600
            mask_ss = t_sand_full <= test2_truncate_time
        else:
            mask_ss = t_sand_full <= sand_limit_seconds
        ss_sand = SteadyStateAnalyzer.steady_value(t_sand_full[mask_ss], temp_sand_full[mask_ss], smooth_win=5)
        
        # Filter data for plotting
        if test2_truncate_hours is not None and 'Test2' in sand_test_id:
            test2_truncate_time = test2_truncate_hours * 3600
            mask = t_sand_full <= test2_truncate_time
        else:
            mask = t_sand_full <= sand_limit_seconds
        
        t_sand = t_sand_full[mask]
        temp_sand = temp_sand_full[mask]
        
        # Track maximum temperature and time for axis scaling
        sand_max_temp = max(sand_max_temp, np.max(temp_sand))
        all_max_temps.append(np.max(temp_sand))
        sand_max_time = max(sand_max_time, np.max(t_sand))
        
        # Determine time scale for this subplot
        if sand_max_time > 7200:  # > 2 hours
            t_sand_display = t_sand / 3600
        else:
            t_sand_display = t_sand / 60
        
        # Create label with component and test
        label = f'{comp_name} (Sand:{sand_test_id})'
        
        # Plot time series
        ax.plot(t_sand_display, temp_sand, '-', linewidth=1.5,
               color=color, label=label, alpha=0.8)
        
        # Plot steady-state line
        ax.axhline(ss_sand['value'], linestyle='--', linewidth=1.0,
                  color=color, alpha=0.4)
    
    # Set time label based on actual data duration
    if sand_max_time > 7200:
        sand_time_label = 'Time (hr)'
    else:
        sand_time_label = 'Time (min)'
    
    ax.set_xlabel(sand_time_label, fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Thermocouple Measurements (Embedded)', fontsize=12)
    ax.set_xlim(0, sand_max_time / (3600 if sand_max_time > 7200 else 60))
    ax.grid(True, alpha=0.3)
    # Set x-axis ticks manually every 30 minutes
    if sand_max_time <= 7200:  # Minutes scale
        max_val = int(sand_max_time / 60) + 30
        ax.set_xticks([i for i in range(0, max_val, 30)])
    else:  # Hours scale
        max_val_hours = sand_max_time / 3600
        if max_val_hours <= 6:
            # Every 30 minutes (0.5 hours) for tests <= 6 hours
            ax.set_xticks([i * 0.5 for i in range(0, int(max_val_hours * 2) + 1)])
        elif max_val_hours <= 12:
            # Every 1 hour for tests between 6-12 hours
            ax.set_xticks([i for i in range(0, int(max_val_hours) + 2)])
        elif max_val_hours <= 24:
            # Every 4 hours for tests between 12-24 hours
            ax.set_xticks([i * 4 for i in range(0, int(max_val_hours / 4) + 2)])
        else:
            # Every 5 hours for tests > 24 hours (ideal for ~40 hour tests)
            ax.set_xticks([i * 5 for i in range(0, int(max_val_hours / 5) + 2)])
    
    # Apply y-axis limits
    if unified_y_axis:
        # Calculate unified limit from all data
        unified_max_temp = max(all_max_temps) + 2 if all_max_temps else 35
        if y_max is not None:
            unified_max_temp = min(unified_max_temp, y_max)
        else:
            unified_max_temp = min(unified_max_temp, 35)  # Default cap
        for ax in axes:
            ax.set_ylim(y_min, unified_max_temp)
    else:
        # Independent y-axis for each subplot
        axes[0].set_ylim(y_min, min(flir_max_temp + 2, y_max) if y_max else flir_max_temp + 2)
        axes[1].set_ylim(y_min, min(air_max_temp + 2, y_max) if y_max else air_max_temp + 2)
        axes[2].set_ylim(y_min, min(sand_max_temp + 2, y_max) if y_max else sand_max_temp + 2)
    
    # Add unified legend to the right of the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=7, title='Components', title_fontsize=12)
    
    plt.tight_layout()
    # Adjust to make room for legend with better centering
    plt.subplots_adjust(right=0.78, left=0.12)
    
    # Save files
    if pcb_name:
        base_filename = os.path.join(output_dir, f'{pcb_name}_all_components_three_condition_comparison')
    else:
        base_filename = os.path.join(output_dir, 'all_components_three_condition_comparison')
    
    png_file = viz.save_plot(base_filename, 'png')
    pdf_file = viz.save_plot(base_filename, 'pdf')
    output_files.extend([png_file, pdf_file])
    
    print(f"Created combined 3-condition comparison: {png_file}")
    plt.close()
    
    return output_files, valid_entries


def create_component_detail_plots(flir_data: Dict, air_data: Dict, sand_data: Dict,
                                  valid_entries: List[Dict], output_dir: str,
                                  pcb_name: str = None, plot_settings: Dict = None) -> List[str]:
    """
    Create individual component detail plots showing FLIR, Air, and Sand data.
    
    For each component, creates a 3-subplot figure with:
    - FLIR camera data
    - Air thermocouple data
    - Sand thermocouple data
    
    Plots are organized by component type in subdirectories.
    
    Args:
        flir_data: Dict of FLIR temperature DataFrames
        air_data: Dict of air thermocouple temperature DataFrames
        sand_data: Dict of sand thermocouple temperature DataFrames
        valid_entries: List of valid component entries with air_key, sand_key, component, etc.
        output_dir: Output directory for plot files
        pcb_name: Name of PCB for filename labeling
        plot_settings: Optional dict with y_axis_min, y_axis_max, max_sand_time_hours, max_air_time_hours,
                   flir_subplot_max_hours, air_subplot_max_hours, sand_subplot_max_hours,
                   test2_truncate_hours, air_test3_truncate_hours
        
    Returns:
        List of output file paths
    """
    from loader_thermistor import SteadyStateAnalyzer
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract plot settings with defaults
    if plot_settings is None:
        plot_settings = {}
    y_min = plot_settings.get('y_axis_min', 20)
    y_max = plot_settings.get('y_axis_max', None)
    max_sand_hours = plot_settings.get('max_sand_time_hours', 7)
    max_air_hours = plot_settings.get('max_air_time_hours', None)
    test2_truncate_hours = plot_settings.get('test2_truncate_hours', None)  # Hours to truncate Test2 sand data
    air_test3_truncate_hours = plot_settings.get('air_test3_truncate_hours', None)  # Hours to truncate Test3 air data
    flir_subplot_max_hours = plot_settings.get('flir_subplot_max_hours', None)
    air_subplot_max_hours = plot_settings.get('air_subplot_max_hours', None)
    sand_subplot_max_hours = plot_settings.get('sand_subplot_max_hours', None)
    
    viz.setup_ieee_plot_style()
    output_files = []
    
    # Create main component_details directory with board name
    if pcb_name:
        details_dir = os.path.join(output_dir, 'component_details', pcb_name)
    else:
        details_dir = os.path.join(output_dir, 'component_details')
    os.makedirs(details_dir, exist_ok=True)
    
    # Map component prefix to full type name
    type_mapping = {
        'C': 'Capacitors',
        'DL': 'LEDs',
        'F': 'Fuses',
        'J': 'Connectors',
        'L': 'Inductors',
        'R': 'Resistors',
        'U': 'ICs'
    }
    
    print(f"\n  Creating individual component detail plots...")
    
    max_sand_time = max_sand_hours * 3600  # Convert hours to seconds
    sand_limit_seconds = max_sand_time
    if sand_subplot_max_hours is not None:
        sand_limit_seconds = min(sand_limit_seconds, sand_subplot_max_hours * 3600)
    
    for entry in valid_entries:
        comp_name = entry['component']
        air_key = entry['air_key']
        sand_key = entry['sand_key']
        air_test_id = entry['air_test']
        sand_test_id = entry['sand_test']
        
        # Determine component type folder
        comp_prefix = ''.join([c for c in comp_name if c.isalpha()])
        type_folder = type_mapping.get(comp_prefix, 'Other')
        type_dir = os.path.join(details_dir, type_folder)
        os.makedirs(type_dir, exist_ok=True)
        
        # Load data
        flir_df = flir_data[comp_name]
        air_df = air_data[air_key]
        sand_df = sand_data[sand_key]
        
        t_flir_full = flir_df['Time'].values
        temp_flir_full = flir_df['Temperature'].values

        # Apply FLIR truncation if configured
        if flir_subplot_max_hours is not None:
            flir_limit = flir_subplot_max_hours * 3600
            mask = t_flir_full <= flir_limit
            t_flir = t_flir_full[mask]
            temp_flir = temp_flir_full[mask]
        else:
            t_flir = t_flir_full
            temp_flir = temp_flir_full
        
        t_air_full = air_df['Time'].values
        temp_air_full = air_df['Temperature'].values
        
        # Apply air time limit if specified (use stricter of max_air_hours or air_subplot_max_hours)
        # For Test3_Air, also apply air_test3_truncate_hours
        air_limits = []
        if max_air_hours is not None:
            air_limits.append(max_air_hours * 3600)
        if air_subplot_max_hours is not None:
            air_limits.append(air_subplot_max_hours * 3600)
        if air_test3_truncate_hours is not None and 'Test3' in air_test_id:
            air_limits.append(air_test3_truncate_hours * 3600)
        if air_limits:
            air_cutoff = min(air_limits)
            mask = t_air_full <= air_cutoff
            t_air = t_air_full[mask]
            temp_air = temp_air_full[mask]
        else:
            t_air = t_air_full
            temp_air = temp_air_full
        
        t_sand_full = sand_df['Time'].values
        temp_sand_full = sand_df['Temperature'].values
        
        # Calculate steady-state values
        ss_flir = SteadyStateAnalyzer.steady_value(t_flir, temp_flir, smooth_win=5)
        ss_air = SteadyStateAnalyzer.steady_value(t_air, temp_air, smooth_win=5)
        
        # Handle sand truncation for steady-state calculation
        if test2_truncate_hours is not None and 'Test2' in sand_test_id:
            test2_truncate_time = test2_truncate_hours * 3600
            mask_ss = t_sand_full <= test2_truncate_time
        else:
            mask_ss = t_sand_full <= sand_limit_seconds
        ss_sand = SteadyStateAnalyzer.steady_value(t_sand_full[mask_ss], temp_sand_full[mask_ss], smooth_win=5)

        # Truncate sand data for plotting
        if test2_truncate_hours is not None and 'Test2' in sand_test_id:
            test2_truncate_time = test2_truncate_hours * 3600
            mask = t_sand_full <= test2_truncate_time
        else:
            mask = t_sand_full <= sand_limit_seconds
        t_sand = t_sand_full[mask]
        temp_sand = temp_sand_full[mask]
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=False)
        
        # Calculate unified y-axis limits
        all_temps = np.concatenate([temp_flir, temp_air, temp_sand])
        if y_max is not None:
            unified_max = min(np.max(all_temps) + 2, y_max)
        else:
            unified_max = min(np.max(all_temps) + 2, 35)
        unified_min = y_min
        
        # Plot 1: FLIR
        ax = axes[0]
        flir_max_time = np.max(t_flir)
        if flir_max_time > 7200:
            t_flir_display = t_flir / 3600
            time_label = 'Time (hr)'
        else:
            t_flir_display = t_flir / 60
            time_label = 'Time (min)'
        
        ax.plot(t_flir_display, temp_flir, '-', linewidth=1.5, color='#1f77b4', alpha=0.8)
        ax.axhline(ss_flir['value'], linestyle='--', linewidth=1.0, color='#1f77b4', alpha=0.4,
                  label=f"SS: {ss_flir['value']:.1f}°C")
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title(f'{comp_name} - FLIR Camera', fontsize=12)
        ax.set_ylim(unified_min, unified_max)
        ax.set_xlim(t_flir_display[0], t_flir_display[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_xlabel(time_label, fontsize=12)
        
        # Plot 2: Air
        ax = axes[1]
        air_max_time = np.max(t_air)
        if air_max_time > 7200:
            t_air_display = t_air / 3600
            time_label = 'Time (hr)'
        else:
            t_air_display = t_air / 60
            time_label = 'Time (min)'
        
        ax.plot(t_air_display, temp_air, '-', linewidth=1.5, color='#ff7f0e', alpha=0.8)
        ax.axhline(ss_air['value'], linestyle='--', linewidth=1.0, color='#ff7f0e', alpha=0.4,
                  label=f"SS: {ss_air['value']:.1f}°C")
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title(f'{comp_name} - Air Thermocouple (Test: {air_test_id})', fontsize=12)
        ax.set_ylim(unified_min, unified_max)
        ax.set_xlim(t_air_display[0], t_air_display[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_xlabel(time_label, fontsize=12)
        
        # Plot 3: Sand
        ax = axes[2]
        sand_max_time = np.max(t_sand)
        if sand_max_time > 7200:
            t_sand_display = t_sand / 3600
            time_label = 'Time (hr)'
        else:
            t_sand_display = t_sand / 60
            time_label = 'Time (min)'
        
        ax.plot(t_sand_display, temp_sand, '-', linewidth=1.5, color='#2ca02c', alpha=0.8)
        ax.axhline(ss_sand['value'], linestyle='--', linewidth=1.0, color='#2ca02c', alpha=0.4,
                  label=f"SS: {ss_sand['value']:.1f}°C")
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title(f'{comp_name} - Sand Thermocouple (Test: {sand_test_id})', fontsize=12)
        ax.set_ylim(unified_min, unified_max)
        ax.set_xlim(t_sand_display[0], t_sand_display[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_xlabel(time_label, fontsize=12)
        
        plt.tight_layout()
        
        # Save high DPI PNG only
        filename = os.path.join(type_dir, f'{comp_name}_three_condition_detail.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        output_files.append(filename)
        plt.close()
    
    print(f"  Created {len(output_files)} component detail plots in {details_dir}")
    
    return output_files
