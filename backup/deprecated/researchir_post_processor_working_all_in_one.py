#!/usr/bin/env python3
"""
===============================================================================
RESEARCHIR THERMAL DATA POST-PROCESSOR WITH SPATIAL COUPLING ANALYSIS
===============================================================================

PURPOSE:
    Post-processes ResearchIR CSV/TXT exports to analyze temperature transients
    by component type with advanced spatial thermal coupling analysis for potted/
    embedded system failure prediction.

FUNCTIONALITY:
    - Reads ResearchIR CSV/TXT exports from designated folder
    - Groups components by designator prefix (R, C, U, J, VR, etc.)
    - Analyzes temperature transients from first to last frame
    - Generates IEEE-format publication-quality plots with subplots per component type
    - Exports MATLAB .mat files with organized cell arrays
    - Statistical analysis of temperature uniformity by component type
    - **NEW: Spatial thermal coupling analysis**
      * Loads component X,Y coordinates from PCB layout
      * Calculates Euclidean distance-based proximity matrix
      * Identifies thermal neighbors and coupling strength
      * Separates self-heating from proximity heating contributions
      * Analyzes failure risk for potted/embedded conditions
      * Creates spatial thermal coupling heatmaps

DEPENDENCIES:
    - Python 3.7+
    - numpy (pip install numpy)
    - pandas (pip install pandas) 
    - matplotlib (pip install matplotlib)
    - scipy (pip install scipy) - for MATLAB file export and filtering
    - seaborn (pip install seaborn) - for enhanced plotting

INPUT FILES:
    - ResearchIR_Test_Outputs/ folder containing:
      * Individual CSV files per ROI (component_name.csv)
      * Individual TXT files per ROI (component_name.txt)
      * Expected format: Time, Temperature columns
    - Component coordinates CSV (optional, for spatial analysis):
      * Columns: Component, X, Y (in mm)
      * Example: hbridge_pcb_components_enhanced.csv

OUTPUT FILES:
    - IEEE-format plots (PNG/PDF) with component type subplots
    - MATLAB .mat file with organized temperature data
    - Statistical analysis CSV with transient characteristics
    - Summary report with temperature uniformity analysis
    - **NEW: Spatial coupling visualizations**
      * Stacked bar chart: self-heating vs proximity heating
      * Spatial thermal coupling map with PCB layout overlay
      * Potted condition risk analysis CSV
      * Component failure prediction for embedded systems

USAGE:
    # Basic thermal analysis:
    python researchir_post_processor.py --input ResearchIR_Test_Outputs --output results
    
    # With spatial thermal coupling analysis (default 10mm proximity threshold):
    python researchir_post_processor.py --input ResearchIR_Outputs_HBridge --output results \\
        --coordinates hbridge_pcb_components_enhanced.csv
    
    # Custom proximity threshold for thermal coupling (20mm):
    python researchir_post_processor.py --input ResearchIR_Outputs_HBridge --output results \\
        --coordinates hbridge_pcb_components_enhanced.csv --proximity_threshold 20.0
    
    # Disable spatial analysis:
    python researchir_post_processor.py --input ResearchIR_Test_Outputs --output results --no_spatial
    
    # Custom component grouping:
    python researchir_post_processor.py --input ResearchIR_Test_Outputs --custom_groups config.json

SPATIAL THERMAL COUPLING ANALYSIS:
    Analyzes how nearby components thermally influence each other, critical for:
    - Potted/embedded systems (sand, epoxy, potting compounds)
    - High-density PCB layouts with thermal hotspots
    - Failure prediction when convective cooling is eliminated
    
    Metrics calculated:
    - Temperature correlation between neighbors
    - Thermal gradient influence factors
    - Coupling strength (distance-weighted correlation)
    - Self-heating vs proximity heating decomposition
    - Potted condition temperature rise estimates (2.5x multiplier)
    - Component failure risk scores

===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import os
import sys
import glob
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import argparse
import json
from pathlib import Path

try:
    from scipy.io import savemat
    from scipy import signal
    from scipy.ndimage import median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available. MATLAB export and filtering will be disabled.")
    print("Install with: pip install scipy")
    SCIPY_AVAILABLE = False

class ResearchIRPostProcessor:
    """Post-process ResearchIR thermal data exports"""
    
    def __init__(self, debug=False, enable_filtering=True, filter_type='median', filter_params=None, 
                 coordinates_file=None, proximity_threshold=10.0):
        self.debug = debug
        self.enable_filtering = enable_filtering
        self.filter_type = filter_type
        self.filter_params = filter_params or {}
        self.component_data = {}
        self.component_groups = {}
        self.time_range = None
        self.coordinates_file = coordinates_file
        self.proximity_threshold = proximity_threshold  # mm
        self.component_coordinates = {}
        self.proximity_matrix = {}
        self.thermal_coupling_data = {}
        
        # Default component type mapping
        self.default_component_types = {
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
        
        # IEEE plotting configuration
        self.ieee_config = {
            'figure_width': 7.16,  # IEEE single column width in inches
            'figure_height': 9.0,  # Adjustable height
            'font_size': 8,
            'title_size': 8,
            'label_size': 8,
            'legend_size': 8,
            'line_width': 1.5,
            'marker_size': 4,
            'dpi': 300
        }
    
    def load_researchir_files(self, input_folder: str) -> Dict:
        """Load all CSV/TXT files from ResearchIR export folder"""
        
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
                # Extract base name from filename 
                filename = os.path.basename(file_path)
                base_name = filename.replace(' - Stats.txt', '').replace('_Stats.txt', '').replace('.txt', '')
                
                if self.debug:
                    print(f"Processing {filename}")
                
                # Parse ResearchIR stats file
                component_temps = self._parse_researchir_stats_file(file_path)
                
                if component_temps:
                    # Create time series data for each component
                    for component_name, temp_value in component_temps.items():
                        if component_name not in component_data:
                            component_data[component_name] = []
                        
                        # Simulate time data - in real thermal test, this would come from frame timing
                        # For now, create a simple time series
                        time_point = len(component_data[component_name])
                        component_data[component_name].append({'Time': time_point * 60, 'Temperature': temp_value})  # Time in seconds, 1 minute intervals
                
                if self.debug:
                    print(f"Extracted {len(component_temps)} components from {filename}")
                        
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                if self.debug:
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
                
                if self.debug:
                    print(f"Component {component_name}: {len(df)} data points, temp range {df['Temperature'].min():.1f}-{df['Temperature'].max():.1f}°C")
        
        if not final_component_data:
            raise ValueError("No valid component data could be loaded from ResearchIR files")
        
        print(f"Successfully loaded {len(final_component_data)} components from ResearchIR stats")
        return final_component_data
    
    def _parse_researchir_stats_file(self, file_path: str) -> Dict[str, float]:
        """Parse ResearchIR stats file format"""
        
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
            if self.debug:
                print(f"Could not find header or mean temperature line in {file_path}")
            return component_temps
        
        # Parse ResearchIR tab-delimited format with fixed-width columns
        # Split on multiple spaces (ResearchIR uses space-aligned columns)
        import re
        
        # Use regex to split on multiple whitespace characters
        header_parts = re.split(r'\s{2,}', header_line.strip())
        mean_parts = re.split(r'\s{2,}', mean_line.strip())
        
        if self.debug:
            print(f"Header parts: {len(header_parts)}")
            print(f"Mean parts: {len(mean_parts)}")
            print(f"First few headers: {header_parts[:5]}")
            print(f"First few means: {mean_parts[:5]}")
        
        # Skip the first column (statistic description)
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
                
                if self.debug:
                    print(f"  {component_name}: {temp_value}°C")
                    
            except (ValueError, IndexError):
                if self.debug:
                    print(f"  Could not parse temperature for '{component_name}': '{temp_str}'")
                continue
        
        return component_temps
    
    def classify_components(self, component_data: Dict, custom_groups: Optional[Dict] = None) -> Dict:
        """Classify components by type based on designator prefixes"""
        
        if custom_groups:
            component_types = {**self.default_component_types, **custom_groups}
        else:
            component_types = self.default_component_types
        
        grouped_components = {}
        unclassified = []
        
        for component_name in component_data.keys():
            classified = False
            
            # Try to match component prefix
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
            print(f"Unclassified components ({len(unclassified)}): {unclassified[:10]}{'...' if len(unclassified) > 10 else ''}")
            
            # Group by common patterns or put in miscellaneous
            misc_patterns = self._analyze_unclassified_patterns(unclassified)
            
            for component in unclassified:
                # Try pattern matching for common misc components
                matched_pattern = False
                for pattern, group_name in misc_patterns.items():
                    if re.match(pattern, component, re.IGNORECASE):
                        if group_name not in grouped_components:
                            grouped_components[group_name] = []
                        grouped_components[group_name].append(component)
                        matched_pattern = True
                        break
                
                # Put remaining in miscellaneous
                if not matched_pattern:
                    if 'MISC' not in grouped_components:
                        grouped_components['MISC'] = []
                    grouped_components['MISC'].append(component)
        
        # Print grouping summary
        print("\nComponent Classification:")
        for group, components in grouped_components.items():
            group_name = component_types.get(group, {}).get('name', group)
            print(f"  {group_name}: {len(components)} components")
        
        return grouped_components
    
    def _analyze_unclassified_patterns(self, unclassified: List[str]) -> Dict[str, str]:
        """Analyze patterns in unclassified components"""
        
        patterns = {}
        
        # Common patterns for miscellaneous components
        common_patterns = [
            (r'^(FB|FERR)', 'FB'),  # Ferrite beads
            (r'^(X|XTAL)', 'CR'),   # Crystals (alternative naming)
            (r'^(Y)', 'CR'),        # Crystal oscillators
            (r'^(F|FUSE)', 'F'),    # Fuses
            (r'^(P|PAD)', 'TP'),    # Pads/test points
            (r'^(H|HOLE)', 'MECH'), # Mechanical holes
            (r'^(MH)', 'MECH'),     # Mounting holes
            (r'^\d+$', 'MISC'),     # Pure numbers
        ]
        
        for pattern, group in common_patterns:
            patterns[pattern] = group
        
        return patterns
    
    def apply_thermal_filtering(self, temperatures: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Apply filtering to remove camera refocusing artifacts while preserving thermal transients
        
        Available filters:
        - 'savgol': Savitzky-Golay filter (preserves peaks, removes noise)
        - 'median': Median filter (removes spikes, good for refocus artifacts)  
        - 'lowpass': Butterworth low-pass filter (smooth overall trends)
        - 'outlier': Statistical outlier removal + interpolation
        - 'hybrid': Combination of outlier removal + Savitzky-Golay
        """
        if not self.enable_filtering or not SCIPY_AVAILABLE or len(temperatures) < 5:
            return temperatures
            
        filtered_temps = temperatures.copy()
        
        try:
            if self.filter_type == 'savgol':
                # Savitzky-Golay filter - excellent for preserving peaks while smoothing
                window_length = self.filter_params.get('window_length', min(11, len(temperatures)//3))
                if window_length % 2 == 0:
                    window_length += 1  # Must be odd
                window_length = max(5, min(window_length, len(temperatures)))
                polyorder = self.filter_params.get('polyorder', min(3, window_length-1))
                
                filtered_temps = signal.savgol_filter(temperatures, window_length, polyorder)
                
            elif self.filter_type == 'median':
                # Median filter - great for removing sudden spikes from refocusing
                kernel_size = self.filter_params.get('kernel_size', 5)
                filtered_temps = median_filter(temperatures, size=kernel_size)
                
            elif self.filter_type == 'lowpass':
                # Low-pass Butterworth filter
                cutoff_freq = self.filter_params.get('cutoff_freq', 0.1)  # Normalized frequency
                order = self.filter_params.get('order', 4)
                
                b, a = signal.butter(order, cutoff_freq, btype='low')
                filtered_temps = signal.filtfilt(b, a, temperatures)
                
            elif self.filter_type == 'outlier':
                # Statistical outlier removal with interpolation
                std_threshold = self.filter_params.get('std_threshold', 2.5)
                
                # Calculate rolling statistics
                window_size = min(7, len(temperatures)//4)
                rolling_mean = pd.Series(temperatures).rolling(window=window_size, center=True).mean()
                rolling_std = pd.Series(temperatures).rolling(window=window_size, center=True).std()
                
                # Identify outliers (likely refocus artifacts)
                outlier_mask = np.abs(temperatures - rolling_mean) > (std_threshold * rolling_std)
                
                # Replace outliers with interpolated values
                filtered_temps = temperatures.copy()
                outlier_indices = np.where(outlier_mask)[0]
                
                for idx in outlier_indices:
                    # Find nearest non-outlier points for interpolation
                    left_idx = max(0, idx - 1)
                    right_idx = min(len(temperatures) - 1, idx + 1)
                    
                    # Move outward to find non-outlier points
                    while left_idx > 0 and outlier_mask[left_idx]:
                        left_idx -= 1
                    while right_idx < len(temperatures) - 1 and outlier_mask[right_idx]:
                        right_idx += 1
                    
                    # Linear interpolation
                    if left_idx != right_idx:
                        filtered_temps[idx] = np.interp(times[idx], 
                                                      [times[left_idx], times[right_idx]],
                                                      [temperatures[left_idx], temperatures[right_idx]])
                    
            elif self.filter_type == 'hybrid':
                # Hybrid approach: outlier removal followed by Savitzky-Golay
                # First pass: remove outliers
                temp_processor = ResearchIRPostProcessor(debug=False, enable_filtering=True, 
                                                       filter_type='outlier', filter_params=self.filter_params)
                intermediate_temps = temp_processor.apply_thermal_filtering(temperatures, times)
                
                # Second pass: smooth with Savitzky-Golay
                window_length = self.filter_params.get('window_length', min(9, len(temperatures)//3))
                if window_length % 2 == 0:
                    window_length += 1
                window_length = max(5, min(window_length, len(temperatures)))
                polyorder = self.filter_params.get('polyorder', min(2, window_length-1))
                
                filtered_temps = signal.savgol_filter(intermediate_temps, window_length, polyorder)
            
            if self.debug:
                print(f"Applied {self.filter_type} filter to remove refocusing artifacts")
                
        except Exception as e:
            if self.debug:
                print(f"Filtering failed: {e}, using original data")
            filtered_temps = temperatures
            
        return filtered_temps
    
    def analyze_temperature_transients(self, component_data: Dict) -> Dict:
        """Analyze temperature transients for each component"""
        
        analysis_results = {}
        
        for component_name, df in component_data.items():
            if len(df) < 2:
                continue
            
            temps = df['Temperature'].values
            times = df['Time'].values
            
            # Apply filtering to remove camera refocusing artifacts
            filtered_temps = self.apply_thermal_filtering(temps, times)
            
            # Calculate transient characteristics using filtered data
            initial_temp = filtered_temps[0]
            final_temp = filtered_temps[-1]
            delta_temp = final_temp - initial_temp
            max_temp = np.max(filtered_temps)
            min_temp = np.min(filtered_temps)
            mean_temp = np.mean(filtered_temps)
            std_temp = np.std(filtered_temps)
            
            # Find peak time and settling characteristics
            max_idx = np.argmax(filtered_temps)
            peak_time = times[max_idx] if max_idx < len(times) else times[-1]
            
            # Calculate rate of change using filtered data
            if len(filtered_temps) > 1:
                temp_gradient = np.gradient(filtered_temps, times)
                max_rate = np.max(np.abs(temp_gradient))
            else:
                temp_gradient = [0]
                max_rate = 0
            
            analysis_results[component_name] = {
                'initial_temp': initial_temp,
                'final_temp': final_temp,
                'delta_temp': delta_temp,
                'max_temp': max_temp,
                'min_temp': min_temp,
                'mean_temp': mean_temp,
                'std_temp': std_temp,
                'peak_time': peak_time,
                'max_rate': max_rate,
                'temp_range': max_temp - min_temp,
                'data_points': len(temps)
            }
        
        return analysis_results
    
    def apply_filtering_to_component_data(self, component_data: Dict) -> Dict:
        """Apply filtering to all component temperature data for plotting"""
        
        if not self.enable_filtering:
            return component_data
            
        filtered_component_data = {}
        
        for component_name, df in component_data.items():
            if len(df) < 2:
                filtered_component_data[component_name] = df.copy()
                continue
                
            df_copy = df.copy()
            temps = df_copy['Temperature'].values
            times = df_copy['Time'].values
            
            # Apply filtering
            filtered_temps = self.apply_thermal_filtering(temps, times)
            
            # Update dataframe with filtered temperatures
            df_copy['Temperature'] = filtered_temps
            
            # Store both original and filtered for comparison if debug mode
            if self.debug:
                df_copy['Temperature_Original'] = temps
                df_copy['Temperature_Filtered'] = filtered_temps
            
            filtered_component_data[component_name] = df_copy
            
        return filtered_component_data
    
    def create_filtering_comparison_plots(self, original_data: Dict, filtered_data: Dict, 
                                        grouped_components: Dict, output_dir: str) -> List[str]:
        """Create side-by-side comparison plots showing original vs filtered data"""
        
        if not self.enable_filtering or len(grouped_components) == 0:
            return []
        
        # Set up IEEE-style plotting
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': self.ieee_config['font_size'],
            'axes.titlesize': self.ieee_config['title_size'],
            'axes.labelsize': self.ieee_config['label_size'],
            'legend.fontsize': self.ieee_config['legend_size'],
            'lines.linewidth': self.ieee_config['line_width'],
            'lines.markersize': self.ieee_config['marker_size'],
            'figure.dpi': self.ieee_config['dpi'],
            'savefig.dpi': self.ieee_config['dpi'],
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'text.usetex': False,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        output_files = []
        
        # Select top 6 component groups with most components for comparison
        sorted_groups = sorted(grouped_components.items(), key=lambda x: len(x[1]), reverse=True)
        top_groups = sorted_groups[:6]  # Limit to 6 rows for readability
        
        if len(top_groups) == 0:
            return output_files
        
        # Create figure with 2 columns (original vs filtered) and rows for each component type
        n_rows = len(top_groups)
        fig, axes = plt.subplots(n_rows, 2, figsize=(7, 1.5 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Thermal Filtering Comparison: Original vs {self.filter_type.title()} Filtered', 
                    fontsize=8, y=0.98)
        
        for row_idx, (group_key, components) in enumerate(top_groups):
            group_info = self.default_component_types.get(group_key, 
                                                        {'name': group_key, 'color': '#888888', 'marker': 'o'})
            
            # Left subplot: Original data
            ax_orig = axes[row_idx, 0]
            # Right subplot: Filtered data  
            ax_filt = axes[row_idx, 1]
            
            # Calculate noise statistics for the group
            orig_stds = []
            filt_stds = []
            
            # Plot up to 8 components per group to avoid clutter
            plot_components = components[:8]
            
            for i, component in enumerate(plot_components):
                if component in original_data and component in filtered_data:
                    orig_df = original_data[component]
                    filt_df = filtered_data[component]
                    
                    # Extract temperatures and times
                    orig_temps = orig_df['Temperature'].values
                    filt_temps = filt_df['Temperature'].values
                    times = orig_df['Time'].values
                    
                    # Calculate noise (std deviation)
                    orig_stds.append(np.std(orig_temps))
                    filt_stds.append(np.std(filt_temps))
                    
                    # Use different line styles for visibility
                    line_style = ['-', '--', '-.', ':'][i % 4]
                    alpha = 0.7 if len(plot_components) > 4 else 1.0
                    
                    # Convert time to minutes for x-axis
                    def seconds_to_minutes(x, pos):
                        return f'{int(x/60)}'
                    from matplotlib.ticker import FuncFormatter
                    
                    # Plot original data
                    ax_orig.plot(times, orig_temps, 
                               color=group_info['color'], 
                               linestyle=line_style,
                               alpha=alpha,
                               linewidth=self.ieee_config['line_width'],
                               label=component if len(plot_components) <= 4 else None)
                    
                    # Plot filtered data
                    ax_filt.plot(times, filt_temps, 
                               color=group_info['color'], 
                               linestyle=line_style,
                               alpha=alpha,
                               linewidth=self.ieee_config['line_width'],
                               label=component if len(plot_components) <= 4 else None)
            
            # Calculate noise reduction statistics
            avg_orig_std = np.mean(orig_stds) if orig_stds else 0
            avg_filt_std = np.mean(filt_stds) if filt_stds else 0
            noise_reduction = ((avg_orig_std - avg_filt_std) / avg_orig_std * 100) if avg_orig_std > 0 else 0
            
            # Format subplots
            for ax, title_suffix in [(ax_orig, 'Original'), (ax_filt, f'{self.filter_type.title()} Filtered')]:
                ax.set_title(f"{group_info['name']} - {title_suffix}", 
                           fontsize=8)
                ax.set_ylabel('Temp. (°C)', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(FuncFormatter(seconds_to_minutes))
                
                # Only show x-label on bottom row
                if row_idx == len(top_groups) - 1:
                    ax.set_xlabel('Time (minutes)', fontsize=8)
                
                # Add legend for small groups only
                if len(plot_components) <= 4 and any(comp in original_data for comp in plot_components):
                    ax.legend(fontsize=8, loc='best')
            
            # # Add noise reduction statistics to the filtered plot
            # stats_text = f"Noise Reduction: {noise_reduction:.1f}%\\nAvg σ: {avg_orig_std:.2f}°C → {avg_filt_std:.2f}°C"
            # ax_filt.text(0.02, 0.98, stats_text, transform=ax_filt.transAxes, 
            #             verticalalignment='top', 
            #             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            #             fontsize=self.ieee_config['legend_size'])
        
        plt.tight_layout()
        
        # Save comparison plot
        base_filename = os.path.join(output_dir, f'filtering_comparison_{self.filter_type}')
        
        # PNG for presentations
        png_file = f"{base_filename}.png"
        plt.savefig(png_file, dpi=self.ieee_config['dpi'], bbox_inches='tight')
        output_files.append(png_file)
        
        # PDF for publications
        pdf_file = f"{base_filename}.pdf"
        plt.savefig(pdf_file, bbox_inches='tight')
        output_files.append(pdf_file)
        
        plt.close()
        
        if self.debug:
            print(f"Created filtering comparison plots: {len(output_files)} files")
        
        return output_files
    
    def create_multifilter_component_grid(self, original_data: Dict, grouped_components: Dict, 
                                         output_dir: str) -> Tuple[str, pd.DataFrame]:
        """
        Create comprehensive grid comparing all filtering methods across component types
        Returns: (plot_filename, comparison_dataframe)
        """
        
        if len(grouped_components) == 0:
            return None, None
        
        # Define all filter methods to compare
        filter_methods = [
            ('none', 'Original', {}),
            ('savgol', 'Savitzky-Golay', {'window_length': 11, 'polyorder': 3}),
            ('hybrid', 'Hybrid', {'z_threshold': 2.0, 'window_length': 11, 'polyorder': 3}),
            ('median', 'Median', {'kernel_size': 5}),
            ('lowpass', 'Butterworth', {'cutoff_freq': 0.01, 'filter_order': 4}),
            ('outlier', 'Outlier Removal', {'z_threshold': 2.0})
        ]
        
        # Select top component types (limit to 6 rows for readability)
        sorted_groups = sorted(grouped_components.items(), key=lambda x: len(x[1]), reverse=True)
        top_groups = sorted_groups[:6]
        
        if len(top_groups) == 0:
            return None, None
        
        # Select one representative component from each group
        representative_components = {}
        for group_key, components in top_groups:
            # Choose component with most data points or middle of sorted list
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
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 8,
            'axes.titlesize': 8,
            'axes.labelsize': 8,
            'legend.fontsize': 8,
            'lines.linewidth': 1.5,
            'figure.dpi': self.ieee_config['dpi'],
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'axes.grid': True,
            'grid.alpha': 0.4
        })
        
        # Create grid: rows=component types, cols=filter methods
        n_rows = len(representative_components)
        n_cols = len(filter_methods)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Filtering Method Comparison Across Component Types', 
                    fontsize=8, fontweight='bold', y=0.995)
        
        # Storage for steady-state comparison metrics
        comparison_metrics = []
        
        def seconds_to_minutes(x, pos):
            return f'{int(x/60)}'
        from matplotlib.ticker import FuncFormatter
        
        # Process each component type (rows)
        for row_idx, (group_key, component) in enumerate(representative_components.items()):
            group_info = self.default_component_types.get(group_key, 
                                                        {'name': group_key, 'color': '#888888'})
            
            if component not in original_data:
                continue
            
            orig_df = original_data[component]
            orig_temps = orig_df['Temperature'].values
            times = orig_df['Time'].values
            
            # Calculate steady-state temperature (average of last 10% of data)
            steady_state_window = max(1, len(orig_temps) // 10)
            orig_steady_state = np.mean(orig_temps[-steady_state_window:])
            
            # Process each filter method (columns)
            for col_idx, (filter_type, filter_name, filter_params) in enumerate(filter_methods):
                ax = axes[row_idx, col_idx]
                
                # Apply filtering
                if filter_type == 'none':
                    filtered_temps = orig_temps.copy()
                else:
                    # Temporarily modify processor settings
                    old_filter_type = self.filter_type
                    old_filter_params = self.filter_params
                    self.filter_type = filter_type
                    self.filter_params = filter_params
                    
                    filtered_temps = self.apply_thermal_filtering(orig_temps, times)
                    
                    # Restore original settings
                    self.filter_type = old_filter_type
                    self.filter_params = old_filter_params
                
                # Calculate metrics
                filtered_steady_state = np.mean(filtered_temps[-steady_state_window:])
                steady_state_error = abs(filtered_steady_state - orig_steady_state)
                steady_state_error_pct = (steady_state_error / orig_steady_state * 100) if orig_steady_state != 0 else 0
                noise_std = np.std(filtered_temps)
                
                # Calculate transient preservation (compare initial rise)
                initial_window = min(5, len(orig_temps) // 4)
                orig_initial_rise = orig_temps[initial_window] - orig_temps[0]
                filt_initial_rise = filtered_temps[initial_window] - filtered_temps[0]
                transient_preservation = (filt_initial_rise / orig_initial_rise * 100) if orig_initial_rise != 0 else 100
                
                # Store metrics
                comparison_metrics.append({
                    'Component_Type': group_info['name'],
                    'Component': component,
                    'Filter_Method': filter_name,
                    'Original_Steady_State_C': f'{orig_steady_state:.2f}',
                    'Filtered_Steady_State_C': f'{filtered_steady_state:.2f}',
                    'Steady_State_Error_C': f'{steady_state_error:.3f}',
                    'Steady_State_Error_Pct': f'{steady_state_error_pct:.2f}',
                    'Noise_StdDev_C': f'{noise_std:.3f}',
                    'Transient_Preservation_Pct': f'{transient_preservation:.1f}'
                })
                
                # Plot
                ax.plot(times, filtered_temps, color=group_info['color'], linewidth=1.5)
                ax.xaxis.set_major_formatter(FuncFormatter(seconds_to_minutes))
                ax.grid(True, alpha=0.3)
                
                # Titles only on top row
                if row_idx == 0:
                    ax.set_title(filter_name, fontsize=8, fontweight='bold')
                
                # Y-labels only on left column
                if col_idx == 0:
                    ax.set_ylabel(f"{group_info['name']}\\nTemp (°C)", fontsize=8)
                
                # X-labels only on bottom row
                if row_idx == len(representative_components) - 1:
                    ax.set_xlabel('Time (min)', fontsize=8)
                
                # Add metrics annotation
                metrics_text = f"σ={noise_std:.2f}°C\\nΔSS={steady_state_error_pct:.1f}%"
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save grid plot
        base_filename = os.path.join(output_dir, 'filtering_method_comparison_grid')
        
        png_file = f"{base_filename}.png"
        plt.savefig(png_file, dpi=self.ieee_config['dpi'], bbox_inches='tight')
        
        pdf_file = f"{base_filename}.pdf"
        plt.savefig(pdf_file, bbox_inches='tight')
        
        plt.close()
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_metrics)
        
        # Save comparison table
        csv_file = os.path.join(output_dir, 'filtering_method_comparison_table.csv')
        comparison_df.to_csv(csv_file, index=False)
        
        print(f"Created multi-filter comparison grid: {png_file}")
        print(f"Created comparison metrics table: {csv_file}")
        
        return png_file, comparison_df
    
    def create_ieee_plots(self, component_data: Dict, grouped_components: Dict, 
                         analysis_results: Dict, output_dir: str) -> List[str]:
        """Create IEEE-format plots with subplots by component type"""
        
        # Set up IEEE-style plotting
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': self.ieee_config['font_size'],
            'axes.titlesize': self.ieee_config['title_size'],
            'axes.labelsize': self.ieee_config['label_size'],
            'legend.fontsize': self.ieee_config['legend_size'],
            'lines.linewidth': self.ieee_config['line_width'],
            'lines.markersize': self.ieee_config['marker_size'],
            'figure.dpi': self.ieee_config['dpi'],
            'savefig.dpi': self.ieee_config['dpi'],
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        output_files = []
        
        # Create comprehensive multi-subplot figure
        n_groups = len(grouped_components)
        if n_groups == 0:
            return output_files
        
        # Calculate subplot layout (prefer more rows than columns for IEEE format)
        cols = min(2, n_groups)
        rows = (n_groups + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(self.ieee_config['figure_width'] * cols, 
                                                     self.ieee_config['figure_height'] * rows * 0.6))
        
        if n_groups == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # Plot each component group
        group_idx = 0
        for group_key, components in grouped_components.items():
            if group_idx >= len(axes):
                break
            
            ax = axes[group_idx]
            group_info = self.default_component_types.get(group_key, 
                                                        {'name': group_key, 'color': '#888888', 'marker': 'o'})
            
            # Plot temperature transients for this group
            for i, component in enumerate(components[:20]):  # Limit to 20 components per plot
                if component in component_data:
                    df = component_data[component]
                    
                    # Use different line styles for visibility
                    line_style = ['-', '--', '-.', ':'][i % 4]
                    alpha = 0.7 if len(components) > 5 else 1.0
                    
                    ax.plot(df['Time'], df['Temperature'], 
                           color=group_info['color'], 
                           linestyle=line_style,
                           alpha=alpha,
                           label=component if len(components) <= 5 else None,
                           linewidth=self.ieee_config['line_width'])
            
            # Formatting with proper time axis
            ax.set_title(f"{group_info['name']} Temperature Transients", 
                        fontsize=8)
            ax.set_xlabel('Time (minutes)', fontsize=8)
            ax.set_ylabel('Temperature (°C)', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Convert x-axis to minutes for better readability
            # Use a function to convert seconds to minutes for tick labels
            def seconds_to_minutes(x, pos):
                return f'{int(x/60)}'
            
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(seconds_to_minutes))
            
            # Add statistics text box
            group_temps = []
            for comp in components:
                if comp in analysis_results:
                    group_temps.append(analysis_results[comp]['mean_temp'])
            
            if group_temps:
                stats_text = f"n={len(components)}\nμ={np.mean(group_temps):.1f}°C\nσ={np.std(group_temps):.1f}°C"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Legend for small groups only
            if len(components) <= 5 and any(component in component_data for component in components):
                ax.legend(fontsize=8, loc='best')
            
            group_idx += 1
        
        # Hide unused subplots
        for idx in range(group_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.close()
        
        # Create summary statistics plot only (removed thermal_transients_by_component_type)
        self._create_summary_statistics_plot(grouped_components, analysis_results, output_dir, output_files)
        
        return output_files
    
    def create_thermal_coupling_visualizations(self, component_data: Dict, coupling_metrics: Dict, 
                                              analysis_results: Dict, output_dir: str) -> List[str]:
        """
        Create comprehensive thermal coupling visualization plots:
        1. Stacked bar chart: self-heating vs proximity heating
        2. Spatial thermal coupling heatmap
        3. Component risk assessment for potted conditions
        """
        
        output_files = []
        
        if not coupling_metrics:
            print("Warning: No coupling metrics available for visualization")
            return output_files
        
        # Set up plotting
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 8,
            'axes.titlesize': 8,
            'axes.labelsize': 8,
            'legend.fontsize': 8,
            'lines.linewidth': 1.5,
            'figure.dpi': self.ieee_config['dpi'],
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        # --- Plot 1: Stacked Bar Chart - Self-Heating vs Proximity Heating ---
        
        # Select top 15 components by total temperature rise
        sorted_components = sorted(
            [(comp, analysis_results[comp]['delta_temp']) for comp in coupling_metrics.keys() if comp in analysis_results],
            key=lambda x: x[1],
            reverse=True
        )[:15]
        
        if sorted_components:
            fig, ax = plt.subplots(figsize=(7, 5))
            
            component_names = []
            self_heating_vals = []
            proximity_heating_vals = []
            activity_colors = []
            
            for comp, _ in sorted_components:
                heating_sources = self.estimate_heating_sources(comp, coupling_metrics, analysis_results)
                
                if heating_sources:
                    component_names.append(comp)
                    self_heating_vals.append(heating_sources['self_heating_C'])
                    proximity_heating_vals.append(heating_sources['proximity_heating_C'])
                    
                    # Color by activity type
                    if heating_sources['component_activity'] == 'active':
                        activity_colors.append('#d62728')  # Red
                    elif heating_sources['component_activity'] == 'passive_high_heat':
                        activity_colors.append('#ff7f0e')  # Orange
                    else:
                        activity_colors.append('#1f77b4')  # Blue
            
            # Create stacked bar chart
            x_pos = np.arange(len(component_names))
            
            bars1 = ax.bar(x_pos, self_heating_vals, label='Self-Heating', color='#ff7f0e', alpha=0.8)
            bars2 = ax.bar(x_pos, proximity_heating_vals, bottom=self_heating_vals, 
                          label='Proximity Heating', color='#2ca02c', alpha=0.8)
            
            ax.set_xlabel('Component', fontsize=8)
            ax.set_ylabel('Temperature Rise (°C)', fontsize=8)
            ax.set_title('Heat Source Decomposition: Self-Heating vs Proximity Heating', fontsize=8, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(component_names, fontsize=7)
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.tick_params(axis='y', labelsize=8)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            filename = os.path.join(output_dir, 'thermal_coupling_heat_sources')
            plt.savefig(f"{filename}.png", dpi=self.ieee_config['dpi'], bbox_inches='tight')
            plt.savefig(f"{filename}.pdf", bbox_inches='tight')
            output_files.extend([f"{filename}.png", f"{filename}.pdf"])
            plt.close()
        
        # --- Plot 2: Spatial Thermal Coupling Map ---
        
        if self.component_coordinates:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot all components as points
            for comp in component_data.keys():
                if comp not in self.component_coordinates:
                    continue
                
                coords = self.component_coordinates[comp]
                activity = self.classify_thermal_activity(comp, analysis_results)
                delta_temp = analysis_results[comp]['delta_temp'] if comp in analysis_results else 0
                
                # Size by temperature rise
                marker_size = 30 + delta_temp * 10
                
                # Color by activity and temperature
                if activity == 'active':
                    color = plt.cm.Reds(min(delta_temp / 30.0, 1.0))
                    marker = 's'  # Square for active
                elif activity == 'passive_high_heat':
                    color = plt.cm.Oranges(min(delta_temp / 20.0, 1.0))
                    marker = '^'  # Triangle
                else:
                    color = plt.cm.Blues(min(delta_temp / 10.0, 1.0))
                    marker = 'o'  # Circle for passive
                
                ax.scatter(coords['x'], coords['y'], s=marker_size, c=[color], 
                          marker=marker, alpha=0.7, edgecolors='black', linewidth=0.5)
                
                # Label high-temperature components
                if delta_temp > 10.0:
                    ax.annotate(comp, (coords['x'], coords['y']), 
                              fontsize=6, ha='center', va='bottom')
            
            # Draw proximity connections for high coupling strength
            for comp, metrics in coupling_metrics.items():
                if comp not in self.component_coordinates:
                    continue
                
                comp_coords = self.component_coordinates[comp]
                
                for neighbor_info in metrics['neighbors'][:3]:  # Top 3 strongest couplings
                    neighbor = neighbor_info['neighbor']
                    if neighbor not in self.component_coordinates:
                        continue
                    
                    coupling_strength = neighbor_info['coupling_strength']
                    
                    if coupling_strength > 0.3:  # Significant coupling threshold
                        neighbor_coords = self.component_coordinates[neighbor]
                        
                        # Line thickness by coupling strength
                        linewidth = coupling_strength * 2
                        alpha = min(coupling_strength, 0.7)
                        
                        ax.plot([comp_coords['x'], neighbor_coords['x']], 
                               [comp_coords['y'], neighbor_coords['y']], 
                               'r-', linewidth=linewidth, alpha=alpha)
            
            ax.set_xlabel('X Position (mm)', fontsize=8)
            ax.set_ylabel('Y Position (mm)', fontsize=8)
            ax.set_title('Spatial Thermal Coupling Map', fontsize=8, fontweight='bold')
            ax.grid(True, alpha=0.2)
            ax.set_aspect('equal')
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, label='Active IC'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=8, label='High-Power Passive'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Passive'),
                Line2D([0], [0], color='red', linewidth=2, label='Strong Thermal Coupling')
            ]
            ax.legend(handles=legend_elements, fontsize=8, loc='best')
            
            plt.tight_layout()
            
            filename = os.path.join(output_dir, 'spatial_thermal_coupling_map')
            plt.savefig(f"{filename}.png", dpi=self.ieee_config['dpi'], bbox_inches='tight')
            plt.savefig(f"{filename}.pdf", bbox_inches='tight')
            output_files.extend([f"{filename}.png", f"{filename}.pdf"])
            plt.close()
        
        print(f"Created thermal coupling visualizations: {len(output_files)} files")
        return output_files
    
    def analyze_potted_condition_risk(self, coupling_metrics: Dict, analysis_results: Dict, 
                                     output_dir: str) -> Tuple[str, pd.DataFrame]:
        """
        Analyze component failure risk in potted/embedded conditions
        
        In potted systems:
        - No convective cooling → temperatures rise significantly
        - Thermal coupling dominates → hot components affect neighbors more
        - Cumulative heating effects critical for failure prediction
        """
        
        if not coupling_metrics:
            return None, None
        
        risk_analysis = []
        
        for component, metrics in coupling_metrics.items():
            if component not in analysis_results:
                continue
            
            stats = analysis_results[component]
            heating = self.estimate_heating_sources(component, coupling_metrics, analysis_results)
            
            # Risk factors for potted conditions
            baseline_temp = stats['final_temp']
            proximity_heating = heating['proximity_heating_C']
            num_hot_neighbors = metrics['num_hot_neighbors']
            
            # Estimate temperature rise in potted condition (rough 2-3x multiplier)
            # In potting compound: no air cooling, only conduction
            potted_temp_multiplier = 2.5
            estimated_potted_temp = baseline_temp + (stats['delta_temp'] * (potted_temp_multiplier - 1))
            
            # Risk score (0-10 scale)
            # Factors: absolute temp, proximity heating, number of hot neighbors
            risk_score = 0
            
            if estimated_potted_temp > 85:  # Standard industrial temp limit
                risk_score += 5
            elif estimated_potted_temp > 70:
                risk_score += 3
            elif estimated_potted_temp > 60:
                risk_score += 1
            
            if proximity_heating > 5.0:
                risk_score += 3
            elif proximity_heating > 2.0:
                risk_score += 1
            
            if num_hot_neighbors >= 3:
                risk_score += 2
            elif num_hot_neighbors >= 2:
                risk_score += 1
            
            # Risk classification
            if risk_score >= 7:
                risk_level = 'HIGH'
            elif risk_score >= 4:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            risk_analysis.append({
                'Component': component,
                'Activity_Type': metrics['component_activity'],
                'Baseline_Temp_C': f'{baseline_temp:.1f}',
                'Delta_Temp_C': f'{stats["delta_temp"]:.1f}',
                'Self_Heating_C': f'{heating["self_heating_C"]:.1f}',
                'Proximity_Heating_C': f'{proximity_heating:.1f}',
                'Hot_Neighbors': num_hot_neighbors,
                'Estimated_Potted_Temp_C': f'{estimated_potted_temp:.1f}',
                'Risk_Score': risk_score,
                'Risk_Level': risk_level
            })
        
        # Sort by risk score (highest first)
        risk_analysis.sort(key=lambda x: x['Risk_Score'], reverse=True)
        
        # Create DataFrame
        df_risk = pd.DataFrame(risk_analysis)
        
        # Save to CSV
        csv_filename = os.path.join(output_dir, 'potted_condition_risk_analysis.csv')
        df_risk.to_csv(csv_filename, index=False)
        
        # Print summary
        print(f"\n=== Potted Condition Risk Analysis ===")
        high_risk = sum(1 for r in risk_analysis if r['Risk_Level'] == 'HIGH')
        medium_risk = sum(1 for r in risk_analysis if r['Risk_Level'] == 'MEDIUM')
        low_risk = sum(1 for r in risk_analysis if r['Risk_Level'] == 'LOW')
        
        print(f"HIGH RISK components: {high_risk}")
        print(f"MEDIUM RISK components: {medium_risk}")
        print(f"LOW RISK components: {low_risk}")
        
        if high_risk > 0:
            print(f"\nTop HIGH RISK components:")
            for r in risk_analysis[:min(5, high_risk)]:
                if r['Risk_Level'] == 'HIGH':
                    print(f"  {r['Component']}: Est. potted temp {r['Estimated_Potted_Temp_C']}°C, "
                          f"Risk score {r['Risk_Score']}")
        
        print(f"\nRisk analysis saved to: {csv_filename}")
        
        return csv_filename, df_risk
    
    def _create_summary_statistics_plot(self, grouped_components: Dict, analysis_results: Dict, 
                                      output_dir: str, output_files: List[str]):
        """Create summary statistics comparison plot"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 6))
        
        # Prepare data for plotting
        group_names = []
        mean_temps = []
        std_temps = []
        delta_temps = []
        max_rates = []
        
        for group_key, components in grouped_components.items():
            group_info = self.default_component_types.get(group_key, {'name': group_key})
            group_names.append(group_info['name'])
            
            group_means = [analysis_results[comp]['mean_temp'] for comp in components if comp in analysis_results]
            group_stds = [analysis_results[comp]['std_temp'] for comp in components if comp in analysis_results]
            group_deltas = [analysis_results[comp]['delta_temp'] for comp in components if comp in analysis_results]
            group_rates = [analysis_results[comp]['max_rate'] for comp in components if comp in analysis_results]
            
            mean_temps.append(np.mean(group_means) if group_means else 0)
            std_temps.append(np.mean(group_stds) if group_stds else 0)
            delta_temps.append(np.mean(group_deltas) if group_deltas else 0)
            max_rates.append(np.mean(group_rates) if group_rates else 0)
        
        # Plot 1: Mean temperatures by component type
        ax1.bar(group_names, mean_temps, color=[self.default_component_types.get(k, {'color': '#888888'})['color'] 
                                               for k in grouped_components.keys()])
        ax1.set_title('Mean Temperature by Component Type', fontsize=8)
        ax1.set_ylabel('Temperature (°C)', fontsize=8)
        ax1.tick_params(axis='x', rotation=45, labelsize=7)
        ax1.tick_params(axis='y', labelsize=8)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Plot 2: Temperature variability
        ax2.bar(group_names, std_temps, color=[self.default_component_types.get(k, {'color': '#888888'})['color'] 
                                              for k in grouped_components.keys()])
        ax2.set_title('Temperature Variability (Std Dev)', fontsize=8)
        ax2.set_ylabel('Std Dev (°C)', fontsize=8)
        ax2.tick_params(axis='x', rotation=45, labelsize=7)
        ax2.tick_params(axis='y', labelsize=8)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Plot 3: Temperature rise
        ax3.bar(group_names, delta_temps, color=[self.default_component_types.get(k, {'color': '#888888'})['color'] 
                                                for k in grouped_components.keys()])
        ax3.set_title('Temperature Rise (Final - Initial)', fontsize=8)
        ax3.set_ylabel('ΔT (°C)', fontsize=8)
        ax3.tick_params(axis='x', rotation=45, labelsize=7)
        ax3.tick_params(axis='y', labelsize=8)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Plot 4: Maximum rate of change
        ax4.bar(group_names, max_rates, color=[self.default_component_types.get(k, {'color': '#888888'})['color'] 
                                              for k in grouped_components.keys()])
        ax4.set_title('Maximum Rate of Change', fontsize=8)
        ax4.set_ylabel('Rate (°C/s)', fontsize=8)
        ax4.tick_params(axis='x', rotation=45, labelsize=7)
        ax4.tick_params(axis='y', labelsize=8)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        plt.tight_layout()
        
        # Save summary plot
        summary_file = os.path.join(output_dir, 'thermal_analysis_summary')
        plt.savefig(f"{summary_file}.png", dpi=self.ieee_config['dpi'], bbox_inches='tight')
        plt.savefig(f"{summary_file}.pdf", bbox_inches='tight')
        output_files.extend([f"{summary_file}.png", f"{summary_file}.pdf"])
        
        plt.close()
    
    def export_matlab_data(self, component_data: Dict, grouped_components: Dict, 
                          analysis_results: Dict, output_dir: str) -> str:
        """Export data to MATLAB .mat file with organized cell arrays"""
        
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not available, skipping MATLAB export")
            return None
        
        # Organize data for MATLAB export
        matlab_data = {}
        
        # Component data organized by type
        for group_key, components in grouped_components.items():
            group_info = self.default_component_types.get(group_key, {'name': group_key})
            group_name = group_info['name'].replace(' ', '_').replace('/', '_')
            
            # Create cell arrays for this component type
            component_names = []
            time_data = []
            temp_data = []
            stats_data = []
            
            for component in components:
                if component in component_data:
                    component_names.append(component)
                    
                    df = component_data[component]
                    time_data.append(df['Time'].values)
                    temp_data.append(df['Temperature'].values)
                    
                    if component in analysis_results:
                        stats = analysis_results[component]
                        stats_data.append([
                            stats['initial_temp'],
                            stats['final_temp'],
                            stats['delta_temp'],
                            stats['max_temp'],
                            stats['min_temp'],
                            stats['mean_temp'],
                            stats['std_temp'],
                            stats['peak_time'],
                            stats['max_rate'],
                            stats['temp_range']
                        ])
            
            if component_names:
                matlab_data[f'{group_name}_names'] = component_names
                matlab_data[f'{group_name}_time'] = time_data
                matlab_data[f'{group_name}_temperature'] = temp_data
                matlab_data[f'{group_name}_statistics'] = np.array(stats_data) if stats_data else []
        
        # Add metadata
        matlab_data['metadata'] = {
            'created_date': datetime.now().isoformat(),
            'total_components': len(component_data),
            'component_groups': list(grouped_components.keys()),
            'statistics_columns': ['initial_temp', 'final_temp', 'delta_temp', 'max_temp', 
                                 'min_temp', 'mean_temp', 'std_temp', 'peak_time', 
                                 'max_rate', 'temp_range']
        }
        
        # Save MATLAB file
        mat_filename = os.path.join(output_dir, 'thermal_analysis_data.mat')
        savemat(mat_filename, matlab_data)
        
        print(f"MATLAB data exported to: {mat_filename}")
        
        # Create MATLAB usage script
        self._create_matlab_usage_script(output_dir, grouped_components)
        
        return mat_filename
    
    def _create_matlab_usage_script(self, output_dir: str, grouped_components: Dict):
        """Create MATLAB script showing how to use the exported data"""
        
        script_content = f"""% MATLAB Script for Thermal Analysis Data
% Generated on {datetime.now().isoformat()}
% Load the exported thermal data

clear; clc;

% Load data
data = load('thermal_analysis_data.mat');

% Available component groups:
"""
        
        for group_key, components in grouped_components.items():
            group_info = self.default_component_types.get(group_key, {'name': group_key})
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

% Convert time axis to minutes for better readability
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
            group_info = self.default_component_types.get(group_key, {'name': group_key})
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
    
    def load_component_coordinates(self, coordinates_file: str) -> Dict:
        """Load component X,Y coordinates from CSV file"""
        
        if not os.path.exists(coordinates_file):
            print(f"Warning: Coordinates file not found: {coordinates_file}")
            return {}
        
        try:
            df = pd.read_csv(coordinates_file)
            
            # Expected columns: Component, X, Y (in mm)
            if 'Component' not in df.columns or 'X' not in df.columns or 'Y' not in df.columns:
                # Try alternative column names
                if 'Reference' in df.columns:
                    df['Component'] = df['Reference']
                else:
                    print(f"Warning: Could not find Component/Reference column in {coordinates_file}")
                    return {}
            
            coordinates = {}
            for _, row in df.iterrows():
                comp_name = str(row['Component']).strip()
                x = float(row['X'])
                y = float(row['Y'])
                coordinates[comp_name] = {'x': x, 'y': y}
            
            print(f"Loaded coordinates for {len(coordinates)} components from {coordinates_file}")
            return coordinates
            
        except Exception as e:
            print(f"Error loading coordinates file: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return {}
    
    def calculate_euclidean_distance(self, comp1: str, comp2: str) -> float:
        """Calculate Euclidean distance between two components in mm"""
        
        if comp1 not in self.component_coordinates or comp2 not in self.component_coordinates:
            return float('inf')
        
        c1 = self.component_coordinates[comp1]
        c2 = self.component_coordinates[comp2]
        
        distance = np.sqrt((c1['x'] - c2['x'])**2 + (c1['y'] - c2['y'])**2)
        return distance
    
    def build_proximity_matrix(self, component_data: Dict) -> Dict:
        """Build proximity matrix identifying thermal neighbors for each component"""
        
        if not self.component_coordinates:
            print("Warning: No coordinate data available for proximity analysis")
            return {}
        
        proximity_matrix = {}
        
        for comp1 in component_data.keys():
            if comp1 not in self.component_coordinates:
                continue
            
            neighbors = []
            for comp2 in component_data.keys():
                if comp1 == comp2:
                    continue
                
                distance = self.calculate_euclidean_distance(comp1, comp2)
                
                if distance <= self.proximity_threshold:
                    neighbors.append({
                        'component': comp2,
                        'distance_mm': distance
                    })
            
            # Sort by distance (closest first)
            neighbors.sort(key=lambda x: x['distance_mm'])
            proximity_matrix[comp1] = neighbors
        
        # Statistics
        total_proximities = sum(len(n) for n in proximity_matrix.values())
        avg_neighbors = total_proximities / len(proximity_matrix) if proximity_matrix else 0
        
        print(f"Proximity analysis: {len(proximity_matrix)} components with {total_proximities} thermal neighbor pairs")
        print(f"Average neighbors per component: {avg_neighbors:.1f} (within {self.proximity_threshold}mm)")
        
        return proximity_matrix
    
    def classify_thermal_activity(self, component_name: str, analysis_results: Dict) -> str:
        """
        Classify component as active heat generator or passive based on type and temperature rise
        
        Active generators: ICs, voltage regulators, transistors, diodes with high power dissipation
        Passive: Resistors, capacitors, inductors, connectors with lower temperature rise
        """
        
        if component_name not in analysis_results:
            return 'unknown'
        
        stats = analysis_results[component_name]
        delta_temp = stats['delta_temp']
        
        # Component type prefixes for active components
        active_prefixes = ['U', 'IC', 'VR', 'Q', 'D', 'DL']
        
        # Check if component starts with active prefix
        is_active_type = any(component_name.upper().startswith(prefix) for prefix in active_prefixes)
        
        # Temperature rise threshold for active components (significant self-heating)
        # In PCB thermal analysis, active components typically show >5°C rise
        active_temp_threshold = 5.0  # °C
        
        if is_active_type and delta_temp > active_temp_threshold:
            return 'active'
        elif delta_temp > active_temp_threshold:
            return 'passive_high_heat'  # Passive component with high self-heating (e.g., power resistor)
        else:
            return 'passive'
    
    def calculate_thermal_coupling_metrics(self, component_data: Dict, analysis_results: Dict) -> Dict:
        """
        Calculate comprehensive thermal coupling metrics between nearby components
        
        Returns dictionary with:
        - Temperature correlation coefficients
        - Thermal gradient influence factors
        - Steady-state thermal coupling strength
        - Cumulative heat load estimates
        """
        
        if not self.proximity_matrix:
            print("Warning: No proximity matrix available for thermal coupling analysis")
            return {}
        
        coupling_metrics = {}
        
        for component, neighbors in self.proximity_matrix.items():
            if component not in component_data or not neighbors:
                continue
            
            comp_temps = component_data[component]['Temperature'].values
            comp_times = component_data[component]['Time'].values
            
            neighbor_coupling = []
            
            for neighbor_info in neighbors:
                neighbor = neighbor_info['component']
                distance = neighbor_info['distance_mm']
                
                if neighbor not in component_data:
                    continue
                
                neighbor_temps = component_data[neighbor]['Temperature'].values
                
                # Ensure same length time series (interpolate if needed)
                if len(comp_temps) != len(neighbor_temps):
                    continue
                
                # 1. Temperature Correlation Coefficient
                correlation = np.corrcoef(comp_temps, neighbor_temps)[0, 1]
                
                # 2. Thermal Gradient Influence (steady-state temperature difference)
                comp_steady_state = np.mean(comp_temps[-10:]) if len(comp_temps) >= 10 else comp_temps[-1]
                neighbor_steady_state = np.mean(neighbor_temps[-10:]) if len(neighbor_temps) >= 10 else neighbor_temps[-1]
                temp_gradient = neighbor_steady_state - comp_steady_state
                
                # 3. Thermal Coupling Strength (correlation weighted by inverse distance)
                # Closer components with high correlation = stronger coupling
                coupling_strength = correlation / (1 + distance/10.0)  # Normalize distance to 10mm scale
                
                # 4. Thermal influence factor (if neighbor is hotter, how much does it influence?)
                if temp_gradient > 0:  # Neighbor is hotter
                    thermal_influence = coupling_strength * temp_gradient
                else:
                    thermal_influence = 0.0  # Only count heating influence, not cooling
                
                # Classify neighbor thermal activity
                neighbor_activity = self.classify_thermal_activity(neighbor, analysis_results)
                
                neighbor_coupling.append({
                    'neighbor': neighbor,
                    'distance_mm': distance,
                    'correlation': correlation,
                    'temp_gradient_C': temp_gradient,
                    'coupling_strength': coupling_strength,
                    'thermal_influence': thermal_influence,
                    'neighbor_activity': neighbor_activity,
                    'neighbor_steady_state_C': neighbor_steady_state
                })
            
            # Sort by thermal influence (strongest first)
            neighbor_coupling.sort(key=lambda x: x['thermal_influence'], reverse=True)
            
            # Calculate cumulative heat load from all neighbors
            total_external_influence = sum(nc['thermal_influence'] for nc in neighbor_coupling if nc['thermal_influence'] > 0)
            
            # Component's own thermal activity
            comp_activity = self.classify_thermal_activity(component, analysis_results)
            comp_delta_temp = analysis_results[component]['delta_temp'] if component in analysis_results else 0
            
            coupling_metrics[component] = {
                'neighbors': neighbor_coupling,
                'total_external_influence': total_external_influence,
                'num_hot_neighbors': sum(1 for nc in neighbor_coupling if nc['temp_gradient_C'] > 2.0),
                'component_activity': comp_activity,
                'self_heating_C': comp_delta_temp,
                'estimated_proximity_heating_C': total_external_influence  # Simplified model
            }
        
        return coupling_metrics
    
    def estimate_heating_sources(self, component: str, coupling_metrics: Dict, analysis_results: Dict) -> Dict:
        """
        Separate self-heating from proximity heating for a component
        
        Uses thermal coupling model to estimate:
        - Internal heat generation (self-heating)
        - External heating from nearby hot components
        - Total temperature rise
        """
        
        if component not in coupling_metrics or component not in analysis_results:
            return {}
        
        metrics = coupling_metrics[component]
        stats = analysis_results[component]
        
        total_temp_rise = stats['delta_temp']
        comp_activity = metrics['component_activity']
        
        # Estimate self-heating baseline based on component type
        if comp_activity == 'active':
            # Active components: assume most heating is self-generated
            self_heating_fraction = 0.7  # 70% self-heating
        elif comp_activity == 'passive_high_heat':
            # High-power passive (resistors): significant self-heating
            self_heating_fraction = 0.6  # 60% self-heating
        else:
            # Passive components: mostly heated by surroundings
            self_heating_fraction = 0.2  # 20% self-heating, 80% external
        
        # Calculate heating sources
        estimated_self_heating = total_temp_rise * self_heating_fraction
        estimated_proximity_heating = total_temp_rise * (1 - self_heating_fraction)
        
        # Identify top heat contributors
        top_contributors = []
        if 'neighbors' in metrics:
            for neighbor in metrics['neighbors'][:5]:  # Top 5 contributors
                if neighbor['thermal_influence'] > 0.1:  # Significant influence threshold
                    top_contributors.append({
                        'component': neighbor['neighbor'],
                        'distance_mm': neighbor['distance_mm'],
                        'contribution_C': neighbor['thermal_influence'],
                        'neighbor_temp_C': neighbor['neighbor_steady_state_C']
                    })
        
        return {
            'component': component,
            'total_temp_rise_C': total_temp_rise,
            'self_heating_C': estimated_self_heating,
            'proximity_heating_C': estimated_proximity_heating,
            'self_heating_fraction': self_heating_fraction,
            'proximity_heating_fraction': 1 - self_heating_fraction,
            'component_activity': comp_activity,
            'num_hot_neighbors': metrics['num_hot_neighbors'],
            'top_heat_contributors': top_contributors
        }
    
    def export_summary_csv(self, grouped_components: Dict, analysis_results: Dict, output_dir: str) -> str:
        """Export statistical summary to CSV"""
        
        summary_data = []
        
        for group_key, components in grouped_components.items():
            group_info = self.default_component_types.get(group_key, {'name': group_key})
            
            for component in components:
                if component in analysis_results:
                    stats = analysis_results[component]
                    summary_data.append({
                        'Component': component,
                        'Type': group_info['name'],
                        'Type_Code': group_key,
                        'Initial_Temp_C': stats['initial_temp'],
                        'Final_Temp_C': stats['final_temp'],
                        'Delta_Temp_C': stats['delta_temp'],
                        'Max_Temp_C': stats['max_temp'],
                        'Min_Temp_C': stats['min_temp'],
                        'Mean_Temp_C': stats['mean_temp'],
                        'Std_Temp_C': stats['std_temp'],
                        'Peak_Time_Seconds': stats['peak_time'],
                        'Max_Rate_C_per_s': stats['max_rate'],
                        'Temp_Range_C': stats['temp_range'],
                        'Data_Points': stats['data_points']
                    })
        
        # Create DataFrame and save
        df_summary = pd.DataFrame(summary_data)
        csv_filename = os.path.join(output_dir, 'thermal_analysis_summary.csv')
        df_summary.to_csv(csv_filename, index=False)
        
        print(f"Summary CSV exported to: {csv_filename}")
        return csv_filename
    
    def process_researchir_data(self, input_folder: str, output_dir: str = None, 
                               custom_groups: Dict = None) -> Dict:
        """Complete ResearchIR data processing workflow"""
        
        if output_dir is None:
            output_dir = 'thermal_analysis_results'
        
        # Create output directory (overwrite existing files)
        os.makedirs(output_dir, exist_ok=True)
        
        print("=== ResearchIR Thermal Data Post-Processing ===")
        print(f"Input folder: {input_folder}")
        print(f"Output directory: {output_dir}")
        
        # Load data
        print("\n1. Loading ResearchIR export files...")
        component_data = self.load_researchir_files(input_folder)
        
        # Apply filtering to remove camera refocusing artifacts
        if self.enable_filtering:
            print(f"\n2. Applying {self.filter_type} filtering to remove camera refocusing artifacts...")
        
        # Classify components
        print(f"\n{3 if self.enable_filtering else 2}. Classifying components by type...")
        grouped_components = self.classify_components(component_data, custom_groups)
        
        # Analyze temperature transients (filtering applied internally)
        print(f"\n{4 if self.enable_filtering else 3}. Analyzing temperature transients...")
        analysis_results = self.analyze_temperature_transients(component_data)
        
        # Apply filtering to component data for plotting
        filtered_component_data = self.apply_filtering_to_component_data(component_data)
        
        # Load coordinates and perform spatial analysis if coordinates file provided
        coupling_metrics = {}
        coupling_plots = []
        risk_csv = None
        risk_df = None
        
        if self.coordinates_file:
            step_num = 5 if self.enable_filtering else 4
            print(f"\n{step_num}. Loading component coordinates for spatial analysis...")
            self.component_coordinates = self.load_component_coordinates(self.coordinates_file)
            
            if self.component_coordinates:
                print(f"\n{step_num+1}. Building proximity matrix (threshold: {self.proximity_threshold}mm)...")
                self.proximity_matrix = self.build_proximity_matrix(component_data)
                
                if self.proximity_matrix:
                    print(f"\n{step_num+2}. Calculating thermal coupling metrics...")
                    coupling_metrics = self.calculate_thermal_coupling_metrics(filtered_component_data, analysis_results)
                    
                    print(f"\n{step_num+3}. Creating thermal coupling visualizations...")
                    coupling_plots = self.create_thermal_coupling_visualizations(filtered_component_data, 
                                                                                coupling_metrics, 
                                                                                analysis_results, 
                                                                                output_dir)
                    
                    print(f"\n{step_num+4}. Analyzing potted condition failure risk...")
                    risk_csv, risk_df = self.analyze_potted_condition_risk(coupling_metrics, 
                                                                          analysis_results, 
                                                                          output_dir)
            step_offset = 5 if self.component_coordinates and self.proximity_matrix else 0
        else:
            step_offset = 0
        
        # Create IEEE-format plots
        base_step = (5 if self.enable_filtering else 4) + step_offset
        print(f"\n{base_step}. Creating IEEE-format plots...")
        plot_files = self.create_ieee_plots(filtered_component_data, grouped_components, 
                                           analysis_results, output_dir)
        
        # Create filtering comparison plots if filtering is enabled
        comparison_files = []
        multifilter_grid = None
        comparison_df = None
        if self.enable_filtering:
            print(f"\n{base_step+1}. Creating filtering comparison plots...")
            comparison_files = self.create_filtering_comparison_plots(component_data, filtered_component_data, 
                                                                    grouped_components, output_dir)
            
            print(f"\n{base_step+2}. Creating comprehensive multi-filter comparison grid...")
            multifilter_grid, comparison_df = self.create_multifilter_component_grid(component_data, 
                                                                                     grouped_components, 
                                                                                     output_dir)
        
        # Export MATLAB data
        matlab_step = base_step + (3 if self.enable_filtering else 1)
        print(f"\n{matlab_step}. Exporting MATLAB data...")
        mat_file = self.export_matlab_data(filtered_component_data, grouped_components, 
                                          analysis_results, output_dir)
        
        # Export summary CSV
        print(f"\n{matlab_step+1}. Creating summary reports...")
        csv_file = self.export_summary_csv(grouped_components, analysis_results, output_dir)
        
        # Results summary
        results = {
            'input_folder': input_folder,
            'output_dir': output_dir,
            'total_components': len(component_data),
            'component_groups': grouped_components,
            'analysis_results': analysis_results,
            'filtering_comparison_df': comparison_df,
            'coupling_metrics': coupling_metrics,
            'risk_analysis_df': risk_df,
            'output_files': {
                'plots': plot_files,
                'comparison_plots': comparison_files,
                'multifilter_grid': multifilter_grid,
                'coupling_plots': coupling_plots,
                'matlab': mat_file,
                'summary_csv': csv_file,
                'risk_csv': risk_csv
            }
        }
        
        print(f"\n=== Processing Complete ===")
        print(f"Total components processed: {len(component_data)}")
        print(f"Component groups: {len(grouped_components)}")
        if coupling_metrics:
            print(f"Spatial coupling analysis: {len(coupling_metrics)} components analyzed")
        total_files = (len(plot_files) + len(comparison_files) + len(coupling_plots) + 
                      (1 if mat_file else 0) + (1 if csv_file else 0) + (1 if risk_csv else 0))
        print(f"Output files created: {total_files}")
        
        return results

def display_configuration_ui(config: Dict) -> Dict:
    """
    Interactive UI to display and verify configuration before processing
    
    Returns modified configuration dictionary after user confirmation
    """
    
    print("\n" + "="*80)
    print("RESEARCHIR THERMAL ANALYSIS - CONFIGURATION VERIFICATION")
    print("="*80)
    
    # Display current configuration
    print("\nCURRENT CONFIGURATION:")
    print("-" * 80)
    
    print(f"\nINPUT/OUTPUT:")
    print(f"   Input Folder:        {config['input_folder']}")
    print(f"   Output Directory:    {config['output_dir']}")
    
    # Check if input folder exists
    if os.path.exists(config['input_folder']):
        file_count = len(glob.glob(os.path.join(config['input_folder'], "*.txt")))
        print(f"   [OK] Input folder exists ({file_count} .txt files found)")
    else:
        print(f"   [WARNING] Input folder does not exist!")
    
    print(f"\nSPATIAL THERMAL COUPLING ANALYSIS:")
    if config['spatial_enabled']:
        print(f"   Status:              ENABLED")
        print(f"   Coordinates File:    {config['coordinates_file'] or 'NOT SET'}")
        
        if config['coordinates_file'] and os.path.exists(config['coordinates_file']):
            # Try to count components in coordinates file
            try:
                df = pd.read_csv(config['coordinates_file'])
                comp_count = len(df)
                print(f"   [OK] Coordinates file exists ({comp_count} components)")
            except:
                print(f"   [WARNING] Could not read coordinates file")
        elif config['coordinates_file']:
            print(f"   [WARNING] Coordinates file not found - spatial analysis will be skipped")
        
        print(f"   Proximity Threshold: {config['proximity_threshold']} mm")
    else:
        print(f"   Status:              DISABLED")
    
    print(f"\nADVANCED OPTIONS:")
    print(f"   Custom Grouping:     {config['custom_groups_file'] or 'None (using defaults)'}")
    if config['custom_groups_file'] and os.path.exists(config['custom_groups_file']):
        print(f"   [OK] Custom groups file exists")
    print(f"   Debug Mode:          {'ENABLED' if config['debug'] else 'DISABLED'}")
    print(f"   Filtering:           ENABLED (median filter for camera refocusing artifacts)")
    
    print("\n" + "-" * 80)
    
    # Interactive prompt
    while True:
        print("\nOPTIONS:")
        print("   [1] Continue with current configuration")
        print("   [2] Change input folder")
        print("   [3] Change output directory")
        print("   [4] Toggle spatial analysis ON/OFF")
        print("   [5] Change coordinates file")
        print("   [6] Change proximity threshold")
        print("   [7] Toggle debug mode")
        print("   [8] Cancel and exit")
        print("   [9] Show example command-line usage")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            print("\nConfiguration confirmed. Starting analysis...\n")
            return config
        
        elif choice == '2':
            new_input = input(f"Enter new input folder path [{config['input_folder']}]: ").strip()
            if new_input:
                config['input_folder'] = new_input
                print(f"Input folder updated to: {new_input}")
        
        elif choice == '3':
            new_output = input(f"Enter new output directory [{config['output_dir']}]: ").strip()
            if new_output:
                config['output_dir'] = new_output
                print(f"Output directory updated to: {new_output}")
        
        elif choice == '4':
            config['spatial_enabled'] = not config['spatial_enabled']
            status = "ENABLED" if config['spatial_enabled'] else "DISABLED"
            print(f"Spatial analysis {status}")
        
        elif choice == '5':
            new_coords = input(f"Enter coordinates file path [{config['coordinates_file']}]: ").strip()
            if new_coords:
                config['coordinates_file'] = new_coords
                print(f"Coordinates file updated to: {new_coords}")
        
        elif choice == '6':
            try:
                new_threshold = float(input(f"Enter proximity threshold in mm [{config['proximity_threshold']}]: ").strip())
                config['proximity_threshold'] = new_threshold
                print(f"Proximity threshold updated to: {new_threshold} mm")
            except ValueError:
                print("Invalid number, keeping current value")
        
        elif choice == '7':
            config['debug'] = not config['debug']
            status = "ENABLED" if config['debug'] else "DISABLED"
            print(f"Debug mode {status}")
        
        elif choice == '8':
            print("\nAnalysis cancelled by user")
            sys.exit(0)
        
        elif choice == '9':
            print("\n" + "="*80)
            print("EXAMPLE COMMAND-LINE USAGE:")
            print("="*80)
            print("\n# Basic usage:")
            print("python researchir_post_processor.py --input ResearchIR_Test_Outputs")
            print("\n# With spatial analysis:")
            print("python researchir_post_processor.py --input ResearchIR_Test_Outputs \\")
            print("    --coordinates hbridge_pcb_components_enhanced.csv \\")
            print("    --proximity_threshold 10.0")
            print("\n# Disable spatial analysis:")
            print("python researchir_post_processor.py --input ResearchIR_Test_Outputs \\")
            print("    --no_spatial")
            print("\n# Custom output directory:")
            print("python researchir_post_processor.py --input ResearchIR_Test_Outputs \\")
            print("    --output my_results --debug")
            print("="*80)
            input("\nPress Enter to continue...")
        
        else:
            print("Invalid choice, please select 1-9")
        
        # Re-display configuration after changes
        if choice in ['2', '3', '4', '5', '6', '7']:
            return display_configuration_ui(config)

def main():
    parser = argparse.ArgumentParser(
        description='Post-process ResearchIR thermal data exports with spatial thermal coupling analysis'
    )
    
    parser.add_argument('--input', type=str, default='inputs/ResearchIR_Outputs_HBridge_15s',
                       help='Input folder with ResearchIR CSV/TXT exports')
    parser.add_argument('--output', type=str, default='outputs/thermal_analysis_results',
                       help='Output directory for results')
    parser.add_argument('--custom_groups', type=str,
                       help='JSON file with custom component grouping rules')
    parser.add_argument('--coordinates', type=str, default='inputs/hbridge_pcb_components_enhanced.csv',
                       help='CSV file with component X,Y coordinates in mm (default: inputs/hbridge_pcb_components_enhanced.csv)')
    parser.add_argument('--proximity_threshold', type=float, default=10.0,
                       help='Distance threshold in mm for thermal coupling analysis (default: 10.0)')
    parser.add_argument('--no_spatial', action='store_true',
                       help='Disable spatial thermal coupling analysis')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--no_ui', action='store_true',
                       help='Skip interactive configuration UI and run directly')
    
    args = parser.parse_args()
    
    # Build initial configuration dictionary
    config = {
        'input_folder': args.input,
        'output_dir': args.output,
        'custom_groups_file': args.custom_groups,
        'coordinates_file': args.coordinates if not args.no_spatial else None,
        'proximity_threshold': args.proximity_threshold,
        'spatial_enabled': not args.no_spatial,
        'debug': args.debug
    }
    
    # Resolve coordinates file path
    if config['spatial_enabled'] and config['coordinates_file']:
        if os.path.exists(config['coordinates_file']):
            pass  # Already absolute or relative path exists
        else:
            # Try relative to script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, config['coordinates_file'])
            if os.path.exists(alt_path):
                config['coordinates_file'] = alt_path
    
    # Show interactive UI unless --no_ui flag is set
    if not args.no_ui:
        config = display_configuration_ui(config)
    
    # Load custom grouping rules if provided
    custom_groups = None
    if config['custom_groups_file'] and os.path.exists(config['custom_groups_file']):
        try:
            with open(config['custom_groups_file'], 'r') as f:
                custom_groups = json.load(f)
            print(f"Loaded custom grouping rules from {config['custom_groups_file']}")
        except Exception as e:
            print(f"Error loading custom groups: {e}")
    
    # Determine final coordinates file (respect UI changes)
    coordinates_file = None
    if config['spatial_enabled'] and config['coordinates_file']:
        if os.path.exists(config['coordinates_file']):
            coordinates_file = config['coordinates_file']
        else:
            print(f"Warning: Coordinates file not found: {config['coordinates_file']}")
            print(f"   Spatial thermal coupling analysis will be skipped")
    
    # Process data
    processor = ResearchIRPostProcessor(
        debug=config['debug'],
        coordinates_file=coordinates_file,
        proximity_threshold=config['proximity_threshold']
    )
    
    try:
        results = processor.process_researchir_data(
            input_folder=config['input_folder'],
            output_dir=config['output_dir'],
            custom_groups=custom_groups
        )
        
        print(f"\nResults saved to: {config['output_dir']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        if config['debug']:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()