"""
===============================================================================
PHASE 2: FILTERING
===============================================================================
Applies thermal filtering to remove camera refocusing artifacts and noise
while preserving authentic thermal transients.

This module provides:
- Multiple filtering algorithms (median, Savitzky-Golay, lowpass, outlier removal)
- Filter comparison analysis
- Temperature transient analysis with filtered data
===============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import shutil

# Import scipy for advanced filtering (optional)
try:
    from scipy import signal
    from scipy.ndimage import median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Advanced filtering disabled.")


def apply_thermal_filtering(temperatures: np.ndarray, times: np.ndarray,
                            filter_type: str = 'median', filter_params: Dict = None,
                            debug: bool = False) -> np.ndarray:
    """
    Apply filtering to remove camera refocusing artifacts while preserving thermal transients
    
    Camera refocusing during thermal tests creates artificial temperature spikes.
    This function removes these artifacts using various filtering methods.
    
    Available filters:
    - 'savgol': Savitzky-Golay filter (preserves peaks, removes noise)
    - 'median': Median filter (removes spikes, good for refocus artifacts) [DEFAULT]
    - 'lowpass': Butterworth low-pass filter (smooth overall trends)
    - 'outlier': Statistical outlier removal + interpolation
    - 'hybrid': Combination of outlier removal + Savitzky-Golay
    - 'none': No filtering
    
    Args:
        temperatures: Array of temperature values (Celsius)
        times: Array of time values (seconds)
        filter_type: Type of filter to apply
        filter_params: Optional dict of filter-specific parameters
        debug: Enable verbose output
    
    Returns:
        Filtered temperature array (same length as input)
    """
    
    if filter_type == 'none' or not SCIPY_AVAILABLE or len(temperatures) < 5:
        return temperatures
    
    if filter_params is None:
        filter_params = {}
    
    filtered_temps = temperatures.copy()
    
    try:
        if filter_type == 'savgol':
            # Savitzky-Golay filter - excellent for preserving peaks while smoothing
            window_length = filter_params.get('window_length', min(11, len(temperatures)//3))
            if window_length % 2 == 0:
                window_length += 1  # Must be odd
            window_length = max(5, min(window_length, len(temperatures)))
            polyorder = filter_params.get('polyorder', min(3, window_length-1))
            
            filtered_temps = signal.savgol_filter(temperatures, window_length, polyorder)
            
        elif filter_type == 'median':
            # Median filter - great for removing sudden spikes from refocusing
            kernel_size = filter_params.get('kernel_size', 5)
            filtered_temps = median_filter(temperatures, size=kernel_size)
            
        elif filter_type == 'lowpass':
            # Low-pass Butterworth filter
            cutoff_freq = filter_params.get('cutoff_freq', 0.1)  # Normalized frequency
            order = filter_params.get('order', 4)
            
            b, a = signal.butter(order, cutoff_freq, btype='low')
            filtered_temps = signal.filtfilt(b, a, temperatures)
            
        elif filter_type == 'outlier':
            # Statistical outlier removal with interpolation
            std_threshold = filter_params.get('std_threshold', 2.5)
            
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
                
        elif filter_type == 'hybrid':
            # Hybrid approach: outlier removal followed by Savitzky-Golay
            # First pass: remove outliers
            intermediate_temps = apply_thermal_filtering(temperatures, times,
                                                       filter_type='outlier',
                                                       filter_params=filter_params,
                                                       debug=False)
            
            # Second pass: smooth with Savitzky-Golay
            window_length = filter_params.get('window_length', min(9, len(temperatures)//3))
            if window_length % 2 == 0:
                window_length += 1
            window_length = max(5, min(window_length, len(temperatures)))
            polyorder = filter_params.get('polyorder', min(2, window_length-1))
            
            filtered_temps = signal.savgol_filter(intermediate_temps, window_length, polyorder)
        
        if debug:
            print(f"Applied {filter_type} filter to remove refocusing artifacts")
            
    except Exception as e:
        if debug:
            print(f"Filtering failed: {e}, using original data")
        filtered_temps = temperatures
        
    return filtered_temps


def analyze_temperature_transients(component_data: Dict, filter_type: str = 'median',
                                  filter_params: Dict = None, debug: bool = False) -> Dict:
    """
    Analyze temperature transients for each component using filtered data
    
    Calculates key thermal characteristics:
    - Initial/final temperatures
    - Temperature rise (delta)
    - Peak temperature and timing
    - Mean and standard deviation
    - Maximum rate of temperature change
    
    Args:
        component_data: Dictionary of component DataFrames from Phase 1
        filter_type: Type of filter to apply before analysis
        filter_params: Optional filter parameters
        debug: Enable verbose output
    
    Returns:
        Dictionary mapping component names to analysis results with keys:
        - initial_temp, final_temp, delta_temp
        - max_temp, min_temp, mean_temp, std_temp
        - peak_time, max_rate, temp_range, data_points
    """
    
    analysis_results = {}
    
    for component_name, df in component_data.items():
        if len(df) < 2:
            continue
        
        temps = df['Temperature'].values
        times = df['Time'].values
        
        # Apply filtering to remove camera refocusing artifacts
        filtered_temps = apply_thermal_filtering(temps, times, filter_type,
                                               filter_params, debug)
        
        # Calculate transient characteristics using filtered data
        initial_temp = filtered_temps[0]
        final_temp = filtered_temps[-1]
        delta_temp = final_temp - initial_temp
        max_temp = np.max(filtered_temps)
        min_temp = np.min(filtered_temps)
        mean_temp = np.mean(filtered_temps)
        std_temp = np.std(filtered_temps)
        
        # Find peak time
        max_idx = np.argmax(filtered_temps)
        peak_time = times[max_idx] if max_idx < len(times) else times[-1]
        
        # Calculate rate of change (temperature gradient)
        if len(filtered_temps) > 1:
            temp_gradient = np.gradient(filtered_temps, times)
            max_rate = np.max(np.abs(temp_gradient))
        else:
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


def apply_filtering_to_component_data(component_data: Dict, filter_type: str = 'median',
                                     filter_params: Dict = None, debug: bool = False) -> Dict:
    """
    Apply filtering to all component temperature data for plotting
    
    Creates filtered copies of all component DataFrames. Original data is preserved.
    
    Args:
        component_data: Dictionary of component DataFrames from Phase 1
        filter_type: Type of filter to apply
        filter_params: Optional filter parameters
        debug: Enable verbose output
    
    Returns:
        Dictionary of filtered component DataFrames (same structure as input)
    """
    
    if filter_type == 'none':
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
        filtered_temps = apply_thermal_filtering(temps, times, filter_type,
                                               filter_params, debug)
        
        # Update dataframe with filtered temperatures
        df_copy['Temperature'] = filtered_temps
        
        # Store both original and filtered for comparison if debug mode
        if debug:
            df_copy['Temperature_Original'] = temps
            df_copy['Temperature_Filtered'] = filtered_temps
        
        filtered_component_data[component_name] = df_copy
        
    return filtered_component_data


def compare_filter_methods(component_data: Dict, component_name: str,
                          methods: list = None) -> Dict:
    """
    Compare multiple filtering methods on a single component
    
    Useful for determining the best filter for your thermal data.
    
    Args:
        component_data: Dictionary of component DataFrames
        component_name: Component to analyze
        methods: List of filter types to compare (default: all)
    
    Returns:
        Dictionary mapping filter names to filtered temperature arrays
    """
    
    if methods is None:
        methods = ['none', 'median', 'savgol', 'lowpass', 'outlier', 'hybrid']
    
    if component_name not in component_data:
        raise ValueError(f"Component {component_name} not found in data")
    
    df = component_data[component_name]
    temps = df['Temperature'].values
    times = df['Time'].values
    
    comparison_results = {}
    
    for method in methods:
        if method == 'none':
            comparison_results[method] = temps
        else:
            filtered = apply_thermal_filtering(temps, times, filter_type=method)
            comparison_results[method] = filtered
    
    return comparison_results


def filter_flir_frames_for_ml(
    input_folder: Path,
    output_folder: Path,
    kernel_size: int = 5,
    verbose: bool = True
) -> None:
    """
    Filter full FLIR frame sequences for ML training.
    
    Applies temporal median filter to each pixel across all frames.
    This removes camera refocusing artifacts while preserving spatial information
    needed for U-Net CNN training.
    
    Args:
        input_folder: Path to ResearchIR_Outputs folder (e.g., ResearchIR_Outputs_HBridge_15s)
        output_folder: Path to filtered outputs (e.g., ResearchIR_Outputs_HBridge_15s_filtered)
        kernel_size: Median filter window size (default 5, matches component filtering)
        verbose: Print progress updates
    
    Processing:
        - Loads all frames into memory (~350MB for 300 frames)
        - Filters each pixel's time series independently
        - Writes filtered frames to new CSV files
        - Copies metadata files unchanged
    
    Time estimate: ~4 minutes per board (300 frames × 480×640 pixels)
    Disk space: ~7MB per board
    
    Example:
        filter_flir_frames_for_ml(
            Path("inputs/ResearchIR_Outputs_HBridge_15s"),
            Path("inputs/ResearchIR_Outputs_HBridge_15s_filtered"),
            kernel_size=5
        )
    """
    if not SCIPY_AVAILABLE:
        print("ERROR: scipy not available. Cannot filter FLIR frames.")
        return
    
    # Convert to Path objects
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files (FLIR frames)
    csv_files = sorted(input_folder.glob("*.csv"))
    n_frames = len(csv_files)
    
    if n_frames == 0:
        print(f"ERROR: No CSV files found in {input_folder}")
        return
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"FILTERING FULL FLIR FRAMES FOR ML TRAINING")
        print(f"{'='*80}")
        print(f"Input:  {input_folder}")
        print(f"Output: {output_folder}")
        print(f"Frames: {n_frames}")
        print(f"Filter: median (kernel_size={kernel_size})")
        print(f"\nStep 1: Loading frames into memory...")
    
    # Load all frames into memory (480×640×300 ≈ 350MB for float32)
    frames_list = []
    for i, csv_file in enumerate(csv_files):
        try:
            # ResearchIR CSVs have 5-line header, then 480 rows × 640 columns
            # Skip first 5 lines (metadata header)
            df = pd.read_csv(csv_file, header=None, skiprows=5)
            frames_list.append(df.values)
            
            if verbose and (i+1) % 50 == 0:
                print(f"  Loaded {i+1}/{n_frames} frames...")
        except Exception as e:
            print(f"ERROR loading {csv_file.name}: {e}")
            return
    
    # Convert to numpy array: shape (n_frames, height, width)
    frames = np.array(frames_list, dtype=np.float32)
    height, width = frames.shape[1], frames.shape[2]
    
    if verbose:
        print(f"  ✓ Loaded {n_frames} frames")
        print(f"  Frame dimensions: {height} × {width}")
        print(f"  Total pixels: {height * width:,}")
        print(f"  Memory usage: ~{frames.nbytes / 1e6:.1f} MB")
        print(f"\nStep 2: Filtering pixels (this may take a few minutes)...")
    
    # Filter each pixel's time series
    filtered_frames = np.zeros_like(frames)
    total_pixels = height * width
    pixels_processed = 0
    
    for i in range(height):
        for j in range(width):
            # Extract time series for this pixel across all frames
            pixel_temps = frames[:, i, j]
            
            # Apply median filter to remove temporal spikes
            filtered_temps = median_filter(pixel_temps, size=kernel_size, mode='nearest')
            
            # Store filtered values
            filtered_frames[:, i, j] = filtered_temps
            pixels_processed += 1
        
        # Progress update every 50 rows
        if verbose and (i+1) % 50 == 0:
            progress = pixels_processed / total_pixels * 100
            print(f"  Row {i+1}/{height} ({progress:.1f}% complete)")
    
    if verbose:
        print(f"  ✓ Filtered all {total_pixels:,} pixels")
        print(f"\nStep 3: Writing filtered frames to CSV...")
    
    # Write filtered frames to CSV (same format as input: 5-line header + data)
    for i, csv_file in enumerate(csv_files):
        output_csv = output_folder / csv_file.name
        
        try:
            # Read original header lines (first 5 lines)
            with open(csv_file, 'r') as f:
                header_lines = [f.readline() for _ in range(5)]
            
            # Write header + filtered data
            with open(output_csv, 'w') as f:
                # Write header
                f.writelines(header_lines)
                
                # Write filtered frame data (480 rows × 640 columns)
                df_filtered = pd.DataFrame(filtered_frames[i])
                df_filtered.to_csv(f, header=False, index=False)
            
            if verbose and (i+1) % 50 == 0:
                print(f"  Wrote {i+1}/{n_frames} files...")
        except Exception as e:
            print(f"ERROR writing {output_csv.name}: {e}")
            return
    
    if verbose:
        print(f"  ✓ Wrote all {n_frames} CSV files")
        print(f"\nStep 4: Copying metadata files...")
    
    # Copy .txt metadata files (small, don't filter)
    txt_files = list(input_folder.glob("*.txt"))
    for txt_file in txt_files:
        try:
            shutil.copy2(txt_file, output_folder / txt_file.name)
        except Exception as e:
            print(f"WARNING: Could not copy {txt_file.name}: {e}")
    
    if verbose:
        print(f"  ✓ Copied {len(txt_files)} metadata file(s)")
        print(f"\n{'='*80}")
        print(f"✓ FILTERING COMPLETE!")
        print(f"{'='*80}")
        print(f"Filtered frames: {output_folder}")
        print(f"Total files: {n_frames} CSVs + {len(txt_files)} TXT files")
        print(f"{'='*80}\n")


def export_filtered_temperatures(filtered_data: Dict, output_dir: Path, 
                                 board_name: str = "HBridge") -> Path:
    """
    Export filtered component temperatures to CSV for standalone analysis.
    
    Args:
        filtered_data: Dictionary of component DataFrames (from apply_filtering_to_component_data)
        output_dir: Directory to save CSV (e.g., outputs/TIMESTAMP/)
        board_name: Board identifier for filename
    
    Returns:
        Path to exported CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{board_name}_phase2_filtered_temperatures.csv"
    
    # Build DataFrame from filtered component data using pd.concat for better performance
    data_dict = {}
    
    # Extract time from first component (all should have same time array)
    if filtered_data:
        first_component = next(iter(filtered_data.values()))
        data_dict['time_s'] = first_component['Time'].values
    
    # Add each component's filtered temperatures
    for component_name, df in filtered_data.items():
        data_dict[component_name] = df['Temperature'].values
    
    # Create DataFrame from dictionary all at once
    result_df = pd.DataFrame(data_dict)
    
    # Export to CSV
    result_df.to_csv(csv_path, index=False, float_format='%.3f')
    
    print(f"  ✓ Exported filtered temperatures: {csv_path.name}")
    print(f"    Shape: {len(result_df)} samples × {len(result_df.columns)-1} components")
    
    return csv_path


def export_filter_comparison(component_data: Dict, filter_type: str,
                             output_dir: Path, board_name: str = "HBridge",
                             max_components: int = 5) -> Path:
    """
    Export filter comparison data for visualization (optional - for debugging).
    
    Exports raw vs filtered data for first few components to enable
    standalone viz_phase2_filtering.py plotting.
    
    Args:
        component_data: Dictionary of component DataFrames (raw data)
        filter_type: Filter type used ('median', 'savgol', etc.)
        output_dir: Directory to save CSV
        board_name: Board identifier
        max_components: Maximum components to export (default: 5 to limit file size)
    
    Returns:
        Path to exported CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{board_name}_phase2_filter_comparison.csv"
    
    rows = []
    
    # Get first max_components from the dictionary
    component_names = list(component_data.keys())[:max_components]
    
    for component in component_names:
        df = component_data[component]
        times = df['Time'].values
        raw_temps = df['Temperature'].values
        
        # Apply filtering
        filtered_temps = apply_thermal_filtering(
            raw_temps, times, filter_type=filter_type
        )
        
        # Build rows for this component
        for t_idx, time_val in enumerate(times):
            rows.append({
                'time_s': time_val,
                'component': component,
                'raw': raw_temps[t_idx],
                'filtered': filtered_temps[t_idx],
                'filter_type': filter_type
            })
    
    # Export to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format='%.3f')
    
    print(f"  ✓ Exported filter comparison: {csv_path.name}")
    print(f"    Components: {len(component_names)}, Samples: {len(times)}")
    
    return csv_path
