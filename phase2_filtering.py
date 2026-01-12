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
