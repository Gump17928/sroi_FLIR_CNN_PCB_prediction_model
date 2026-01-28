#!/usr/bin/env python3
"""
Manual Time Alignment Tool

Visually compare FLIR ROI temperatures vs sand thermistor data to manually
determine the correct time offset.

Usage:
    python manual_time_alignment.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_flir_roi_timeseries(flir_folder, pixel_map_csv, component_name):
    """
    Load FLIR timeseries for a specific component's ROI.
    
    Returns:
        times: Array of timestamps (seconds)
        temps: Array of mean ROI temperatures
    """
    # Load pixel map to get ROI
    pixel_df = pd.read_csv(pixel_map_csv)
    comp_row = pixel_df[pixel_df['component_name'] == component_name]
    
    if len(comp_row) == 0:
        raise ValueError(f"Component '{component_name}' not found in pixel map")
    
    pixel_list = eval(comp_row.iloc[0]['pixel_list'])
    
    if len(pixel_list) == 0:
        raise ValueError(f"Component '{component_name}' has no ROI pixels")
    
    # Load FLIR frames
    flir_files = sorted(Path(flir_folder).glob("*.csv"))
    
    times = []
    temps = []
    
    print(f"Loading {len(flir_files)} FLIR frames for component '{component_name}'...")
    
    for frame_index, frame_file in enumerate(flir_files):
        # Use FRAME INDEX for timestamp (same as flir_frame_loader)
        # Frame 0 = 0s, Frame 1 = 15s, Frame 2 = 30s, etc.
        time_s = frame_index * 15.0
        
        # Load frame
        try:
            frame = np.loadtxt(frame_file, delimiter=',', skiprows=5)
        except:
            continue
        
        # Extract ROI temperatures
        roi_temps = []
        for pixel_x, pixel_y in pixel_list:
            if 0 <= pixel_y < frame.shape[0] and 0 <= pixel_x < frame.shape[1]:
                roi_temps.append(frame[pixel_y, pixel_x])
        
        if roi_temps:
            times.append(time_s)
            temps.append(np.mean(roi_temps))
    
    return np.array(times), np.array(temps)


def load_sand_timeseries(thermistor_csv, component_name):
    """
    Load sand thermistor timeseries for a component.
    
    Returns:
        times: Array of timestamps (seconds)
        temps: Array of temperatures
    """
    df = pd.read_csv(thermistor_csv)
    
    if component_name not in df.columns:
        raise ValueError(f"Component '{component_name}' not found in thermistor CSV")
    
    times = df['Time (s)'].values
    temps = df[component_name].values
    
    return times, temps


def plot_alignment(flir_times, flir_temps, sand_times, sand_temps, 
                   offset=0, component_name="Component", zoom_minutes=10):
    """
    Plot FLIR vs sand temperatures with a time offset applied.
    
    Args:
        offset: Time offset in seconds (positive = FLIR leads sand)
        zoom_minutes: How many minutes of data to show
    """
    plt.figure(figsize=(14, 8))
    
    # Apply offset to FLIR times
    flir_times_adjusted = flir_times + offset
    
    # Convert to minutes for readability
    flir_mins = flir_times_adjusted / 60
    sand_mins = sand_times / 60
    
    # Determine zoom range
    zoom_end = zoom_minutes
    
    # Plot full timeseries
    plt.subplot(2, 1, 1)
    plt.plot(sand_mins, sand_temps, 'b-', alpha=0.6, label=f'Sand thermistor (36 hrs)', linewidth=1)
    plt.plot(flir_mins, flir_temps, 'r-', linewidth=2, label=f'FLIR ROI (70 min, offset={offset:.1f}s)')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'{component_name} - Full Timeseries Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, zoom_minutes * 2)
    
    # Plot zoomed section
    plt.subplot(2, 1, 2)
    
    # Filter data for zoom
    flir_mask = (flir_mins >= 0) & (flir_mins <= zoom_end)
    sand_mask = (sand_mins >= 0) & (sand_mins <= zoom_end)
    
    plt.plot(sand_mins[sand_mask], sand_temps[sand_mask], 'b.-', 
             label=f'Sand thermistor', linewidth=2, markersize=4)
    plt.plot(flir_mins[flir_mask], flir_temps[flir_mask], 'r.-', 
             label=f'FLIR ROI (offset={offset:.1f}s)', linewidth=2, markersize=6)
    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'{component_name} - First {zoom_minutes} Minutes (For Visual Alignment)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, zoom_end)
    
    # Add correlation info
    if len(flir_temps[flir_mask]) > 0 and len(sand_temps[sand_mask]) > 0:
        # Resample to common time grid for correlation
        common_times = np.arange(0, zoom_end * 60, 15)  # 15s intervals
        flir_interp = np.interp(common_times, flir_times_adjusted, flir_temps, 
                                left=np.nan, right=np.nan)
        sand_interp = np.interp(common_times, sand_times, sand_temps,
                               left=np.nan, right=np.nan)
        
        valid = ~np.isnan(flir_interp) & ~np.isnan(sand_interp)
        if np.sum(valid) > 2:
            corr = np.corrcoef(flir_interp[valid], sand_interp[valid])[0, 1]
            plt.text(0.02, 0.98, f'Correlation: {corr:.3f}', 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    project_root = Path(__file__).parent
    
    # Paths
    session_dir = project_root / "outputs" / "0115_1806_P1-7"
    flir_folder = project_root / "inputs" / "ResearchIR_Outputs_HBridge_15s_filtered"
    pixel_map_csv = session_dir / "HBridge_15s_roi_pixel_map.csv"
    thermistor_csv = session_dir / "HBridge_15s_thermistor_timeseries.csv"
    
    # Choose a component with good ROI coverage (pick from pixel map)
    # Must be in both pixel map AND thermistor CSV
    # Good options: PS3, DL10, U19, PS2
    component_name = "PS3"  # Can change this
    
    print("="*80)
    print("MANUAL TIME ALIGNMENT TOOL")
    print("="*80)
    print(f"\nComponent: {component_name}")
    print(f"FLIR folder: {flir_folder.name}")
    print(f"Thermistor CSV: {thermistor_csv.name}\n")
    
    # Load data
    print("Loading FLIR ROI timeseries...")
    flir_times, flir_temps = load_flir_roi_timeseries(flir_folder, pixel_map_csv, component_name)
    print(f"  ✅ Loaded {len(flir_times)} FLIR frames")
    print(f"  Time range: {flir_times[0]:.1f}s - {flir_times[-1]:.1f}s ({(flir_times[-1]-flir_times[0])/60:.1f} min)")
    print(f"  Temp range: {np.min(flir_temps):.1f}°C - {np.max(flir_temps):.1f}°C")
    
    print("\nLoading sand thermistor timeseries...")
    sand_times, sand_temps = load_sand_timeseries(thermistor_csv, component_name)
    print(f"  ✅ Loaded {len(sand_times)} thermistor samples")
    print(f"  Time range: {sand_times[0]:.1f}s - {sand_times[-1]:.1f}s ({(sand_times[-1]-sand_times[0])/3600:.1f} hrs)")
    print(f"  Temp range: {np.min(sand_temps):.1f}°C - {np.max(sand_temps):.1f}°C")
    
    # Interactive alignment
    print("\n" + "="*80)
    print("VISUAL ALIGNMENT")
    print("="*80)
    print("\nTry different offsets to align the curves.")
    print("Positive offset = FLIR happened BEFORE sand test started")
    print("Negative offset = FLIR happened AFTER sand test started")
    print("\nRecommended starting values to try:")
    print("  0s     - Tests started simultaneously")
    print("  +60s   - FLIR started 1 min before sand")
    print("  +300s  - FLIR started 5 min before sand")
    print("  -60s   - Sand started 1 min before FLIR")
    
    while True:
        print("\n" + "-"*80)
        offset_input = input("Enter time offset in seconds (or 'q' to quit): ").strip()
        
        if offset_input.lower() == 'q':
            print("Exiting.")
            break
        
        try:
            offset = float(offset_input)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        # Plot
        fig = plot_alignment(flir_times, flir_temps, sand_times, sand_temps,
                           offset=offset, component_name=component_name, zoom_minutes=15)
        
        plt.savefig(project_root / f"time_alignment_offset_{offset:.0f}s.png", dpi=150, bbox_inches='tight')
        print(f"  📊 Plot saved: time_alignment_offset_{offset:.0f}s.png")
        
        plt.show()
        
        # Ask if this looks good
        satisfied = input("Does this alignment look correct? (y/n): ").strip().lower()
        if satisfied == 'y':
            print(f"\n✅ Selected offset: {offset:.1f}s")
            print(f"\nTo use this offset in the dataset builder:")
            print(f"  1. Edit cnn_data_preprocessor.py")
            print(f"  2. In detect_time_offset(), set: self.time_offset = {offset:.1f}")
            print(f"  3. Or pass as parameter to build_dataset()")
            break
