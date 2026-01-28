"""
===============================================================================
CNN DATA PREPROCESSOR - Training Dataset Builder for Spatial Thermal CNN
===============================================================================
Preprocess and align FLIR thermal frames with thermistor ground truth data.

Purpose:
    - Load FLIR frame sequences (from flir_frame_loader)
    - Load thermistor time series (from Phase 6 calibration database)
    - Load ROI pixel maps (from roi_pixel_mapper)
    - Detect temporal offset via cross-correlation
    - Interpolate thermistor data to FLIR timestamps
    - Build ROI masks for supervised learning
    - Split components into train/validation sets (80/20)
    - Export HDF5 dataset for CNN training

HDF5 Dataset Structure:
    /flir_frames: [n_frames, H, W] - Input thermal images 15s
    /sand_temps: [n_frames, n_components] - Ground truth thermistor temps 0.5s
    /roi_masks: [n_components, H, W] - Binary masks for each ROI
    /metadata:
        - Timestamps, component names, alignment info
        - train_component_indices: Indices of training components (80%)
        - val_component_indices: Indices of validation components (20%)
        - component_val_split: Fraction used for validation

Component-Level Split (NEW):
    - Random 80/20 split of components (seed=42 for reproducibility)
    - Training: 80% of components have thermistor data in input Channel 1
    - Validation: Same 80% in input, but predict held-out 20% of components
    - Tests if model learns thermal coupling patterns vs copying inputs

Workflow:
    1. Load FLIR frames (640x480, 15s intervals)
    2. Load thermistor CSV (Phase 6 calibration data)
    3. Cross-correlate to find time offset
    4. Padding thermistor data with average steady state values
    5. Generate ROI masks from pixel maps
    6. Split components randomly (80/20)
    7. Export aligned HDF5 dataset with split metadata

Created: January 12, 2026
Updated: January 23, 2026 - Component-level validation split
===============================================================================
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from scipy import signal
from scipy.interpolate import interp1d
from flir_frame_loader import FLIRFrameLoader
from loader_thermistor import ThermalDataLoader, SteadyStateAnalyzer


class CNNDataPreprocessor:
    """
    Preprocessor for building CNN training datasets from FLIR and thermistor data.
    
    Handles temporal alignment, ROI mask generation, and HDF5 export.
    """
    
    def __init__(self, image_width: int = 640, image_height: int = 480, verbose: bool = True):
        """
        Initialize CNN data preprocessor.
        
        Args:
            image_width: FLIR image width (pixels)
            image_height: FLIR image height (pixels)
            verbose: Enable verbose output
        """
        self.image_width = image_width
        self.image_height = image_height
        self.verbose = verbose
        
        # Data containers
        self.flir_frames = None
        self.flir_timestamps = None
        self.thermistor_data = None
        self.roi_pixel_map = None
        self.roi_masks = None
        self.frame_indices = None  # Mapping from timestamp index to frame index (for extended data)
        
        # Alignment results
        self.time_offset = 0.0
        self.aligned_sand_temps = None
    
    def load_flir_frames(self, flir_folder: str, max_frames: Optional[int] = None):
        """
        Load FLIR thermal frame sequence.
        
        Args:
            flir_folder: Path to ResearchIR exports or .npz file
            max_frames: Maximum frames to load (None = all)
        """
        if self.verbose:
            print("\n" + "="*80)
            print("  STEP 1: LOADING FLIR FRAMES")
            print("="*80)
        
        loader = FLIRFrameLoader(verbose=self.verbose)
        
        # Check if .npz file (pre-exported) or folder
        flir_path = Path(flir_folder)
        if flir_path.suffix == '.npz':
            loader.load_from_npz(str(flir_path))
        else:
            loader.load_sequence(str(flir_path), max_frames=max_frames)
        
        self.flir_frames = loader.frames
        self.flir_timestamps = loader.timestamps
        
        if self.verbose:
            print(f"  ✓ FLIR frames loaded: {self.flir_frames.shape}")
            print(f"  ✓ Time range: {self.flir_timestamps[0]:.1f}s - {self.flir_timestamps[-1]:.1f}s\n")
    
    def _save_flir_frame_as_csv(self, frame: np.ndarray, output_path: Path):
        """
        Save a FLIR frame to CSV in ResearchIR export format.
        
        Format:
        - 5 header lines (metadata)
        - Followed by temperature matrix (one row per image row)
        
        Args:
            frame: NumPy array [H, W] with thermal data
            output_path: Path to save CSV file
        """
        with open(output_path, 'w') as f:
            # Write 5-line header (ResearchIR format)
            f.write("Filename = STEADY_STATE_AVERAGED\\n")
            f.write("Units = Temperature (C)\\n")
            f.write("Time = N/A (Averaged Frame)\\n")
            f.write("FrameNumber = AVERAGED\\n")
            f.write("Preset = Averaged from last frames\\n")
            f.write("\\n")
            
            # Write temperature matrix
            for row in frame:
                row_str = ','.join([f'{val:.6e}' for val in row])
                f.write(row_str + '\\n')
    
    def load_thermistor_data(self, thermistor_csv: str, time_column: str = 'Time (s)', 
                            component_prefix: str = 'Temp_'):
        """
        Load thermistor ground truth data from Phase 6 calibration.
        
        Args:
            thermistor_csv: Path to thermistor time series CSV
            time_column: Name of time column
            component_prefix: Prefix for component temperature columns
        """
        if self.verbose:
            print("="*80)
            print("  STEP 2: LOADING THERMISTOR DATA")
            print("="*80)
        
        # Store CSV path for debug reporting
        self.thermistor_csv_path = Path(thermistor_csv)
        print(f"  📂 LOADING FROM: {self.thermistor_csv_path.absolute()}")
        
        df = pd.read_csv(thermistor_csv)
        print(f"  📊 CSV Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  📋 Columns: {df.columns.tolist()}")
        
        # Extract time and temperature columns
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found in {thermistor_csv}")
        
        # Find all temperature columns (exclude time column)
        temp_cols = [col for col in df.columns if col.startswith(component_prefix) and col != time_column]
        
        if not temp_cols:
            raise ValueError(f"No temperature columns found with prefix '{component_prefix}'")
        
        self.thermistor_data = {
            'time': df[time_column].values,
            'components': {col.replace(component_prefix, ''): df[col].values for col in temp_cols},
            'component_names': [col.replace(component_prefix, '') for col in temp_cols]
        }
        
        if self.verbose:
            print(f"  ✓ Loaded thermistor data")
            print(f"  Components: {len(self.thermistor_data['component_names'])}")
            print(f"  Component list: {', '.join(self.thermistor_data['component_names'])}")
            print(f"  Time range: {self.thermistor_data['time'][0]:.1f}s - {self.thermistor_data['time'][-1]:.1f}s")
            print(f"  Sample rate: {len(self.thermistor_data['time'])} samples\n")
    
    def load_roi_pixel_map(self, pixel_map_csv: str):
        """
        Load ROI pixel map from roi_pixel_mapper output.
        
        Args:
            pixel_map_csv: Path to pixel map CSV
        """
        if self.verbose:
            print("="*80)
            print("  STEP 3: LOADING ROI PIXEL MAP")
            print("="*80)
        
        df = pd.read_csv(pixel_map_csv)
        
        # Parse pixel_list column (stored as string representation of list)
        df['pixel_list_parsed'] = df['pixel_list'].apply(eval)
        
        self.roi_pixel_map = df
        
        if self.verbose:
            print(f"  ✓ Loaded pixel map")
            print(f"  Components: {len(df)}")
            print(f"  Total pixels across all ROIs: {df['pixel_count'].sum()}\n")
    
    def detect_time_offset(self, reference_component: Optional[str] = None, manual_offset: Optional[float] = None) -> float:
        """
        Detect time offset between FLIR and thermistor via cross-correlation.
        
        Uses mean FLIR temperature vs thermistor temperature to find optimal offset.
        Resamples both to common time grid before correlation to handle different sampling rates.
        
        Args:
            reference_component: Specific component to use (None = use mean of all)
            manual_offset: If provided, skip cross-correlation and use this offset (seconds)
                          Positive = FLIR leads thermistor (FLIR started earlier)
        
        Returns:
            Time offset in seconds (positive = FLIR leads thermistor)
        """
        if self.verbose:
            print("="*80)
            print("  STEP 4: DETECTING TIME OFFSET (CROSS-CORRELATION)")
            print("="*80)
        
        # Use manual offset if provided
        if manual_offset is not None:
            self.time_offset = manual_offset
            if self.verbose:
                print(f"  ✓ Using MANUAL offset: {self.time_offset:.1f}s")
                print(f"  Interpretation: FLIR {'leads' if self.time_offset > 0 else 'lags'} thermistor by {abs(self.time_offset):.1f}s\n")
            return self.time_offset
        flir_mean_temps = np.mean(self.flir_frames, axis=(1, 2))
        
        # Get thermistor reference signal
        if reference_component:
            if reference_component not in self.thermistor_data['components']:
                raise ValueError(f"Component '{reference_component}' not found in thermistor data")
            thermistor_signal_full = self.thermistor_data['components'][reference_component]
        else:
            # Use mean of all thermistor components
            all_temps = np.array([temps for temps in self.thermistor_data['components'].values()])
            thermistor_signal_full = np.mean(all_temps, axis=0)
        
        thermistor_time_full = self.thermistor_data['time']
        
        # Resample thermistor to match FLIR sampling rate for fair comparison
        # Use a common time grid that covers the overlap region
        flir_start = self.flir_timestamps[0]
        flir_end = self.flir_timestamps[-1]
        therm_start = thermistor_time_full[0]
        therm_end = thermistor_time_full[-1]
        
        # Create search window - allow offset up to ±1 hour
        search_window = 3600  # seconds
        search_start = max(flir_start - search_window, therm_start)
        search_end = min(flir_end + search_window, therm_end)
        
        # Resample both to common 1 Hz grid for correlation
        common_dt = 1.0  # 1 Hz sampling for correlation
        common_times = np.arange(search_start, search_end, common_dt)
        
        # Interpolate FLIR to common grid
        flir_interp = interp1d(self.flir_timestamps, flir_mean_temps, 
                              kind='linear', bounds_error=False, fill_value=np.nan)
        flir_resampled = flir_interp(common_times)
        
        # Interpolate thermistor to common grid
        therm_interp = interp1d(thermistor_time_full, thermistor_signal_full,
                               kind='linear', bounds_error=False, fill_value=np.nan)
        therm_resampled = therm_interp(common_times)
        
        # Remove NaN values (edges where extrapolation not valid)
        valid_mask = ~(np.isnan(flir_resampled) | np.isnan(therm_resampled))
        if np.sum(valid_mask) < 10:
            print("  Warning: Insufficient overlap between FLIR and thermistor data")
            print("  Using zero offset (no alignment)")
            self.time_offset = 0.0
            return self.time_offset
        
        flir_valid = flir_resampled[valid_mask]
        therm_valid = therm_resampled[valid_mask]
        
        # Cross-correlate on resampled data (normalize by removing mean)

        # Check?
        correlation = signal.correlate(therm_valid - np.mean(therm_valid), 
                                      flir_valid - np.mean(flir_valid), 
                                      mode='full')
        lags = signal.correlation_lags(len(therm_valid), len(flir_valid), mode='full')
        
        # Find peak correlation
        peak_idx = np.argmax(correlation)
        lag_samples = lags[peak_idx]
        
        # Convert lag to time offset
        # Positive lag = FLIR leads thermistor (FLIR happened earlier)
        self.time_offset = lag_samples * common_dt
        
        if self.verbose:
            print(f"  ✓ Cross-correlation complete")
            print(f"  Peak correlation: {correlation[peak_idx]:.0f}")
            print(f"  Time offset: {self.time_offset:.1f}s")
            print(f"  Interpretation: FLIR {'leads' if self.time_offset > 0 else 'lags'} thermistor by {abs(self.time_offset):.1f}s")
            print(f"  FLIR time range: {flir_start:.1f}s - {flir_end:.1f}s")
            print(f"  Thermistor time range: {therm_start:.1f}s - {therm_end:.1f}s\n")
        
        return self.time_offset
    
    def align_and_interpolate(self):
        """
        Align thermistor data to FLIR timestamps using detected offset and interpolation.
        
        Creates aligned_sand_temps: [n_frames, n_components]
        Also tracks outliers for debug reporting.
        
        IMPORTANT: Thermistor data may span longer time than FLIR (e.g., 36 hrs vs 70 min).
        We interpolate thermistor to FLIR timestamps for training, using only the overlapping period.
        The full thermistor dataset is preserved in self.thermistor_data for prediction phase.
        """
        if self.verbose:
            print("="*80)
            print("  STEP 5: ALIGNING AND INTERPOLATING DATA")
            print("="*80)
        
        # Adjust FLIR timestamps by offset
        adjusted_flir_times = self.flir_timestamps + self.time_offset
        
        # Find overlapping time range
        flir_start = adjusted_flir_times[0]
        flir_end = adjusted_flir_times[-1]
        therm_start = self.thermistor_data['time'][0]
        therm_end = self.thermistor_data['time'][-1]
        
        overlap_start = max(flir_start, therm_start)
        overlap_end = min(flir_end, therm_end)
        
        if self.verbose:
            print(f"  Time ranges:")
            print(f"    FLIR (adjusted): {flir_start:.1f}s - {flir_end:.1f}s ({(flir_end-flir_start)/60:.1f} min)")
            print(f"    Thermistor: {therm_start:.1f}s - {therm_end:.1f}s ({(therm_end-therm_start)/3600:.1f} hrs)")
            print(f"    Overlap: {overlap_start:.1f}s - {overlap_end:.1f}s ({(overlap_end-overlap_start)/60:.1f} min)")
            
            if overlap_end <= overlap_start:
                print(f"  ⚠️  WARNING: No temporal overlap! Check time offset.")
            elif (overlap_end - overlap_start) < 60:
                print(f"  ⚠️  WARNING: Very short overlap (<1 min)")
            else:
                print(f"  ✅ Good overlap for training")
        
        n_frames = len(adjusted_flir_times)
        n_components = len(self.thermistor_data['component_names'])
        
        self.aligned_sand_temps = np.zeros((n_frames, n_components), dtype=np.float32)
        
        # Initialize outlier tracking list
        self.outlier_records = []
        
        # Interpolate each component with forward-fill for out-of-range values
        for idx, comp_name in enumerate(self.thermistor_data['component_names']):
            thermistor_temps = self.thermistor_data['components'][comp_name]
            thermistor_times = self.thermistor_data['time']
            
            # For each adjusted FLIR timestamp, find the corresponding thermistor value
            interpolated_temps = np.zeros(n_frames, dtype=np.float32)
            
            for i, adj_time in enumerate(adjusted_flir_times):
                if adj_time < thermistor_times[0]:
                    # Before thermistor data starts - use first value (backward fill)
                    interpolated_temps[i] = thermistor_temps[0]
                elif adj_time > thermistor_times[-1]:
                    # After thermistor data ends - use last value (forward fill)
                    interpolated_temps[i] = thermistor_temps[-1]
                else:
                    # Within range - use linear interpolation
                    idx_after = np.searchsorted(thermistor_times, adj_time)
                    if idx_after == 0:
                        interpolated_temps[i] = thermistor_temps[0]
                    elif idx_after >= len(thermistor_times):
                        interpolated_temps[i] = thermistor_temps[-1]
                    else:
                        # Linear interpolation between two points
                        t0, t1 = thermistor_times[idx_after-1], thermistor_times[idx_after]
                        v0, v1 = thermistor_temps[idx_after-1], thermistor_temps[idx_after]
                        alpha = (adj_time - t0) / (t1 - t0) if (t1 - t0) > 0 else 0
                        interpolated_temps[i] = v0 + alpha * (v1 - v0)
            
            self.aligned_sand_temps[:, idx] = interpolated_temps
            
            # DEBUG: Check interpolated values (align phase)
            if idx == 0:  # Only print for first component
                print(f"\n  🔍 DEBUG - Component '{comp_name}' interpolation:")
                print(f"     Source range: [{thermistor_temps.min():.2f}, {thermistor_temps.max():.2f}]°C")
                print(f"     Interpolated range: [{interpolated_temps.min():.2f}, {interpolated_temps.max():.2f}]°C")
                print(f"     Time range: [{thermistor_times[0]:.1f}, {thermistor_times[-1]:.1f}]s")
                print(f"     Adjusted FLIR times: [{adjusted_flir_times[0]:.1f}, {adjusted_flir_times[-1]:.1f}]s")
            
            # Track outliers in ORIGINAL full dataset (for awareness)
            outlier_mask = (thermistor_temps < -100) | (thermistor_temps > 300)
            if np.any(outlier_mask):
                outlier_count = np.sum(outlier_mask)
                min_temp = np.min(thermistor_temps)
                max_temp = np.max(thermistor_temps)
                
                self.outlier_records.append({
                    'thermistor_file': str(self.thermistor_csv_path),
                    'component_name': comp_name,
                    'min_temp': min_temp,
                    'max_temp': max_temp,
                    'outlier_count': outlier_count,
                    'total_samples': len(thermistor_temps),
                    'outlier_percentage': 100 * outlier_count / len(thermistor_temps),
                    'note': 'Outliers in full dataset - training uses overlapping time only'
                })
        
        if self.verbose:
            print(f"  ✓ Interpolation complete")
            print(f"  Aligned temperatures: {self.aligned_sand_temps.shape}")
            print(f"  Temperature range (training data): {np.min(self.aligned_sand_temps):.1f}°C - {np.max(self.aligned_sand_temps):.1f}°C")
            if self.outlier_records:
                print(f"  ℹ️  Note: {len(self.outlier_records)} components have outliers in FULL dataset")
                print(f"      (Training uses only overlapping {(overlap_end-overlap_start)/60:.1f} min)")
            print()
        
        # Store overlap info for later use in prediction
        self.overlap_info = {
            'overlap_start': overlap_start,
            'overlap_end': overlap_end,
            'flir_duration_min': (flir_end - flir_start) / 60,
            'thermistor_duration_hrs': (therm_end - therm_start) / 3600,
            'overlap_duration_min': (overlap_end - overlap_start) / 60
        }
    
    def extend_flir_to_match_thermistor(self, flir_folder: str, steady_state_window: int = 5):
        """
        Extend FLIR data by adding steady-state reference frame to match thermistor duration.
        
        Strategy (OPTIMIZED):
        1. Calculate steady-state FLIR frame (average of last N frames)
        2. Save steady-state frame as CSV in FLIR folder (for reference/debugging)
        3. Store metadata indicating which time indices should use steady-state frame
        4. Interpolate thermistor data to extended timeline without duplicating FLIR data
        
        This approach saves 95% storage vs duplicating frames thousands of times.
        
        Args:
            flir_folder: Path to FLIR folder (to save steady-state CSV)
            steady_state_window: Number of frames to average for steady-state (default: 5)
        """
        if self.verbose:
            print("="*80)
            print("  EXTENDING FLIR DATA TO MATCH THERMISTOR DURATION")
            print("="*80)
        
        # Get original dimensions
        original_flir_end = self.flir_timestamps[-1]
        thermistor_end = self.thermistor_data['time'][-1]
        flir_interval = np.mean(np.diff(self.flir_timestamps))  # ~15s
        
        if thermistor_end <= original_flir_end:
            if self.verbose:
                print(f"  ℹ️  Thermistor duration ({thermistor_end/3600:.1f}h) ≤ FLIR duration ({original_flir_end/3600:.1f}h)")
                print(f"  No extension needed.\n")
            return
        
        # Calculate steady-state FLIR frame (average of last N frames)
        steady_state_frame = np.mean(self.flir_frames[-steady_state_window:], axis=0)
        
        if self.verbose:
            print(f"  Original FLIR duration: {original_flir_end/3600:.2f}h ({len(self.flir_timestamps)} frames)")
            print(f"  Thermistor duration: {thermistor_end/3600:.2f}h")
            print(f"  Steady-state frame: average of last {steady_state_window} frames")
            print(f"  Steady-state temp range: {steady_state_frame.min():.1f}°C - {steady_state_frame.max():.1f}°C")
        
        # Save steady-state frame as CSV (same format as ResearchIR exports)
        flir_path = Path(flir_folder)
        steady_state_csv = flir_path / "Steady_State_AVERAGED.csv"
        self._save_flir_frame_as_csv(steady_state_frame, steady_state_csv)
        
        if self.verbose:
            print(f"  ✓ Saved steady-state frame: {steady_state_csv.name}")
        
        # Generate extended timestamps (but DON'T duplicate frames)
        n_new_frames = int((thermistor_end - original_flir_end) / flir_interval)
        new_timestamps = original_flir_end + flir_interval * np.arange(1, n_new_frames + 1)
        
        # Store steady-state frame as SINGLE frame (don't duplicate)
        # Add it to the end of the frame stack
        extended_flir_frames = np.vstack([
            self.flir_frames,
            steady_state_frame[np.newaxis, :, :]  # Just ONE copy
        ])
        
        # Extend timestamps
        extended_timestamps = np.concatenate([self.flir_timestamps, new_timestamps])
        
        # Create index mapping: which FLIR frame index to use for each timestamp
        # For original frames: use their actual index
        # For extended frames: use the steady-state frame index (last one)
        n_original = len(self.flir_timestamps)
        steady_state_idx = n_original  # Index of the steady-state frame we just added
        
        frame_indices = np.concatenate([
            np.arange(n_original),  # Original frames: use sequential indices
            np.full(n_new_frames, steady_state_idx)  # Extended period: all use steady-state
        ])
        
        if self.verbose:
            print(f"  Generated {n_new_frames} extended timestamps (using 1 steady-state frame)")
            print(f"  New timeline duration: {extended_timestamps[-1]/3600:.2f}h ({len(extended_timestamps)} timestamps)")
            print(f"  Actual stored frames: {len(extended_flir_frames)} (saved {n_new_frames-1} duplicate frames!)")
        
        # Update stored data
        self.flir_frames = extended_flir_frames
        self.flir_timestamps = extended_timestamps
        self.frame_indices = frame_indices  # NEW: mapping from timestamp to frame index
        
        # Re-align thermistor data to extended timestamps using forward-fill
        if self.verbose:
            print(f"  Re-aligning thermistor data to extended timestamps (forward-fill)...")
        
        adjusted_times = extended_timestamps + self.time_offset
        n_frames = len(extended_timestamps)
        n_components = len(self.thermistor_data['component_names'])
        
        self.aligned_sand_temps = np.zeros((n_frames, n_components), dtype=np.float32)
        
        for idx, comp_name in enumerate(self.thermistor_data['component_names']):
            thermistor_temps = self.thermistor_data['components'][comp_name]
            thermistor_times = self.thermistor_data['time']
            
            # DEBUG: Check source data
            if idx in [8, 16, 20]:  # Components with extreme values
                print(f"\n🔍 DEBUG Component {idx} '{comp_name}':")
                print(f"   Source temps: [{thermistor_temps.min():.2f}, {thermistor_temps.max():.2f}]°C")
                print(f"   Source time: [{thermistor_times[0]:.1f}, {thermistor_times[-1]:.1f}]s")
                print(f"   Adjusted times: [{adjusted_times[0]:.1f}, {adjusted_times[-1]:.1f}]s")
            
            # Use forward-fill for out-of-range values to avoid extrapolation artifacts
            interpolated_temps = np.zeros(n_frames, dtype=np.float32)
            
            for i, adj_time in enumerate(adjusted_times):
                if adj_time < thermistor_times[0]:
                    # Before data - use first value
                    interpolated_temps[i] = thermistor_temps[0]
                elif adj_time > thermistor_times[-1]:
                    # After data - use last value (forward fill)
                    interpolated_temps[i] = thermistor_temps[-1]
                else:
                    # Within range - linear interpolation
                    idx_after = np.searchsorted(thermistor_times, adj_time)
                    if idx_after == 0:
                        interpolated_temps[i] = thermistor_temps[0]
                    elif idx_after >= len(thermistor_times):
                        interpolated_temps[i] = thermistor_temps[-1]
                    else:
                        t0, t1 = thermistor_times[idx_after-1], thermistor_times[idx_after]
                        v0, v1 = thermistor_temps[idx_after-1], thermistor_temps[idx_after]
                        alpha = (adj_time - t0) / (t1 - t0) if (t1 - t0) > 0 else 0
                        interpolated_temps[i] = v0 + alpha * (v1 - v0)
            
            # DEBUG: Check component interpolation during extension
            if idx in [8, 16, 20]:  # Problem components
                print(f"   Comp {idx}: [{interpolated_temps.min():.2f}, {interpolated_temps.max():.2f}]°C")
                extreme_c = (interpolated_temps < 0) | (interpolated_temps > 150)
                if np.sum(extreme_c) > 0:
                    print(f"      ⚠️  {np.sum(extreme_c)} extreme values: {interpolated_temps[extreme_c][:3]}")
            
            self.aligned_sand_temps[:, idx] = interpolated_temps
        
        # DEBUG: Check all interpolated values after extension
        print(f"\n  🔍 DEBUG - After extension to {len(extended_timestamps)} timestamps:")
        print(f"     aligned_sand_temps shape: {self.aligned_sand_temps.shape}")
        print(f"     Temperature range: [{self.aligned_sand_temps.min():.2f}, {self.aligned_sand_temps.max():.2f}]°C")
        extreme_mask = (self.aligned_sand_temps < 0) | (self.aligned_sand_temps > 150)
        print(f"     Extreme values: {np.sum(extreme_mask)} / {self.aligned_sand_temps.size}")
        if np.sum(extreme_mask) > 0:
            print(f"     ⚠️  Extreme value examples: {self.aligned_sand_temps[extreme_mask][:10]}")
        
        if self.verbose:
            print(f"  ✓ Extension complete")
            print(f"  Final dataset: {len(extended_timestamps)} timestamps × {n_components} components")
            print(f"  Coverage: {extended_timestamps[-1]/3600:.2f} hours")
            percent_increase = (n_new_frames / (len(self.flir_timestamps) - 1)) * 100
            print(f"  Training samples increased by {percent_increase:.0f}%")
            storage_saved = (n_new_frames - 1) * steady_state_frame.nbytes / (1024**2)
            print(f"  Storage saved: ~{storage_saved:.0f}MB (vs duplicating frames)\n")
    
    def generate_roi_masks(self):
        """
        Generate binary ROI masks from pixel map, aligned with thermistor components.
        
        Creates roi_masks: [n_thermistor_components, H, W]
        Each mask is 1 at ROI pixels, 0 elsewhere.
        ONLY includes components that have BOTH thermistor data AND ROI pixel map entry.
        """
        if self.verbose:
            print("="*80)
            print("  STEP 6: GENERATING ROI MASKS (ALIGNED WITH THERMISTORS)")
            print("="*80)
        
        # Get thermistor component names (excluding "Time (s)" if present)
        thermistor_comp_names = [name for name in self.thermistor_data['component_names'] 
                                  if name != "Time (s)"]
        
        # Create lookup dictionary for ROI pixel map by component name
        roi_lookup = {}
        for idx, row in self.roi_pixel_map.iterrows():
            roi_lookup[row['component_name']] = row['pixel_list_parsed']
        
        # Build aligned masks - only for components with BOTH thermistor data AND ROI pixels
        n_thermistor_comps = len(thermistor_comp_names)
        self.roi_masks = np.zeros((n_thermistor_comps, self.image_height, self.image_width), dtype=np.uint8)
        
        matched_count = 0
        unmatched_thermistor = []
        
        for idx, comp_name in enumerate(thermistor_comp_names):
            if comp_name in roi_lookup:
                # Found matching ROI - create mask
                pixel_list = roi_lookup[comp_name]
                for x, y in pixel_list:
                    if 0 <= x < self.image_width and 0 <= y < self.image_height:
                        self.roi_masks[idx, y, x] = 1
                matched_count += 1
            else:
                # No ROI pixel map for this thermistor component
                unmatched_thermistor.append(comp_name)
                # Mask remains all zeros for this component
        
        if self.verbose:
            print(f"  ✓ ROI masks generated (aligned with thermistor components)")
            print(f"  Thermistor components: {n_thermistor_comps}")
            print(f"  Matched with ROI pixels: {matched_count}")
            print(f"  Missing ROI pixels: {len(unmatched_thermistor)}")
            if unmatched_thermistor:
                print(f"  Components without ROI pixels: {', '.join(unmatched_thermistor[:5])}")
                if len(unmatched_thermistor) > 5:
                    print(f"    ... and {len(unmatched_thermistor) - 5} more")
            print(f"  Mask shape: {self.roi_masks.shape}")
            print(f"  Total ROI pixels: {np.sum(self.roi_masks)}")
            print(f"  Coverage: {100 * np.sum(self.roi_masks) / (self.image_height * self.image_width):.2f}%\n")
    
    def calculate_steady_state(self):
        """
        Calculate steady-state values for both FLIR and thermistor data.
        
        Uses SteadyStateAnalyzer to find final stable temperature values.
        Stores in self.steady_state_values for later export to HDF5.
        """
        if self.verbose:
            print("="*80)
            print("  STEP 6: CALCULATING STEADY-STATE VALUES")
            print("="*80)
        
        self.steady_state_values = {
            'flir': {},
            'thermistor': {}  # From full long-duration test
        }
        
        # Calculate steady state for each component
        for comp_idx, comp_name in enumerate(self.thermistor_data['component_names']):
            if comp_name == "Time (s)":
                continue
            
            # 1. FLIR surface steady state (mean over component ROI)
            if hasattr(self, 'roi_masks') and comp_idx < len(self.roi_masks):
                mask = self.roi_masks[comp_idx]
                flir_temps = []
                for frame in self.flir_frames:
                    roi_temps = frame[mask > 0]
                    flir_temps.append(np.mean(roi_temps) if len(roi_temps) > 0 else np.nan)
                
                flir_temps = np.array(flir_temps)
                ss_flir = SteadyStateAnalyzer.steady_value(
                    self.flir_timestamps,
                    flir_temps,
                    min_dur=120.0
                )
                self.steady_state_values['flir'][comp_name] = ss_flir['value']
            
            # 2. Thermistor steady state from FULL test (36-hour baseline)
            # This provides the true embedded temperature
            if comp_name in self.thermistor_data['components']:
                full_temps = self.thermistor_data['components'][comp_name]
                full_time = self.thermistor_data['time']
                ss_therm = SteadyStateAnalyzer.steady_value(
                    full_time,
                    full_temps,
                    min_dur=600.0,  # 10 minutes for full test
                    slope_frac=0.10,
                    flat_frac=0.01
                )
                self.steady_state_values['thermistor'][comp_name] = ss_therm['value']
        
        if self.verbose:
            print(f"  ✓ Calculated steady-state values for {len(self.steady_state_values['thermistor'])} components")
            print(f"  FLIR surface: {len(self.steady_state_values['flir'])} components")
            print(f"  Thermistor (full test): {len(self.steady_state_values['thermistor'])} components")
            
            # Show example values
            if len(self.steady_state_values['thermistor']) > 0:
                comp_name = list(self.steady_state_values['thermistor'].keys())[0]
                print(f"\n  Example ({{comp_name}}):")
                print(f"    FLIR surface: {self.steady_state_values['flir'].get(comp_name, np.nan):.2f}°C")
                print(f"    Thermistor (full test): {self.steady_state_values['thermistor'][comp_name]:.2f}°C")
            print()
    
    def split_components_for_validation(self, val_split: float = 0.2):
        """
        Randomly split components into training and validation sets.
        
        Args:
            val_split: Fraction of components for validation (0.0-1.0)
        """
        if self.verbose:
            print("="*80)
            print("  SPLITTING COMPONENTS FOR VALIDATION")
            print("="*80)
        
        n_components = len(self.thermistor_data['component_names'])
        n_val = int(n_components * val_split)  # Allow 0 validation components for cross-PCB mode
        n_train = n_components - n_val
        
        # Random split with fixed seed for reproducibility
        rng = np.random.default_rng(42)
        all_indices = np.arange(n_components)
        rng.shuffle(all_indices)
        
        self.train_component_indices = sorted(all_indices[:n_train])
        self.val_component_indices = sorted(all_indices[n_train:])
        
        if self.verbose:
            train_names = [self.thermistor_data['component_names'][i] for i in self.train_component_indices]
            val_names = [self.thermistor_data['component_names'][i] for i in self.val_component_indices]
            
            print(f"  Total components: {n_components}")
            print(f"  Training components: {n_train} ({100*(1-val_split):.0f}%)")
            print(f"    {', '.join(train_names)}")
            print(f"  Validation components: {n_val} ({100*val_split:.0f}%)")
            print(f"    {', '.join(val_names)}")
            print()
    
    def export_hdf5(self, output_file: str):
        """
        Export aligned dataset to HDF5 for CNN training.
        
        Args:
            output_file: Path to output .h5 file
        """
        if self.verbose:
            print("="*80)
            print("  STEP 7: EXPORTING HDF5 DATASET")
            print("="*80)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Filter out "Time (s)" column from component names and sand_temps
        component_names_filtered = []
        component_indices = []
        
        for idx, name in enumerate(self.thermistor_data['component_names']):
            if name != "Time (s)":
                component_names_filtered.append(name)
                component_indices.append(idx)
        
        # Extract only the columns for actual components (exclude "Time (s)")
        sand_temps_filtered = self.aligned_sand_temps[:, component_indices]
        
        if self.verbose:
            print(f"  Filtering out non-component columns...")
            print(f"  Original components: {len(self.thermistor_data['component_names'])}")
            print(f"  Filtered components: {len(component_names_filtered)}")
            print(f"  Sand temps shape: {self.aligned_sand_temps.shape} → {sand_temps_filtered.shape}")
            print(f"  ROI masks shape: {self.roi_masks.shape}")
        
        with h5py.File(output_file, 'w') as f:
            # Main datasets (using filtered data)
            f.create_dataset('flir_frames', data=self.flir_frames, compression='gzip')
            f.create_dataset('sand_temps', data=sand_temps_filtered, compression='gzip')
            f.create_dataset('roi_masks', data=self.roi_masks, compression='gzip')
            f.create_dataset('timestamps', data=self.flir_timestamps)
            
            # Frame indices mapping (for extended data with steady-state reuse)
            if self.frame_indices is not None:
                f.create_dataset('frame_indices', data=self.frame_indices)
            
            # Metadata (using filtered component count)
            meta_group = f.create_group('metadata')
            meta_group.attrs['n_frames'] = len(self.flir_timestamps)
            meta_group.attrs['n_components'] = len(component_names_filtered)
            meta_group.attrs['image_width'] = self.image_width
            meta_group.attrs['image_height'] = self.image_height
            meta_group.attrs['time_offset'] = self.time_offset
            meta_group.attrs['time_start'] = self.flir_timestamps[0]
            meta_group.attrs['time_end'] = self.flir_timestamps[-1]
            
            # Store whether dataset uses frame_indices mapping
            if self.frame_indices is not None:
                meta_group.attrs['uses_frame_indices'] = True
                meta_group.attrs['actual_frame_count'] = len(self.flir_frames)
                meta_group.attrs['timestamp_count'] = len(self.flir_timestamps)
            else:
                meta_group.attrs['uses_frame_indices'] = False
            
            # Component names (store filtered list as string array)
            component_names_str = [name.encode('utf-8') for name in component_names_filtered]
            meta_group.create_dataset('component_names', data=component_names_str)
            
            # Component train/val split indices
            if hasattr(self, 'train_component_indices') and hasattr(self, 'val_component_indices'):
                meta_group.create_dataset('train_component_indices', data=self.train_component_indices)
                meta_group.create_dataset('val_component_indices', data=self.val_component_indices)
                meta_group.attrs['component_val_split'] = len(self.val_component_indices) / len(component_names_filtered)
            
            # Normalization statistics (CACHED for memory-efficient training)
            # Compute min/max for FLIR frames
            print("  Computing normalization statistics...")
            flir_min = float(np.min(self.flir_frames))
            flir_max = float(np.max(self.flir_frames))
            sand_min = float(np.min(sand_temps_filtered))
            sand_max = float(np.max(sand_temps_filtered))
            time_min = float(self.flir_timestamps.min())
            time_max = float(self.flir_timestamps.max())
            
            # Store in metadata for fast access during training
            meta_group.create_dataset('flir_min', data=flir_min)
            meta_group.create_dataset('flir_max', data=flir_max)
            meta_group.create_dataset('sand_min', data=sand_min)
            meta_group.create_dataset('sand_max', data=sand_max)
            meta_group.create_dataset('time_min', data=time_min)
            meta_group.create_dataset('time_max', data=time_max)
            
            if self.verbose:
                print(f"    FLIR range: [{flir_min:.2f}, {flir_max:.2f}]°C")
                print(f"    Sand temp range: [{sand_min:.2f}, {sand_max:.2f}]°C")
                print(f"    Time range: [{time_min:.1f}, {time_max:.1f}]s")
            
            # ROI pixel map (store as DataFrame CSV in string format)
            roi_csv_str = self.roi_pixel_map.to_csv(index=False)
            meta_group.create_dataset('roi_pixel_map_csv', data=roi_csv_str.encode('utf-8'))
            
            # Steady-state values (if calculated)
            if hasattr(self, 'steady_state_values'):
                ss_group = meta_group.create_group('steady_state')
                
                # FLIR surface steady states
                for comp_name, value in self.steady_state_values['flir'].items():
                    ss_group.attrs[f'flir_{comp_name}'] = value
                
                # Thermistor steady states (from full test)
                for comp_name, value in self.steady_state_values['thermistor'].items():
                    ss_group.attrs[f'thermistor_{comp_name}'] = value
        
        if self.verbose:
            file_size_mb = Path(output_file).stat().st_size / 1e6
            print(f"  ✓ HDF5 dataset exported")
            print(f"  File: {output_file}")
            print(f"  Size: {file_size_mb:.1f} MB")
            print(f"  Datasets: flir_frames, sand_temps, roi_masks, timestamps, metadata")
            print(f"  Components in HDF5: {len(component_names_filtered)} (excluded 'Time (s)')\n")
        
        # Export debug report if outliers were detected
        if hasattr(self, 'outlier_records') and self.outlier_records:
            self.export_debug_report(output_file)
    
    def export_debug_report(self, dataset_file: str):
        """
        Export debug report CSV with outlier information.
        
        Args:
            dataset_file: Path to the HDF5 dataset (used to determine output path)
        """
        # Save to outputs/debug_report.csv
        outputs_dir = Path(__file__).parent / 'outputs'
        outputs_dir.mkdir(exist_ok=True)
        debug_csv = outputs_dir / 'debug_report.csv'
        
        if self.outlier_records:
            df = pd.DataFrame(self.outlier_records)
            df.to_csv(debug_csv, index=False)
            
            if self.verbose:
                print("="*80)
                print("  DEBUG REPORT: OUTLIER DETECTION")
                print("="*80)
                print(f"  Found {len(self.outlier_records)} components with extreme values")
                print(f"  Report saved to: {debug_csv}")
                print(f"\n  Summary:")
                print(f"  - Total components with outliers: {len(self.outlier_records)}")
                print(f"  - Temperature range: {df['min_temp'].min():.1f}°C to {df['max_temp'].max():.1f}°C")
                print(f"  - Total outlier samples: {df['outlier_count'].sum()}")
                print()

    def build_dataset(self, flir_folder: str, thermistor_csv: str, 
                     pixel_map_csv: str, output_h5: str,
                     reference_component: Optional[str] = None,
                     max_frames: Optional[int] = None,
                     time_column: str = 'Time (s)',
                     component_prefix: str = 'Temp_',
                     manual_time_offset: Optional[float] = None,
                     component_val_split: float = 0.2):
        """
        Complete end-to-end dataset building pipeline.
        
        Args:
            flir_folder: Path to FLIR frames folder or .npz
            thermistor_csv: Path to thermistor CSV
            pixel_map_csv: Path to ROI pixel map CSV
            output_h5: Path to output HDF5 file
            reference_component: Component for time alignment (None = use mean)
            max_frames: Max frames to load (None = all)
            time_column: Name of time column in thermistor CSV
            component_prefix: Prefix for component temperature columns
            manual_time_offset: Manual time offset in seconds (positive = FLIR leads)
                               If None, uses cross-correlation
            component_val_split: Fraction of components to reserve for validation (0.0-1.0)
        """
        print("\n" + "="*80)
        print("  CNN TRAINING DATASET BUILDER")
        print("="*80 + "\n")
        
        # Execute pipeline
        self.load_flir_frames(flir_folder, max_frames=max_frames)
        self.load_thermistor_data(thermistor_csv, time_column=time_column, 
                                 component_prefix=component_prefix)
        self.load_roi_pixel_map(pixel_map_csv)
        self.detect_time_offset(reference_component=reference_component, manual_offset=manual_time_offset)
        self.align_and_interpolate()
        self.extend_flir_to_match_thermistor(flir_folder)  # Pass flir_folder to save CSV
        self.generate_roi_masks()
        self.split_components_for_validation(component_val_split)
        self.export_hdf5(output_h5)
        
        print("="*80)
        print("  DATASET BUILDING COMPLETE ✓")
        print("="*80 + "\n")
        
        return output_h5


def load_dataset(h5_file: str) -> Dict:
    """
    Load HDF5 dataset for training.
    
    Args:
        h5_file: Path to HDF5 file
    
    Returns:
        Dictionary with all dataset components
    """
    with h5py.File(h5_file, 'r') as f:
        dataset = {
            'flir_frames': f['flir_frames'][:],
            'sand_temps': f['sand_temps'][:],
            'roi_masks': f['roi_masks'][:],
            'timestamps': f['timestamps'][:],
            'metadata': {
                'n_frames': f['metadata'].attrs['n_frames'],
                'n_components': f['metadata'].attrs['n_components'],
                'image_width': f['metadata'].attrs['image_width'],
                'image_height': f['metadata'].attrs['image_height'],
                'time_offset': f['metadata'].attrs['time_offset'],
                'time_start': f['metadata'].attrs['time_start'],
                'time_end': f['metadata'].attrs['time_end'],
                'component_names': [name.decode('utf-8') for name in f['metadata/component_names'][:]],
            }
        }
        
        # Load frame_indices if present (for extended datasets with steady-state reuse)
        if 'frame_indices' in f:
            dataset['frame_indices'] = f['frame_indices'][:]
            dataset['metadata']['uses_frame_indices'] = f['metadata'].attrs.get('uses_frame_indices', True)
            dataset['metadata']['actual_frame_count'] = f['metadata'].attrs.get('actual_frame_count', len(dataset['flir_frames']))
        else:
            dataset['frame_indices'] = None
            dataset['metadata']['uses_frame_indices'] = False
    
    return dataset


def main():
    """
    Example usage: Build CNN training datasets for HBridge and LoadShedding.
    """
    
    # HBridge Dataset
    print("\n" + "="*80)
    print("  BUILDING HBRIDGE CNN DATASET")
    print("="*80 + "\n")
    
    preprocessor_hb = CNNDataPreprocessor(verbose=True)
    
    preprocessor_hb.build_dataset(
        flir_folder="outputs/flir_frames/HBridge_thermal_frames.npz",
        thermistor_csv="outputs/phase6_calibration/HBridge_thermistor_data.csv",
        pixel_map_csv="outputs/roi_pixel_maps/HBridge_pixel_map.csv",
        output_h5="ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5",
        reference_component=None,  # Use mean of all components
        max_frames=None  # Use all frames
    )
    
    # LoadShedding Dataset
    print("\n" + "="*80)
    print("  BUILDING LOADSHEDDING CNN DATASET")
    print("="*80 + "\n")
    
    preprocessor_ls = CNNDataPreprocessor(verbose=True)
    
    preprocessor_ls.build_dataset(
        flir_folder="outputs/flir_frames/LoadShedding_thermal_frames.npz",
        thermistor_csv="outputs/phase6_calibration/LoadShedding_thermistor_data.csv",
        pixel_map_csv="outputs/roi_pixel_maps/LoadShedding_pixel_map.csv",
        output_h5="ml_model/cnn_thermal_modeling/datasets/LoadShedding_cnn_dataset.h5",
        reference_component=None,
        max_frames=None
    )
    
    print("\n" + "="*80)
    print("  ALL DATASETS BUILT SUCCESSFULLY ✓")
    print("="*80 + "\n")
    
    # Test loading
    print("Testing dataset loading...")
    dataset = load_dataset("ml_model/cnn_thermal_modeling/datasets/HBridge_cnn_dataset.h5")
    print(f"  ✓ Loaded HBridge dataset: {dataset['flir_frames'].shape}")


if __name__ == "__main__":
    main()
